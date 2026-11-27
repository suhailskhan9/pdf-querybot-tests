"""Run the golden dataset against the live app and store the raw responses.

Usage (from the repo root):
    python eval\\run_eval.py                 # 1 run of every question, cache enabled
    python eval\\run_eval.py --repeats 3     # each question 3 times (measures variance)
    python eval\\run_eval.py --no-cache      # ignore the cache and call the app again

What it does
  1. Loads and validates the dataset (aborts if labels are wrong).
  2. Hashes the three sample PDFs (the "session document hash").
  3. For every (question, repeat): reuses the cached response if there is one,
     otherwise asks the app. Only HTTP 200 responses are cached; errors never are.
  4. The session is created LAZILY: if every answer comes from the cache there is
     no upload at all, so a re-run costs zero API calls.
  5. Always deletes the session at the end, even after Ctrl+C or an error.
  6. Writes eval/reports/raw_<timestamp>.json.

The cache key is (question, repeat, document hash, LLM model name, embedding model
name). The app is a black box and does not report its models, so YOU tell the
runner via LLM_MODEL and EMBEDDING_MODEL in .env. Change them whenever the app's
models change, or old cached answers will be mistaken for new ones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from eval.validate_dataset import DATASET_PATH, PDF_DIR, load_pages, validate  # noqa: E402
from helpers import DEFAULT_TIMEOUT, LLM_TIMEOUT  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVAL_DIR / "cache"
REPORTS_DIR = EVAL_DIR / "reports"
PDF_NAMES = ["doc_a.pdf", "doc_b.pdf", "doc_injection.pdf"]
WARMUP_SECONDS = 90  # a sleeping Render service can take a minute to wake up


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def compute_docs_hash(pdf_dir: Path = PDF_DIR, names: list[str] = PDF_NAMES) -> str:
    """SHA-256 over the file names and bytes of the sample PDFs."""
    h = hashlib.sha256()
    for name in sorted(names):
        h.update(name.encode())
        h.update((pdf_dir / name).read_bytes())
    return h.hexdigest()


def cache_key(question: str, repeat: int, docs_hash: str, llm_model: str, embedding_model: str) -> str:
    payload = json.dumps(
        {"q": question, "repeat": repeat, "docs": docs_hash, "llm": llm_model, "emb": embedding_model},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    """One small JSON file per key under eval/cache/."""

    def __init__(self, directory: Path = CACHE_DIR):
        self.dir = directory

    def get(self, key: str) -> Optional[dict]:
        path = self.dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["record"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None  # a damaged cache file is simply a cache miss

    def put(self, key: str, record: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{key}.json").write_text(
            json.dumps({"record": record}, ensure_ascii=False, indent=1), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# HTTP (small and explicit: no retries, so a failure shows the raw response)
# ---------------------------------------------------------------------------
def warm_up(base_url: str, max_seconds: Optional[int] = None) -> float:
    """GET /health until it answers 200. Free (no LLM call). Returns seconds waited."""
    max_seconds = WARMUP_SECONDS if max_seconds is None else max_seconds
    start = time.perf_counter()
    last = "no response"
    while time.perf_counter() - start < max_seconds:
        try:
            r = requests.get(f"{base_url}/health", timeout=DEFAULT_TIMEOUT)
            if r.status_code == 200:
                return time.perf_counter() - start
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = type(exc).__name__
        time.sleep(3)
    raise RuntimeError(f"Backend at {base_url} not healthy after {max_seconds}s ({last})")


def upload_pdfs(base_url: str, pdf_dir: Path = PDF_DIR, names: list[str] = PDF_NAMES) -> str:
    handles = [open(pdf_dir / n, "rb") for n in names]
    try:
        files = [("files", (n, h, "application/pdf")) for n, h in zip(names, handles)]
        resp = requests.post(f"{base_url}/upload", files=files, timeout=LLM_TIMEOUT)
    finally:
        for h in handles:
            h.close()
    if resp.status_code != 200:
        raise RuntimeError(f"Upload failed: HTTP {resp.status_code} {resp.text[:300]}")
    return resp.json()["session_id"]


def ask_once(base_url: str, session_id: str, question: str, timeout: int) -> dict:
    """One POST /ask. Never raises: failures are returned inside the record."""
    start = time.perf_counter()
    status, body, error = None, None, None
    try:
        resp = requests.post(
            f"{base_url}/ask", json={"question": question, "session_id": session_id}, timeout=timeout
        )
        status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            error = f"response was not JSON: {resp.text[:200]!r}"
    except requests.RequestException as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "status_code": status,
        "response": body,
        "latency_s": round(time.perf_counter() - start, 3),
        "error": error,
    }


def git_commit() -> Optional[str]:
    """Commit of THIS test repo (the app under test is a black box)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    path = directory / f"{stem}{suffix}"
    n = 2
    while path.exists():
        path = directory / f"{stem}_{n}{suffix}"
        n += 1
    return path


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def is_request_failure(record: dict) -> bool:
    """Infrastructure-type failure: no response, or any HTTP status other than 200."""
    return record.get("status_code") != 200


def failure_detail(record: dict) -> str:
    if record.get("error"):
        return str(record["error"])[:160]
    body = record.get("response")
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])[:160]
    return ""


def run_evaluation(
    base_url: str,
    dataset: list[dict],
    repeats: int = 1,
    use_cache: bool = True,
    llm_model: str = "unknown",
    embedding_model: str = "unknown",
    delay: float = 0.0,
    ask_timeout: int = LLM_TIMEOUT,
    max_live_calls: Optional[int] = None,
    max_consecutive_errors: int = 3,
    cache_dir: Path = CACHE_DIR,
    reports_dir: Path = REPORTS_DIR,
    pdf_dir: Path = PDF_DIR,
    log=print,
) -> tuple[dict, Path]:
    """Run every question `repeats` times. Returns (raw_run_dict, path_of_raw_file).

    max_live_calls: stop before making more than this many live /ask calls (a quota
        guard). Cached answers do not count. Re-run later to continue.
    max_consecutive_errors: stop after this many live calls in a row fail with a
        request failure (HTTP != 200 or no response), e.g. quota exhausted.
        0 disables the check.
    """
    base_url = base_url.rstrip("/")
    docs_hash = compute_docs_hash(pdf_dir)
    cache = ResponseCache(cache_dir)
    records: list[dict] = []
    session_id: Optional[str] = None
    warmed = False
    live_calls = cache_hits = consecutive_errors = 0
    stopped_reason: Optional[str] = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = [(repeat, entry) for repeat in range(repeats) for entry in dataset]
    total = len(plan)

    try:
        for repeat, entry in plan:
            key = cache_key(entry["question"], repeat, docs_hash, llm_model, embedding_model)
            n = len(records) + 1
            cached = cache.get(key) if use_cache else None

            if cached is not None:
                record = {**cached, "cached": True}
                cache_hits += 1
                source = "cache"
            else:
                if max_live_calls is not None and live_calls >= max_live_calls:
                    stopped_reason = "budget"
                    log(f"Live-call budget of {max_live_calls} reached. Stopping; re-run later to continue "
                        f"(cached answers are reused).")
                    break
                if session_id is None:  # first live call: wake the app up and upload once
                    if not warmed:
                        waited = warm_up(base_url)
                        warmed = True
                        log(f"Backend healthy (waited {waited:.1f}s). Uploading {len(PDF_NAMES)} PDFs...")
                    session_id = upload_pdfs(base_url, pdf_dir)
                result = ask_once(base_url, session_id, entry["question"], ask_timeout)
                record = {"id": entry["id"], "repeat": repeat, "question": entry["question"], **result,
                          "cached": False}
                live_calls += 1
                source = "live"
                if result["status_code"] == 200 and result["response"] is not None:
                    cache.put(key, {k: v for k, v in record.items() if k != "cached"})
                consecutive_errors = consecutive_errors + 1 if is_request_failure(record) else 0
                if delay:
                    time.sleep(delay)

            record.setdefault("id", entry["id"])
            record.setdefault("repeat", repeat)
            record.setdefault("question", entry["question"])
            records.append(record)
            resp = record.get("response") or {}
            if record.get("status_code") == 200:
                state = f"answerable={resp.get('answerable')}"
            else:
                state = f"ERROR status={record.get('status_code')} {failure_detail(record)}".rstrip()
            lat = record.get("latency_s")
            log(f"[{n:>3}/{total}] {entry['id']} r{repeat} {source:<5} "
                f"{(f'{lat:.1f}s' if lat is not None else '-'):>6}  {state}")

            if source == "live" and max_consecutive_errors and consecutive_errors >= max_consecutive_errors:
                stopped_reason = "consecutive_errors"
                log(f"{consecutive_errors} live calls in a row failed. Stopping to avoid wasting time or quota. "
                    f"Check the backend log (for example a Gemini quota error), then re-run: "
                    f"successful answers are cached, failed ones will be retried.")
                break
    except KeyboardInterrupt:
        stopped_reason = "interrupted"
        log("Interrupted. Saving what was collected (finished calls are cached, so a re-run resumes).")
    finally:
        deleted = None
        if session_id is not None:
            try:
                r = requests.delete(f"{base_url}/session/{session_id}", timeout=DEFAULT_TIMEOUT)
                deleted = r.status_code == 200
            except requests.RequestException:
                deleted = False
            log(f"Session deleted: {deleted}")

    raw = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "base_url": base_url,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "docs_hash": docs_hash,
            "repeats": repeats,
            "use_cache": use_cache,
            "live_calls": live_calls,
            "cache_hits": cache_hits,
            "calls_planned": total,
            "calls_recorded": len(records),
            "session_created": session_id is not None,
            "session_deleted": deleted,
            "stopped_reason": stopped_reason,
            "git_commit": git_commit(),
            "n_questions": len(dataset),
        },
        "records": records,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(reports_dir, f"raw_{timestamp}", ".json")
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    return raw, path


def latest_raw(reports_dir: Path = REPORTS_DIR) -> Optional[Path]:
    files = sorted(reports_dir.glob("raw_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_dataset() -> list[dict]:
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    errors, _ = validate(raw, load_pages())
    if errors:
        raise SystemExit("Dataset has labelling problems, fix them first:\n  - " + "\n  - ".join(errors))
    return raw


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run the golden dataset against the live app.")
    parser.add_argument("--repeats", type=int, default=1, help="runs per question (default 1)")
    parser.add_argument("--no-cache", action="store_true", help="ignore cached responses and call the app")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait between live calls")
    parser.add_argument("--ask-timeout", type=int, default=LLM_TIMEOUT, help="timeout per /ask call in seconds")
    parser.add_argument("--max-live-calls", type=int, default=None,
                        help="stop after this many live /ask calls (protects a small daily quota; cached answers are free)")
    parser.add_argument("--max-consecutive-errors", type=int, default=3,
                        help="stop after this many failed live calls in a row (default 3, 0 = never stop)")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8000"))
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    llm = os.getenv("LLM_MODEL", "").strip()
    emb = os.getenv("EMBEDDING_MODEL", "").strip()
    if not llm or not emb:
        print("WARNING: LLM_MODEL and/or EMBEDDING_MODEL not set in .env. The app does not report its models,\n"
              "         so cached answers cannot be told apart if you change models. Set both in .env.\n")
    llm, emb = llm or "unknown", emb or "unknown"

    dataset = load_dataset()
    print(f"Target: {args.base_url}  models: llm={llm} embedding={emb}")
    print(f"{len(dataset)} questions x {args.repeats} repeat(s), cache {'OFF' if args.no_cache else 'ON'}\n")

    raw, path = run_evaluation(
        args.base_url, dataset, repeats=args.repeats, use_cache=not args.no_cache,
        llm_model=llm, embedding_model=emb, delay=args.delay, ask_timeout=args.ask_timeout,
        max_live_calls=args.max_live_calls, max_consecutive_errors=args.max_consecutive_errors,
    )
    m = raw["meta"]
    print(f"\nLive calls: {m['live_calls']}  Cache hits: {m['cache_hits']}  "
          f"Recorded {m['calls_recorded']}/{m['calls_planned']}  Raw results: {path}")
    failed = sum(1 for r in raw["records"] if is_request_failure(r))
    if failed or m["stopped_reason"]:
        print(f"WARNING: INCOMPLETE RUN. {failed} call(s) failed and {m['calls_planned'] - m['calls_recorded']} "
              f"were not run (stopped: {m['stopped_reason']}). Metrics below cover only the calls that answered.")

    from eval.report import build_report, render_console, write_report

    report = build_report(raw, dataset)
    print("\n" + render_console(report))
    md_path, json_path = write_report(report, path)
    print(f"\nWritten: {md_path}\n         {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
