"""Build the evaluation report from a stored raw results file. No API calls.

Usage (from the repo root):
    python eval\\report.py                          # report on the latest raw file
    python eval\\report.py eval\\reports\\raw_X.json  # report on a specific raw file

Writes eval/reports/report_<timestamp>.md and .json (same timestamp as the raw
file) and prints a readable table to the console.
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import (  # noqa: E402
    METRIC_INFO,
    MetricResult,
    compute_by_category,
    compute_metrics,
    evaluate_all,
    find_failures,
)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
CATEGORY_ORDER = ["single_fact", "multi_fact", "out_of_scope", "injection", "vague"]

LIMITATION_NOTE = (
    "Retrieval limitation: the app returns an empty `sources` list whenever it refuses, so retrieval "
    "cannot be observed for refused questions. The retrieval hit rate therefore only covers questions "
    "the model chose to answer. Read it together with the false refusal rate: a refusal may hide either "
    "a retrieval failure (the right page was not retrieved) or a generation failure (the right page was "
    "retrieved but the model refused anyway), and the API cannot tell these apart. With 9 pages in total "
    "and k=4, roughly half of all chunks are retrieved for every question, so retrieval is easier here "
    "than it would be on a large corpus."
)
SAMPLE_NOTE = (
    "Small sample: every question is worth about 4 percentage points of a 25-question rate. Differences of "
    "one or two questions between runs are noise, not regressions. Use --repeats and several runs to see "
    "the natural variance before setting thresholds."
)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _metric_dict(name: str, res: MetricResult) -> dict:
    higher, desc = METRIC_INFO[name]
    return {**res.to_dict(), "higher_is_better": higher, "description": desc}


def retrieval_summary(dataset: list[dict], results: list[dict]) -> dict:
    """Numbers that make the retrieval limitation concrete."""
    evals = [e for e in evaluate_all(dataset, results) if e.false_refusal is not None]
    answered = [e for e in evals if e.answerable]
    hits = sum(1 for e in answered if e.retrieval_hit)
    return {
        "answerable_expected_calls": len(evals),
        "answered": len(answered),
        "refused_unobservable": len(evals) - len(answered),
        "hits_among_answered": hits,
        "worst_case_hit_rate": hits / len(evals) if evals else None,
    }


def build_report(raw: dict, dataset: list[dict]) -> dict:
    records = raw["records"]
    metrics = compute_metrics(dataset, records)
    return {
        "meta": raw["meta"],
        "n_calls": len(records),
        "n_http_errors": sum(1 for r in records if r.get("status_code") != 200),
        "metrics": {k: _metric_dict(k, v) for k, v in metrics.items()},
        "by_category": {
            cat: {k: v.to_dict() for k, v in ms.items()} for cat, ms in compute_by_category(dataset, records).items()
        },
        "retrieval": retrieval_summary(dataset, records),
        "failures": find_failures(dataset, records),
        "notes": [LIMITATION_NOTE, SAMPLE_NOTE],
    }


def metrics_flat(report: dict) -> dict[str, Optional[float]]:
    """name -> value, for the baseline and regression test."""
    return {k: v["value"] for k, v in report["metrics"].items()}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def incomplete_warning(report: dict) -> Optional[str]:
    """Text when the run is not a clean, complete run (partial, failed calls, or unknown models)."""
    meta = report["meta"]
    parts = []
    planned, recorded = meta.get("calls_planned"), meta.get("calls_recorded", report["n_calls"])
    if planned is not None and recorded < planned:
        parts.append(f"only {recorded} of {planned} planned calls were run (stopped: {meta.get('stopped_reason')})")
    elif meta.get("stopped_reason"):
        parts.append(f"run stopped early: {meta['stopped_reason']}")
    if report["n_http_errors"]:
        parts.append(f"{report['n_http_errors']} call(s) failed with HTTP errors or timeouts "
                     f"(for example a Gemini quota error); they are excluded from the quality metrics")
    if not parts:
        return None
    return ("INCOMPLETE RUN: " + "; ".join(parts) + ". Do not compare these numbers with a baseline "
            "or save them as one.")


def _fmt_value(name: str, value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} s" if name.startswith("latency") else f"{value * 100:.1f}%"


def _fmt_frac(m: dict) -> str:
    if m["numerator"] is None or m["denominator"] is None:
        return f"n={m['denominator']}" if m["denominator"] is not None else ""
    return f"{m['numerator']}/{m['denominator']}"


def _cell(m: Optional[dict]) -> str:
    if m is None or m["value"] is None:
        return "-"
    return f"{m['value'] * 100:.0f}% ({m['numerator']}/{m['denominator']})"


def _safe(text, limit: int = 300) -> str:
    """Single-line, truncated, ASCII-only so a Windows console never raises UnicodeEncodeError."""
    s = re.sub(r"\s+", " ", str(text)).strip()
    if len(s) > limit:
        s = s[: limit - 3] + "..."
    return s.encode("ascii", "replace").decode("ascii")


def _categories(report: dict) -> list[str]:
    cats = report["by_category"]
    return [c for c in CATEGORY_ORDER if c in cats] + [c for c in cats if c not in CATEGORY_ORDER]


def _expected_text(f: dict) -> str:
    if f["should_refuse"]:
        return "expected: refusal (answerable=false, sources=[])"
    src, page = f["expected_source"], f["expected_page"]
    srcs = src if isinstance(src, list) else [src]
    pages = page if isinstance(page, list) else [page]
    pairs = ", ".join(f"{s} p{p}" for s, p in zip(srcs, pages))
    return f"expected: {pairs}; must_contain {f['must_contain']}"


def render_console(report: dict) -> str:
    m, meta = report["metrics"], report["meta"]
    out: list[str] = []
    line = "=" * 78
    out += [line, "EVALUATION REPORT", line]
    out.append(f"Run: {meta['timestamp']}   target: {meta['base_url']}   test-repo commit: {meta.get('git_commit') or 'n/a'}")
    out.append(f"Models: llm={meta['llm_model']}  embedding={meta['embedding_model']}")
    out.append(f"Calls: {report['n_calls']} ({meta['live_calls']} live, {meta['cache_hits']} from cache)   "
               f"repeats: {meta['repeats']}   HTTP errors: {report['n_http_errors']}")
    warning = incomplete_warning(report)
    if warning:
        out.append(textwrap.fill("WARNING: " + _safe(warning, 1000), width=78))
    if meta["llm_model"] == "unknown":
        out.append("WARNING: model names unknown (set LLM_MODEL and EMBEDDING_MODEL in .env).")

    out += ["", "OVERALL METRICS", "-" * 78, f"{'metric':<26}{'value':>10}  {'count':<10} better"]
    for name, d in m.items():
        out.append(f"{name:<26}{_fmt_value(name, d['value']):>10}  {_fmt_frac(d):<10} "
                   f"{'higher' if d['higher_is_better'] else 'lower'}")
    out.append("(vague questions are left out of refusal_accuracy; see the per-category table)")

    cats = _categories(report)
    out += ["", "BY CATEGORY", "-" * 78]
    out.append(f"{'metric':<24}" + "".join(f"{c:>16}" for c in cats))
    for name in ["retrieval_hit_rate", "must_contain_pass_rate", "refusal_accuracy", "false_refusal_rate",
                 "injection_resistance", "formatting_compliance", "contract_compliance"]:
        cells = [_cell(report["by_category"][c].get(name)) for c in cats]
        if all(c == "-" for c in cells):
            continue
        out.append(f"{name:<24}" + "".join(f"{c:>16}" for c in cells))

    r = report["retrieval"]
    out += ["", "RETRIEVAL OBSERVABILITY", "-" * 78,
            f"{r['answerable_expected_calls']} calls had an answer in the documents: {r['answered']} answered "
            f"(retrieval observable, {r['hits_among_answered']} hits), {r['refused_unobservable']} refused "
            f"(retrieval NOT observable).",
            "Worst case (every refusal counted as a retrieval miss): "
            + ("n/a" if r["worst_case_hit_rate"] is None else f"{r['worst_case_hit_rate'] * 100:.1f}%")]

    fails = report["failures"]
    out += ["", f"FAILURES ({len(fails)} call(s))", "-" * 78]
    if not fails:
        out.append("None.")
    for f in fails:
        out.append(f"{f['id']} [{f['category']}] repeat {f['repeat']}: {_safe(f['question'], 120)}")
        for reason in f["reasons"]:
            out.append(f"    - {_safe(reason, 200)}")
        out.append(f"    answerable={f['answerable']} sources={_safe(json.dumps(f['sources']), 160)}")
        out.append(f"    answer: {_safe(f['answer'])}")
        out.append(f"    {_safe(_expected_text(f), 200)}")

    out += ["", "NOTES", "-" * 78]
    out += [textwrap.fill(_safe(n, 2000), width=78) + "\n" for n in report["notes"]]
    out.append(line)
    return "\n".join(out)


def render_markdown(report: dict) -> str:
    m, meta = report["metrics"], report["meta"]
    md: list[str] = ["# Evaluation report", ""]
    md += [
        f"- **Run:** {meta['timestamp']}",
        f"- **Target:** {meta['base_url']}",
        f"- **LLM model:** {meta['llm_model']}",
        f"- **Embedding model:** {meta['embedding_model']}",
        f"- **Test-repo git commit:** {meta.get('git_commit') or 'n/a'}",
        f"- **Documents hash:** `{meta['docs_hash'][:16]}`",
        f"- **Calls:** {report['n_calls']} ({meta['live_calls']} live, {meta['cache_hits']} from cache), "
        f"repeats {meta['repeats']}, HTTP errors {report['n_http_errors']}",
    ]
    warning = incomplete_warning(report)
    if warning:
        md.append(f"- **WARNING:** {warning}")
    md += ["", "## Overall metrics", "", "| Metric | Value | Count | Better | Meaning |", "|---|---|---|---|---|"]
    for name, d in m.items():
        md.append(f"| `{name}` | {_fmt_value(name, d['value'])} | {_fmt_frac(d)} | "
                  f"{'higher' if d['higher_is_better'] else 'lower'} | {d['description']} |")
    md += ["", "Vague questions are left out of `refusal_accuracy`; see the per-category table.", ""]

    cats = _categories(report)
    md += ["## By category", "", "| Metric | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for name in ["retrieval_hit_rate", "must_contain_pass_rate", "refusal_accuracy", "false_refusal_rate",
                 "injection_resistance", "formatting_compliance", "contract_compliance"]:
        cells = [_cell(report["by_category"][c].get(name)) for c in cats]
        if all(c == "-" for c in cells):
            continue
        md.append(f"| `{name}` | " + " | ".join(cells) + " |")

    r = report["retrieval"]
    md += ["", "## Retrieval observability", "",
           f"{r['answerable_expected_calls']} calls had an answer in the documents. {r['answered']} were answered "
           f"({r['hits_among_answered']} with the expected page among the sources) and {r['refused_unobservable']} "
           f"were refused, so retrieval could not be observed for them. Worst case, counting every refusal as a "
           f"retrieval miss: "
           + ("n/a" if r["worst_case_hit_rate"] is None else f"{r['worst_case_hit_rate'] * 100:.1f}%") + ".", ""]

    fails = report["failures"]
    md += [f"## Failures ({len(fails)} call(s))", ""]
    if not fails:
        md += ["None.", ""]
    for f in fails:
        answer_text = re.sub(r"\s+", " ", f["answer"]).strip() or "(none)"
        md += [f"### {f['id']} ({f['category']}), repeat {f['repeat']}", "",
               f"- **Question:** {f['question']}",
               f"- **Why it failed:** " + "; ".join(f["reasons"]),
               f"- **answerable:** {f['answerable']}",
               f"- **sources:** `{json.dumps(f['sources'])}`",
               f"- **Answer:** {answer_text}",
               f"- **{_expected_text(f)}**", ""]

    md += ["## Notes", ""] + [f"- {n}" for n in report["notes"]]
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------
# Output files and CLI
# ---------------------------------------------------------------------------
def stamp_from_raw_path(raw_path: Path) -> str:
    match = re.match(r"raw_(.+)\.json$", raw_path.name)
    return match.group(1) if match else "manual"


def write_report(report: dict, raw_path: Path, reports_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp_from_raw_path(raw_path)
    md_path, json_path = reports_dir / f"report_{stamp}.md", reports_dir / f"report_{stamp}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return md_path, json_path


def report_from_raw_file(raw_path: Path, dataset: list[dict]) -> dict:
    return build_report(json.loads(raw_path.read_text(encoding="utf-8")), dataset)


def main(argv: Optional[list[str]] = None) -> int:
    from eval.run_eval import latest_raw, load_dataset

    args = sys.argv[1:] if argv is None else argv
    raw_path = Path(args[0]) if args else latest_raw()
    if raw_path is None or not raw_path.exists():
        print("No raw results found. Run: python eval\\run_eval.py")
        return 1
    report = report_from_raw_file(raw_path, load_dataset())
    print(render_console(report))
    md_path, json_path = write_report(report, raw_path)
    print(f"\nWritten: {md_path}\n         {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
