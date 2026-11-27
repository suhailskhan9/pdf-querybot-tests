"""Fast tests for eval/run_eval.py against a tiny local stub server. No Gemini, no real app."""
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from eval import run_eval
from eval.run_eval import cache_key, compute_docs_hash, run_evaluation


class StubState:
    def __init__(self):
        self.uploads = 0
        self.asks: list[str] = []
        self.deletes: list[str] = []
        self.fail_questions: set[str] = set()
        self.url = ""


@pytest.fixture
def stub():
    state = StubState()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output quiet
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"status": "ok"}) if self.path == "/health" else self._send(404, {"detail": "no"})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/upload":
                state.uploads += 1
                self._send(200, {"message": "ok", "session_id": str(uuid.uuid4())})
            elif self.path == "/ask":
                question = json.loads(raw)["question"]
                state.asks.append(question)
                if question in state.fail_questions:
                    self._send(500, {"detail": "boom"})
                else:
                    self._send(200, {"answer": "stub answer 27", "answerable": True,
                                     "sources": [{"source": "doc_a.pdf", "page": 2}]})
            else:
                self._send(404, {"detail": "no"})

        def do_DELETE(self):
            state.deletes.append(self.path)
            self._send(200, {"message": "deleted"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
    thread.start()
    yield state
    server.shutdown()
    server.server_close()


DATASET = [
    {"id": f"q{i}", "category": "single_fact", "question": f"fake question number {i}", "expected_source": "doc_a.pdf",
     "expected_page": 2, "must_contain": ["27"], "should_refuse": False, "notes": ""}
    for i in (1, 2, 3)
]


def run(stub, tmp_path, **kwargs):
    return run_evaluation(
        stub.url, DATASET, cache_dir=tmp_path / "cache", reports_dir=tmp_path / "reports",
        log=lambda *_: None, **kwargs,
    )


def test_first_run_uploads_once_asks_everything_and_deletes_session(stub, tmp_path):
    raw, path = run(stub, tmp_path)
    assert stub.uploads == 1
    assert len(stub.asks) == 3
    assert len(stub.deletes) == 1
    assert raw["meta"]["live_calls"] == 3 and raw["meta"]["cache_hits"] == 0
    assert raw["meta"]["session_deleted"] is True
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [r["id"] for r in saved["records"]] == ["q1", "q2", "q3"]
    assert saved["records"][0]["status_code"] == 200
    assert saved["records"][0]["response"]["answerable"] is True
    assert saved["records"][0]["latency_s"] >= 0
    assert path.name.startswith("raw_") and path.suffix == ".json"


def test_second_run_is_served_from_cache_with_no_upload_and_no_asks(stub, tmp_path):
    first, _ = run(stub, tmp_path)
    uploads, asks, deletes = stub.uploads, len(stub.asks), len(stub.deletes)
    second, _ = run(stub, tmp_path)
    assert (stub.uploads, len(stub.asks), len(stub.deletes)) == (uploads, asks, deletes)  # nothing new happened
    assert second["meta"]["cache_hits"] == 3 and second["meta"]["live_calls"] == 0
    assert second["meta"]["session_created"] is False
    for a, b in zip(first["records"], second["records"]):
        assert a["response"] == b["response"] and a["latency_s"] == b["latency_s"]
        assert b["cached"] is True


def test_no_cache_forces_fresh_calls(stub, tmp_path):
    run(stub, tmp_path)
    run(stub, tmp_path, use_cache=False)
    assert stub.uploads == 2 and len(stub.asks) == 6


def test_repeats_are_separate_cache_entries(stub, tmp_path):
    raw, _ = run(stub, tmp_path, repeats=2)
    assert len(stub.asks) == 6
    assert sorted((r["id"], r["repeat"]) for r in raw["records"]) == sorted(
        (f"q{i}", rep) for i in (1, 2, 3) for rep in (0, 1)
    )
    run(stub, tmp_path, repeats=2)
    assert len(stub.asks) == 6  # fully cached now


def test_growing_repeats_reuses_existing_runs(stub, tmp_path):
    run(stub, tmp_path, repeats=1)
    raw, _ = run(stub, tmp_path, repeats=2)
    assert len(stub.asks) == 6          # 3 earlier + only the 3 new repeat-1 calls
    assert raw["meta"]["cache_hits"] == 3 and raw["meta"]["live_calls"] == 3


def test_errors_are_never_cached(stub, tmp_path):
    stub.fail_questions = {"fake question number 2"}
    raw, _ = run(stub, tmp_path)
    bad = next(r for r in raw["records"] if r["id"] == "q2")
    assert bad["status_code"] == 500
    stub.fail_questions = set()
    raw2, _ = run(stub, tmp_path)
    assert raw2["meta"]["cache_hits"] == 2 and raw2["meta"]["live_calls"] == 1  # only q2 was asked again
    assert next(r for r in raw2["records"] if r["id"] == "q2")["status_code"] == 200


def test_changing_model_names_invalidates_the_cache(stub, tmp_path):
    run(stub, tmp_path, llm_model="model-a", embedding_model="emb-1")
    raw, _ = run(stub, tmp_path, llm_model="model-b", embedding_model="emb-1")
    assert raw["meta"]["cache_hits"] == 0 and raw["meta"]["live_calls"] == 3
    assert raw["meta"]["llm_model"] == "model-b"


def test_cache_key_depends_on_every_input():
    base = cache_key("q", 0, "hash", "llm", "emb")
    assert base == cache_key("q", 0, "hash", "llm", "emb")
    for changed in (cache_key("q2", 0, "hash", "llm", "emb"), cache_key("q", 1, "hash", "llm", "emb"),
                    cache_key("q", 0, "hash2", "llm", "emb"), cache_key("q", 0, "hash", "llm2", "emb"),
                    cache_key("q", 0, "hash", "llm", "emb2")):
        assert changed != base


def test_docs_hash_changes_when_a_pdf_changes(tmp_path):
    for name in ("doc_a.pdf", "doc_b.pdf", "doc_injection.pdf"):
        (tmp_path / name).write_bytes(b"pdf-" + name.encode())
    before = compute_docs_hash(tmp_path)
    (tmp_path / "doc_b.pdf").write_bytes(b"changed")
    assert compute_docs_hash(tmp_path) != before


def test_interrupt_still_saves_partial_results_and_deletes_session(stub, tmp_path, monkeypatch):
    real_ask = run_eval.ask_once
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_ask(*args, **kwargs)

    monkeypatch.setattr(run_eval, "ask_once", flaky)
    raw, path = run(stub, tmp_path)
    assert raw["meta"]["stopped_reason"] == "interrupted"
    assert len(raw["records"]) == 1
    assert len(stub.deletes) == 1 and path.exists()


def test_unreachable_backend_fails_fast_with_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "WARMUP_SECONDS", 1)
    monkeypatch.setattr(run_eval.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="not healthy"):
        run_evaluation("http://127.0.0.1:9", DATASET, cache_dir=tmp_path / "c", reports_dir=tmp_path / "r",
                       log=lambda *_: None)


def test_budget_caps_live_calls_and_a_rerun_continues_from_the_cache(stub, tmp_path):
    raw, _ = run(stub, tmp_path, max_live_calls=2)
    assert len(stub.asks) == 2
    assert raw["meta"]["stopped_reason"] == "budget"
    assert raw["meta"]["calls_recorded"] == 2 and raw["meta"]["calls_planned"] == 3
    assert len(stub.deletes) == 1
    raw2, _ = run(stub, tmp_path, max_live_calls=2)          # next day: 2 cached + 1 new live call
    assert raw2["meta"]["cache_hits"] == 2 and raw2["meta"]["live_calls"] == 1
    assert raw2["meta"]["stopped_reason"] is None and len(raw2["records"]) == 3


def test_cached_answers_do_not_use_the_budget(stub, tmp_path):
    run(stub, tmp_path)
    stub.asks.clear()
    raw, _ = run(stub, tmp_path, max_live_calls=0)
    assert raw["meta"]["stopped_reason"] is None and raw["meta"]["cache_hits"] == 3
    assert stub.asks == []


def test_stops_after_consecutive_request_failures(stub, tmp_path):
    stub.fail_questions = {f"fake question number {i}" for i in (1, 2, 3)}
    raw, _ = run(stub, tmp_path, max_consecutive_errors=2)
    assert raw["meta"]["stopped_reason"] == "consecutive_errors"
    assert len(stub.asks) == 2 and len(raw["records"]) == 2
    assert len(stub.deletes) == 1


def test_a_success_resets_the_consecutive_error_counter(stub, tmp_path):
    stub.fail_questions = {"fake question number 1", "fake question number 3"}
    raw, _ = run(stub, tmp_path, max_consecutive_errors=2)
    assert raw["meta"]["stopped_reason"] is None and len(raw["records"]) == 3
