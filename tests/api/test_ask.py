import pytest

from helpers import (
    IN_SCOPE_QUESTION,
    KEY_FACT_KEYWORD,
    NOT_AVAILABLE_ANSWER,
    SAMPLE_PDF_NAME,
    random_uuid,
)

pytestmark = pytest.mark.api


def _assert_error(resp, status: int):
    """Shared check: expected status and a JSON body with a string 'detail'."""
    assert resp.status_code == status, f"Expected {status}, got {resp.status_code}: {resp.text[:300]}"
    assert isinstance(resp.json().get("detail"), str)


# ---- Error paths: no real model call, no real session needed ----

def test_ask_invalid_session_id_returns_400(api):
    """Protects: a session_id that is not a UUID is rejected with 400, not treated as 'not found'."""
    _assert_error(api.ask(IN_SCOPE_QUESTION, "not-a-uuid"), 400)


def test_ask_unknown_session_returns_404(api):
    """Protects: a well-formed UUID with no session gives 404, so the frontend can show
    the 'documents expired' message."""
    _assert_error(api.ask(IN_SCOPE_QUESTION, random_uuid()), 404)


def test_ask_empty_question_returns_400(api):
    """Protects: an empty question is rejected by validation before any session lookup."""
    _assert_error(api.ask("", random_uuid()), 400)


def test_ask_whitespace_question_returns_400(api):
    """Protects: a whitespace-only question counts as empty."""
    _assert_error(api.ask("   \n\t  ", random_uuid()), 400)


def test_ask_question_over_2000_chars_returns_400(api):
    """Protects: the 2000-character question limit (2001 characters is rejected)."""
    _assert_error(api.ask("a" * 2001, random_uuid()), 400)


# ---- Real model calls: 1 LLM call each, against the shared session ----

@pytest.mark.slow
def test_ask_in_scope_question_is_answered_with_sources(in_scope_ask_response):
    """Protects: a question the PDF can answer returns answerable=true, a non-empty
    sources list that points at the uploaded file, and an answer containing the key fact."""
    resp = in_scope_ask_response
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answerable"] is True
    assert body["sources"], "answerable=true must come with at least one source"
    assert {s["source"] for s in body["sources"]} == {SAMPLE_PDF_NAME}
    assert KEY_FACT_KEYWORD.lower() in body["answer"].lower()


@pytest.mark.slow
def test_ask_out_of_scope_question_is_not_answerable(out_of_scope_ask_response):
    """Protects: a question the PDF cannot answer returns answerable=false, no sources,
    and the fixed 'not available' answer, instead of a made-up reply."""
    resp = out_of_scope_ask_response
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answerable"] is False
    assert body["sources"] == []
    assert body["answer"] == NOT_AVAILABLE_ANSWER