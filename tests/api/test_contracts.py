import pytest

from helpers import ERROR_CASES, is_uuid4
from schemas import AskSuccess, ErrorBody, UploadSuccess

pytestmark = pytest.mark.api

ASK_FIXTURES = ["in_scope_ask_response", "out_of_scope_ask_response"]
FORBIDDEN_ANSWER_CHARS = "*_#"


@pytest.mark.slow
def test_upload_success_matches_schema(shared_upload):
    """Protects: the /upload 200 body has exactly the documented fields and types,
    and session_id is a UUID4."""
    assert shared_upload.status_code == 200
    body = UploadSuccess.model_validate(shared_upload.json())
    assert is_uuid4(str(body.session_id))


@pytest.mark.slow
@pytest.mark.parametrize("fixture_name", ASK_FIXTURES)
def test_ask_success_matches_schema(request, fixture_name):
    """Protects: the /ask 200 body has exactly the documented fields and types."""
    resp = request.getfixturevalue(fixture_name)
    assert resp.status_code == 200, resp.text
    AskSuccess.model_validate(resp.json())


@pytest.mark.slow
@pytest.mark.parametrize("fixture_name", ASK_FIXTURES)
def test_ask_answerable_and_sources_are_consistent(request, fixture_name):
    """Protects: the invariant answerable=true <=> sources non-empty, and answerable=false
    means sources == []."""
    body = AskSuccess.model_validate(request.getfixturevalue(fixture_name).json())
    if body.answerable:
        assert body.sources, "answerable=true must have at least one source"
    else:
        assert body.sources == [], "answerable=false must have no sources"


@pytest.mark.slow
@pytest.mark.parametrize("fixture_name", ASK_FIXTURES)
def test_ask_answer_has_no_forbidden_characters(request, fixture_name):
    """Protects: the answer never contains * _ # (the prompt forbids them, and the
    frontend shows the text as plain text)."""
    body = AskSuccess.model_validate(request.getfixturevalue(fixture_name).json())
    found = [c for c in FORBIDDEN_ANSWER_CHARS if c in body.answer]
    assert not found, f"Answer contains forbidden characters {found}: {body.answer!r}"


@pytest.mark.parametrize("case", list(ERROR_CASES))
def test_error_response_matches_contract(error_responses, case):
    """Protects: every documented error returns its documented status and a JSON body
    that is exactly {"detail": <string>}. Free: no model calls."""
    resp = error_responses[case]
    assert resp.status_code == ERROR_CASES[case], f"{case}: {resp.status_code} {resp.text[:300]}"
    ErrorBody.model_validate(resp.json())