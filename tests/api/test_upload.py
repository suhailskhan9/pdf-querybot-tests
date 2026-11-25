import pytest

from helpers import is_uuid4

pytestmark = pytest.mark.api


def _assert_error(resp, status: int):
    """Shared check: expected status and a JSON body with a string 'detail'."""
    assert resp.status_code == status, f"Expected {status}, got {resp.status_code}: {resp.text[:300]}"
    assert isinstance(resp.json().get("detail"), str)


def test_upload_rejects_non_pdf_with_400(api, text_file):
    """Protects: only .pdf files are accepted; a .txt is rejected before any model call."""
    _assert_error(api.upload([text_file]), 400)


def test_upload_rejects_six_files_with_400(api, six_tiny_pdfs):
    """Protects: the 5-file limit; the count check runs before any model call."""
    _assert_error(api.upload(six_tiny_pdfs), 400)


def test_upload_rejects_oversized_file_with_413(api, oversized_pdf):
    """Protects: the 20 MB per-file limit; the size check runs before any model call."""
    _assert_error(api.upload([oversized_pdf], timeout=60), 413)


def test_upload_rejects_image_only_pdf_with_422(api, image_only_pdf):
    """Protects: PDFs with no extractable text are refused with 422 instead of
    producing an empty, useless session."""
    _assert_error(api.upload([image_only_pdf]), 422)


def test_upload_rejects_corrupt_pdf_with_400(api, corrupt_pdf):
    """Protects: random bytes with a .pdf name are rejected as corrupt, not a 500."""
    _assert_error(api.upload([corrupt_pdf]), 400)


@pytest.mark.slow
def test_upload_valid_pdf_returns_uuid_session(shared_session):
    """Protects: a valid PDF upload succeeds and returns a UUID4 session_id.
    Reuses the shared session's single real upload, so it adds no extra model usage."""
    assert is_uuid4(shared_session)