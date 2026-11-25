import pytest
from playwright.sync_api import expect
import json

from helpers import MOCK_SESSION_ID, SAMPLE_PDF_NAME, mock_upload

pytestmark = pytest.mark.ui


def test_upload_shows_file_name_and_stores_session(page, frontend_url, sample_pdf):
    """Protects: after a successful upload the file name is shown, the 'upload a document'
    warning disappears, and the session id is saved under sessionStorage['pdf_session']."""
    calls = mock_upload(page)
    page.goto(frontend_url)

    page.get_by_test_id("file-input").set_input_files(str(sample_pdf))

    expect(page.get_by_test_id("file-name")).to_contain_text(SAMPLE_PDF_NAME)
    expect(page.get_by_test_id("upload-warning")).not_to_be_visible()
    assert len(calls) == 1
    stored = json.loads(page.evaluate("sessionStorage.getItem('pdf_session')"))
    assert stored["id"] == MOCK_SESSION_ID
    assert stored["fileNames"] == [SAMPLE_PDF_NAME]