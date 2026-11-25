import pytest
from playwright.sync_api import expect

from helpers import mock_upload


@pytest.fixture
def uploaded_page(page, frontend_url, sample_pdf):
    """A page where the sample PDF has been 'uploaded' against a mocked /upload.
    Tests register their own /ask mock afterwards, before sending a question."""
    mock_upload(page)
    page.goto(frontend_url)  # full URL: never page.goto("/"), see the base_url note in conftest
    page.get_by_test_id("file-input").set_input_files(str(sample_pdf))
    expect(page.get_by_test_id("file-name")).to_be_visible()
    return page