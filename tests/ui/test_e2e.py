import json
import re

import pytest
from playwright.sync_api import expect

from helpers import KEY_FACT_KEYWORD

# Deliberately NOT marked ui: this file's tests hit the real backend and model.
pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def test_real_upload_and_question_end_to_end(page, frontend_url, sample_pdf, api):
    """Protects: the whole chain browser -> frontend -> backend -> Gemini works with a real
    PDF. Costs 1 upload and 1 LLM call. Needs VITE_API_URL to point at the real backend."""
    page.goto(frontend_url)
    page.get_by_test_id("file-input").set_input_files(str(sample_pdf))
    expect(page.get_by_test_id("file-name")).to_be_visible(timeout=120_000)
    session_id = json.loads(page.evaluate("sessionStorage.getItem('pdf_session')"))["id"]
    try:
        page.get_by_test_id("question-input").fill("Who founded the Zorblat project?")
        page.get_by_test_id("send-button").click()
        expect(page.get_by_test_id("ai-message")).to_contain_text(
            re.compile(KEY_FACT_KEYWORD, re.IGNORECASE), timeout=120_000
        )
        expect(page.get_by_test_id("sources")).to_be_visible()
    finally:
        api.delete(f"/session/{session_id}")  # tidy up the real session