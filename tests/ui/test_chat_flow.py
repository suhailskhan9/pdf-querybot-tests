import pytest
from playwright.sync_api import expect

from helpers import NOT_AVAILABLE_ANSWER, json_fulfill, mock_ask

pytestmark = pytest.mark.ui

QUESTION = "Who founded the Zorblat project?"
MOCK_SOURCE = "playwright_mock.pdf"
ANSWERABLE_BODY = {
    "answer": "Mocked answer from the playwright test.",
    "answerable": True,
    "sources": [{"source": MOCK_SOURCE, "page": 1}],
}
NOT_ANSWERABLE_BODY = {"answer": NOT_AVAILABLE_ANSWER, "answerable": False, "sources": []}


def _send(page, question: str = QUESTION):
    page.get_by_test_id("question-input").fill(question)
    page.get_by_test_id("send-button").click()


def test_answerable_reply_shows_answer_and_sources(uploaded_page):
    """Protects: the question and answer are shown, the sources list shows the cited
    file, and the input is cleared right after sending."""
    page = uploaded_page
    mock_ask(page, ANSWERABLE_BODY)

    _send(page)

    expect(page.get_by_test_id("user-message")).to_contain_text(QUESTION)
    expect(page.get_by_test_id("ai-message")).to_contain_text(ANSWERABLE_BODY["answer"])
    expect(page.get_by_test_id("sources")).to_contain_text(MOCK_SOURCE)
    expect(page.get_by_test_id("question-input")).to_have_value("")


def test_unanswerable_reply_shows_no_sources(uploaded_page):
    """Protects: when answerable is false the 'not available' answer is shown and no
    sources element is rendered."""
    page = uploaded_page
    mock_ask(page, NOT_ANSWERABLE_BODY)

    _send(page, "What is the capital of Australia?")

    expect(page.get_by_test_id("ai-message")).to_contain_text(NOT_AVAILABLE_ANSWER)
    expect(page.get_by_test_id("sources")).to_have_count(0)


def test_double_submit_sends_only_one_request(uploaded_page):
    """Protects: Enter twice plus a click on send while a request is in flight still
    produces exactly one /ask request, and the input is disabled meanwhile."""
    page = uploaded_page
    held = []  # routes we have not answered yet = requests still in flight
    calls = mock_ask(page, ANSWERABLE_BODY, hold=held)

    box = page.get_by_test_id("question-input")
    box.fill(QUESTION)
    page.keyboard.press("Enter")
    page.keyboard.press("Enter")
    page.get_by_test_id("send-button").click(force=True)  # force: do not wait for it to be enabled
    page.wait_for_timeout(500)  # give any (buggy) second request time to show up

    assert len(calls) == 1, f"Expected 1 /ask request, got {len(calls)}"
    expect(box).to_be_disabled()

    json_fulfill(held[0], 200, ANSWERABLE_BODY)  # release the held response
    expect(page.get_by_test_id("ai-message")).to_have_count(1)
    expect(box).to_be_enabled()


def test_expired_session_shows_upload_again_message(uploaded_page):
    """Protects: a 404 from /ask (session gone) shows the 'documents expired' message
    instead of a generic error."""
    page = uploaded_page
    mock_ask(page, {"detail": "Session not found"}, status=404)

    _send(page)

    expect(page.get_by_test_id("status-message")).to_contain_text(
        "Your documents have expired. Please upload them again."
    )

