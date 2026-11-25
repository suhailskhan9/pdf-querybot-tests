"""Tiny pure helper functions shared by tests. Kept out of conftest.py because
conftest.py is for fixtures and should not be imported directly."""
import uuid


def random_uuid() -> str:
    """A valid UUID4 that matches no session. Safe for error-path tests that cost nothing."""
    return str(uuid.uuid4())


def is_uuid4(value: str) -> bool:
    """True if value parses as a UUID and is version 4."""
    try:
        return uuid.UUID(value).version == 4
    except (ValueError, AttributeError, TypeError):
        return False

# Constants shared by fixtures and tests
SAMPLE_PDF_NAME = "zorblat_facts.pdf"
KEY_FACT_KEYWORD = "Okonkwo"  # answer to "Who founded the Zorblat project?"
SECOND_FACT_KEYWORD = "Quillham"

DEFAULT_TIMEOUT = 30  # normal calls (health, upload errors, validation errors)
LLM_TIMEOUT = 120  # calls that reach Gemini (embeddings on upload, answers on /ask)

NOT_AVAILABLE_ANSWER = "The answer is not available in the provided context"

IN_SCOPE_QUESTION = "Who founded the Zorblat project?"
OUT_OF_SCOPE_QUESTION = "What is the capital of Australia?"

# Every documented error response we can trigger for free: case name -> expected status
ERROR_CASES = {
    "upload_non_pdf": 400,
    "upload_six_files": 400,
    "upload_oversized": 413,
    "upload_image_only": 422,
    "upload_corrupt": 400,
    "ask_invalid_session": 400,
    "ask_unknown_session": 404,
    "ask_empty_question": 400,
    "ask_whitespace_question": 400,
    "ask_question_too_long": 400,
    "delete_invalid_session": 400,
}

import json

MOCK_SESSION_ID = "0b6f1d52-3c1e-4c1a-9a57-6d0e7a5c2f10"  # a valid UUID4, used only in mocks

# The frontend (5173) and backend (8000) are different origins, so every mocked
# response needs CORS headers or the browser blocks it.
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def json_fulfill(route, status: int, body: dict) -> None:
    """Answer an intercepted request with a JSON body plus CORS headers."""
    route.fulfill(
        status=status,
        headers=CORS_HEADERS,
        content_type="application/json",
        body=json.dumps(body),
    )


def mock_upload(page, session_id: str = MOCK_SESSION_ID) -> list:
    """Mock POST /upload with a 200. Returns the list of requests the page sent."""
    calls = []

    def handler(route):
        if route.request.method == "OPTIONS":  # preflight, in case Playwright does not answer it
            route.fulfill(status=204, headers=CORS_HEADERS)
            return
        calls.append(route.request)
        json_fulfill(route, 200, {"message": "Files uploaded", "session_id": session_id})

    page.route("**/upload", handler)
    return calls


def mock_ask(page, body: dict, status: int = 200, hold: list | None = None) -> list:
    """Mock POST /ask. Returns the list of requests the page sent.
    If `hold` is a list, routes are appended to it and NOT answered until the test
    calls json_fulfill on them, which keeps the request 'in flight'."""
    calls = []

    def handler(route):
        if route.request.method == "OPTIONS":  # the JSON POST triggers a CORS preflight
            route.fulfill(status=204, headers=CORS_HEADERS)
            return
        calls.append(route.request)
        if hold is not None:
            hold.append(route)
            return
        json_fulfill(route, status, body)

    page.route("**/ask", handler)
    return calls