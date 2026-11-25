"""Shared fixtures. Sample files are generated in code, so no binaries live in the repo."""
import random
from pathlib import Path

import pytest
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

import os

import requests
from dotenv import load_dotenv
from helpers import (
    DEFAULT_TIMEOUT,
    ERROR_CASES,
    IN_SCOPE_QUESTION,
    KEY_FACT_KEYWORD,
    LLM_TIMEOUT,
    OUT_OF_SCOPE_QUESTION,
    SAMPLE_PDF_NAME,
    SECOND_FACT_KEYWORD,
    random_uuid,
)


# Distinctive invented facts. Tests assert on these keywords, not on exact wording.
PAGE_1_TEXT = (
    "The Zorblat project was founded in 2019 by Mira Okonkwo. "
    "It started as a small research group studying tidal energy."
)
PAGE_2_TEXT = (
    "The Zorblat headquarters are located in the city of Quillham. "
    "The flagship product is called the Brindle Engine, "
    "which processes 4200 units per hour."
)

OVERSIZED_BYTES = 21 * 1024 * 1024  # above the 20 MB limit, whether MB means 1000^2 or 1024^2


def _write_text_pdf(path: Path, pages: list[str]) -> Path:
    """Write a real text PDF (one page per string) using fpdf2's built-in font."""
    pdf = FPDF()
    pdf.set_font("Helvetica", size=14)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))
    return path


def _extract_text(path: Path) -> tuple[int, str]:
    reader = PdfReader(str(path))
    return len(reader.pages), " ".join((p.extract_text() or "") for p in reader.pages)


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory) -> Path:
    """One temp directory for all generated files, created once per test run."""
    return tmp_path_factory.mktemp("samples")


@pytest.fixture(scope="session")
def sample_pdf(sample_dir) -> Path:
    """2-page text PDF with invented facts. Verified by pypdf so a broken fixture fails loudly."""
    path = _write_text_pdf(sample_dir / SAMPLE_PDF_NAME, [PAGE_1_TEXT, PAGE_2_TEXT])
    n_pages, text = _extract_text(path)
    if n_pages != 2:
        pytest.fail(f"Fixture bug: sample PDF should have 2 pages, got {n_pages}")
    for keyword in (KEY_FACT_KEYWORD, SECOND_FACT_KEYWORD, "Brindle"):
        if keyword not in text:
            pytest.fail(
                f"Fixture bug: pypdf could not extract '{keyword}' from the sample PDF. "
                f"Extracted text was: {text!r}"
            )
    return path


@pytest.fixture(scope="session")
def image_only_pdf(sample_dir) -> Path:
    """PDF whose only content is a raster image of text, so it has no extractable text (422)."""
    img = Image.new("RGB", (1240, 1754), "white")
    try:
        font = ImageFont.load_default(size=48)  # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text(
        (100, 200), "This text is only pixels, not real PDF text.", fill="black", font=font
    )
    path = sample_dir / "image_only.pdf"
    img.save(path, "PDF", resolution=150.0)

    _, text = _extract_text(path)
    if text.strip():
        pytest.fail(f"Fixture bug: image-only PDF unexpectedly has text: {text!r}")
    return path


@pytest.fixture(scope="session")
def corrupt_pdf(sample_dir) -> Path:
    """Random bytes with a .pdf name. Seeded so every run uses the same bytes."""
    path = sample_dir / "corrupt.pdf"
    path.write_bytes(random.Random(0).randbytes(2048))
    return path


@pytest.fixture(scope="session")
def text_file(sample_dir) -> Path:
    """A non-PDF file, to check the extension validation."""
    path = sample_dir / "notes.txt"
    path.write_text("This is not a PDF.\n")
    return path


@pytest.fixture(scope="session")
def oversized_pdf(sample_dir) -> Path:
    """A valid-looking PDF header followed by zero padding, just over 20 MB (413)."""
    path = sample_dir / "oversized.pdf"
    chunk = b"\0" * (1024 * 1024)
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n")
        for _ in range(OVERSIZED_BYTES // len(chunk)):
            f.write(chunk)
    return path


@pytest.fixture(scope="session")
def six_tiny_pdfs(sample_dir) -> list[Path]:
    """Six small valid PDFs, for the 'more than 5 files' test (400)."""
    paths = []
    for i in range(1, 7):
        path = _write_text_pdf(sample_dir / f"tiny_{i}.pdf", [f"Tiny document number {i}."])
        n_pages, text = _extract_text(path)
        if n_pages != 1 or f"number {i}" not in text:
            pytest.fail(f"Fixture bug: tiny PDF {i} is not valid or has no extractable text")
        paths.append(path)
    return paths

# ---------------------------------------------------------------------------
# Configuration and HTTP helper
# ---------------------------------------------------------------------------
load_dotenv()  # reads .env if present; the defaults below apply otherwise


@pytest.fixture(scope="session")
def base_url() -> str:
    """Backend URL.

    NOTE: this overrides pytest-playwright's own `base_url` fixture. UI tests must therefore
    always use full URLs built from `frontend_url` and never `page.goto("/")`.
    """
    return os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")


@pytest.fixture(scope="session")
def frontend_url() -> str:
    """Frontend URL (Vite dev server)."""
    return os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")


class ApiClient:
    """Thin wrapper over requests: builds URLs and applies sensible default timeouts.
    No retries and no hidden behaviour, so a failing test shows the raw response."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._session = requests.Session()

    def _request(self, method: str, path: str, timeout: int, **kwargs) -> requests.Response:
        return self._session.request(method, f"{self.base_url}{path}", timeout=timeout, **kwargs)

    def get(self, path: str, timeout: int = DEFAULT_TIMEOUT, **kwargs):
        return self._request("GET", path, timeout, **kwargs)

    def delete(self, path: str, timeout: int = DEFAULT_TIMEOUT, **kwargs):
        return self._request("DELETE", path, timeout, **kwargs)

    def ask(self, question: str, session_id: str, timeout: int = DEFAULT_TIMEOUT):
        """POST /ask. Pass timeout=LLM_TIMEOUT for calls that reach the real model."""
        return self._request(
            "POST", "/ask", timeout, json={"question": question, "session_id": session_id}
        )

    def upload(self, paths: list[Path], timeout: int = DEFAULT_TIMEOUT):
        """POST /upload with one or more files under the form field 'files'.
        Pass timeout=LLM_TIMEOUT when the upload will be embedded by the real model."""
        handles = [open(p, "rb") for p in paths]
        try:
            files = [("files", (p.name, h)) for p, h in zip(paths, handles)]
            return self._request("POST", "/upload", timeout, files=files)
        finally:
            for h in handles:
                h.close()


@pytest.fixture(scope="session")
def api(base_url) -> ApiClient:
    """Shared HTTP client for all API tests."""
    return ApiClient(base_url)



# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def shared_upload(api, sample_pdf):
    """ONE real upload per test run. Yields the full response so contract tests can reuse it."""
    resp = api.upload([sample_pdf], timeout=LLM_TIMEOUT)
    assert resp.status_code == 200, f"Shared upload failed: {resp.status_code} {resp.text}"
    yield resp
    api.delete(f"/session/{resp.json()['session_id']}")  # cleanup, ignored if it fails


@pytest.fixture(scope="session")
def shared_session(shared_upload) -> str:
    """The shared session's id. Tests using it must only read (ask), never delete it."""
    return shared_upload.json()["session_id"]


@pytest.fixture
def fresh_session(api, sample_pdf):
    """A brand-new session per test, for tests that destroy it (delete, then ask)."""
    resp = api.upload([sample_pdf], timeout=LLM_TIMEOUT)
    assert resp.status_code == 200, f"Fresh upload failed: {resp.status_code} {resp.text}"
    session_id = resp.json()["session_id"]
    yield session_id
    api.delete(f"/session/{session_id}")  # harmless if the test already deleted it


# ---------------------------------------------------------------------------
# Cached responses, shared by test_ask.py and test_contracts.py
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def in_scope_ask_response(api, shared_session):
    """The single real /ask call for an in-scope question (1 LLM call, made once per run)."""
    return api.ask(IN_SCOPE_QUESTION, shared_session, timeout=LLM_TIMEOUT)


@pytest.fixture(scope="session")
def out_of_scope_ask_response(api, shared_session):
    """The single real /ask call for an out-of-scope question (1 LLM call, made once per run)."""
    return api.ask(OUT_OF_SCOPE_QUESTION, shared_session, timeout=LLM_TIMEOUT)


@pytest.fixture(scope="session")
def error_responses(
    api, text_file, six_tiny_pdfs, oversized_pdf, image_only_pdf, corrupt_pdf
) -> dict:
    """Every documented error response, collected once. All are free: they fail before any model call."""
    responses = {
        "upload_non_pdf": api.upload([text_file]),
        "upload_six_files": api.upload(six_tiny_pdfs),
        "upload_oversized": api.upload([oversized_pdf], timeout=60),
        "upload_image_only": api.upload([image_only_pdf]),
        "upload_corrupt": api.upload([corrupt_pdf]),
        "ask_invalid_session": api.ask(IN_SCOPE_QUESTION, "not-a-uuid"),
        "ask_unknown_session": api.ask(IN_SCOPE_QUESTION, random_uuid()),
        "ask_empty_question": api.ask("", random_uuid()),
        "ask_whitespace_question": api.ask("   \n\t  ", random_uuid()),
        "ask_question_too_long": api.ask("a" * 2001, random_uuid()),
        "delete_invalid_session": api.delete("/session/not-a-uuid"),
    }
    assert set(responses) == set(ERROR_CASES), "error_responses and ERROR_CASES are out of sync"
    return responses