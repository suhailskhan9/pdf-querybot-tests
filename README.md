# PDF Q&A Chatbot: Automated Test Suite

Automated API and browser tests for a PDF question-answering chatbot (FastAPI backend on Render, React frontend on Vercel). Built with Python, pytest and Playwright.

The tests use only the app's public HTTP API and its web page. They never read or change the app's source code.

## Problem

The chatbot depends on a live AI model, so manual checking is slow, costs quota and misses edge cases. Typical failures:

- Bad input (wrong file type, huge file, corrupt PDF, empty question) returns a crash or an unclear error instead of a clean one.
- The AI answers a question the PDF cannot answer, or returns an answer with no sources.
- The frontend and backend drift apart: a changed field name or status code silently breaks the UI.
- The page lets a user send the same question twice, or shows the wrong message when a session expires.

This suite catches those problems in minutes, and keeps real AI usage to a minimum.

## Results

| Group | Command | Tests | AI usage |
|---|---|---|---|
| API (free) | `pytest -m "api and not slow"` | 24 | None |
| UI (mocked) | `pytest -m ui` | 5 | None |
| Slow (real AI) | `pytest -m slow` | 12 | 3 uploads, 3 answer calls |


## Quick start

Requirements: Python 3.10 to 3.13 (3.11 or 3.12 recommended). You do not need the Google API key. Only the backend uses it.

**Windows (cmd)**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
playwright install chromium
copy .env.example .env
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

Then put your app URLs in `.env` (no trailing slash):
```
BASE_URL=https://your-backend.onrender.com
FRONTEND_URL=https://your-app.vercel.app
```
For a local app use `http://localhost:8000` and `http://localhost:5173`.

## Running the tests

```bat
pytest -m "api and not slow"
pytest -m ui
pytest -m slow
pytest -m e2e
```

- Add `--headed` to watch the browser: `pytest -m ui --headed`.
- The e2e test is skipped by default. `-m e2e` and `-m slow` run it.
- Tests marked `slow` use real Gemini quota. Run them on purpose.

## What is covered

- **Health:** the backend is up.
- **Upload:** wrong file type, more than 5 files, over 20 MB, image-only, corrupt, and a valid PDF returning a UUID session id.
- **Ask:** invalid session id, unknown session, empty, whitespace and over-length questions, an answerable question (with sources and a key fact), and an unanswerable one (no sources).
- **Sessions:** delete then ask returns 404, delete with an invalid id returns 400, deleting an unknown id returns 200.
- **Contracts:** every success and error body matches its schema. Answers never contain `*`, `_` or `#`.
- **UI:** upload flow, answer and sources display, input cleared after sending, double-submit guard, expired-session message.
- **E2E:** one real upload and question through the browser.

## Project layout

```
conftest.py     fixtures: sample files, API client, sessions
helpers.py      constants and helper functions
schemas.py      expected response shapes
pytest.ini      markers: api, ui, slow, e2e, eval
tests/api/      health, upload, ask, sessions, contracts
tests/ui/       upload flow, chat flow, e2e
```