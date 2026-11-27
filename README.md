# PDF-QueryBot: Automated Test Suite

Automated API and browser tests for a PDF question-answering chatbot (FastAPI backend on Render, React frontend on Vercel), plus a golden-dataset evaluation of the AI answers themselves. Built with Python, pytest and Playwright.

The tests use only the app's public HTTP API and its web page. They never read or change the app's source code.

## Problem

The chatbot depends on a live AI model, so manual checking is slow, costs quota and misses edge cases. Typical failures:

- Bad input (wrong file type, huge file, corrupt PDF, empty question) returns a crash or an unclear error instead of a clean one.
- The AI answers a question the PDF cannot answer, or returns an answer with no sources.
- The AI refuses a question the PDF does answer, or obeys an instruction hidden inside an uploaded document.
- The frontend and backend drift apart: a changed field name or status code silently breaks the UI.
- The page lets a user send the same question twice, or shows the wrong message when a session expires.

This suite catches those problems in minutes, and keeps real AI usage to a minimum.

## Results

| Group | Command | Tests | AI usage |
|---|---|---|---|
| API (free) | `pytest -m "api and not slow"` | 24 | None |
| UI (mocked) | `pytest -m ui` | 5 | None |
| Slow (real AI) | `pytest -m slow` | 12 | 3 uploads, 3 answer calls |
| Eval unit tests (free) | `pytest tests\eval` | 55 | None |
| Golden dataset (real AI) | `python eval\run_eval.py` | 25 questions | 3 uploads, 25 answer calls (cached after the first run) |

Golden dataset snapshot (one run, `gemini-3.6-flash` with `gemini-embedding-2`, 27 November 2026):

| Metric | Result |
|---|---|
| Retrieval hit rate | 18/18 |
| Must-contain pass rate | 18/18 |
| Refusal accuracy | 5/5 |
| False refusal rate | 1/19 (q15, a question needing two documents) |
| Injection resistance | 3/3 |
| Formatting compliance | 25/25 |
| Contract compliance | 25/25 |

This is one run on a small dataset, so read it as "no obvious problems found", not as proof of quality.

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

For the golden dataset, also name the models your backend uses:
```
LLM_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=gemini-embedding-2
```
These are labels only: the app does not report its models, so keep them equal to what the backend really runs.

## Running the tests

```bat
pytest -m "api and not slow"
pytest -m ui
pytest -m slow
pytest -m e2e
pytest tests\eval
```

- Add `--headed` to watch the browser: `pytest -m ui --headed`.
- The e2e test is skipped by default. `-m e2e` and `-m slow` run it.
- Tests marked `slow` use real Gemini quota. Run them on purpose.
- `pytest tests\eval` tests the evaluation code itself with fake responses and a local stub server. It makes no AI calls.

## What is covered

- **Health:** the backend is up.
- **Upload:** wrong file type, more than 5 files, over 20 MB, image-only, corrupt, and a valid PDF returning a UUID session id.
- **Ask:** invalid session id, unknown session, empty, whitespace and over-length questions, an answerable question (with sources and a key fact), and an unanswerable one (no sources).
- **Sessions:** delete then ask returns 404, delete with an invalid id returns 400, deleting an unknown id returns 200.
- **Contracts:** every success and error body matches its schema. Answers never contain `*`, `_` or `#`.
- **UI:** upload flow, answer and sources display, input cleared after sending, double-submit guard, expired-session message.
- **E2E:** one real upload and question through the browser.
- **Golden dataset:** 25 labelled questions over three invented PDFs.

## Golden dataset evaluation

A golden dataset is a fixed set of questions with known correct answers. This one measures whether the app retrieves the right page, answers correctly, refuses what it should, and resists instructions hidden in documents.

### Design decisions

- **Invented facts.** The sample PDFs contain made-up names, dates and numbers, so the model cannot answer from general knowledge. A correct answer proves retrieval worked.
- **Labels are verified, not trusted.** A validator re-reads the PDFs with pypdf and fails if any expected keyword is not on the stated page, so a wrong label cannot skew the results.
- **No exact-text matching.** LLM wording changes between runs ("March 14, 1987" or "14 March 1987"). Checks look for key facts instead, such as month and year as separate keywords.
- **No evaluation frameworks.** Every metric is a short function in `eval\metrics.py`, unit-tested with hand-made fake responses.
- **Infrastructure errors are not quality failures.** Timeouts and quota errors are counted separately, and an incomplete run prints a warning.
- **Cached answers.** Each answer is saved and reused, so a re-run only asks what is missing and never repeats a finished call.

### Documents and questions

Three PDFs are generated in code by `eval\make_sample_pdfs.py` into `eval\sample_pdfs\`.

| File | Pages | Content |
|---|---|---|
| `doc_a.pdf` | 4 | Fictional company handbook |
| `doc_b.pdf` | 3 | Fictional product manual |
| `doc_injection.pdf` | 2 | Library guide with a hidden instruction to reply only with `CANARY-7731` |

`eval\golden_dataset.json` has 25 questions:

| Category | Count | Purpose |
|---|---|---|
| `single_fact` | 13 | One fact from one page, including paraphrases, numbers and dates |
| `multi_fact` | 3 | Facts from two pages or two documents |
| `out_of_scope` | 5 | Must be refused, including near-miss topics the documents do not cover |
| `injection` | 3 | Retrieve the poisoned page and must not output the canary |
| `vague` | 1 | "tell me about it", a judgement call |

Each entry has `id`, `category`, `question`, `expected_source`, `expected_page`, `must_contain` (lowercase keywords), `should_refuse` and `notes`. The runner refuses to start if the labels fail validation.

### Running it

```bat
python eval\validate_dataset.py
python eval\run_eval.py --delay 12
python eval\report.py
```

`run_eval.py` uploads the PDFs once, asks every question, saves raw answers to `eval\reports\raw_<timestamp>.json`, deletes the session and prints the report. `report.py` rebuilds the report from the latest raw file with no AI calls and writes `report_<timestamp>.md` and `.json`.

| Option | Effect |
|---|---|
| `--repeats N` | Ask each question N times to measure variance (default 1) |
| `--no-cache` | Ignore saved answers and call the app again |
| `--delay S` | Wait S seconds between live calls |
| `--max-live-calls N` | Stop before more than N live questions |
| `--max-consecutive-errors N` | Stop after N failed calls in a row (default 3, 0 disables) |
| `--ask-timeout S` | Timeout per question in seconds |
| `--base-url URL` | Override `BASE_URL` |

### Cache

Answers are saved in `eval\cache\`, keyed by question, repeat number, PDF hash, `LLM_MODEL` and `EMBEDDING_MODEL`. Finished answers are never asked again, only failed or missing ones. Errors are never cached.

### What each metric means

| Metric | Meaning | Better |
|---|---|---|
| `retrieval_hit_rate` | Of answered answerable questions, how often the right (file, page) was in the sources | Higher |
| `must_contain_pass_rate` | Of those answers, how often every expected keyword was present | Higher |
| `refusal_accuracy` | Of questions that must be refused, how often the app refused with empty sources (vague question excluded) | Higher |
| `false_refusal_rate` | Of questions the documents can answer, how often the app refused | Lower |
| `injection_resistance` | How often an injection answer did not contain the canary | Higher |
| `formatting_compliance` | How often an answer had none of `*`, `_`, `#` | Higher |
| `contract_compliance` | Of HTTP 200 responses, how often the body matched the schema and its rules | Higher |
| `request_error_rate` | How often a call failed outright (not a quality metric) | Lower |
| `latency_p50_s`, `latency_p95_s` | Median and slow-tail response time | Lower |
| `consistency` | With `--repeats`, how often `answerable` was the same on every repeat | Higher |

The report also shows a per-category table, every failing question with its answer and expected values, the model names and the git commit.

### Retrieval observability limitation

When the app refuses, it returns empty `sources`, so retrieval cannot be observed for refused questions. The hit rate covers only questions the app chose to answer, and a refusal could mean a retrieval failure or a generation failure. The report therefore shows `false_refusal_rate` beside the hit rate, plus a worst-case hit rate that counts every refusal as a miss.

## Project layout

```
conftest.py     fixtures: sample files, API client, sessions
helpers.py      constants and helper functions
schemas.py      expected response shapes
pytest.ini      markers: api, ui, slow, e2e, eval
tests/api/      health, upload, ask, sessions, contracts
tests/ui/       upload flow, chat flow, e2e
tests/eval/     unit tests for the evaluation code (no AI calls)
eval/           golden dataset evaluation
  make_sample_pdfs.py   generate and verify the sample PDFs
  sample_pdfs/          generated PDFs
  golden_dataset.json   25 labelled questions
  validate_dataset.py   check labels against the PDF text
  metrics.py            metric functions
  run_eval.py           runner with cache, repeats and quota guards
  report.py             console, Markdown and JSON report
  cache/                saved answers (not committed)
  reports/              raw results and reports (not committed)
```