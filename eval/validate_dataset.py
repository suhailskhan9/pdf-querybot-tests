"""Validate eval/golden_dataset.json BEFORE any real evaluation.

Checks:
  1. Schema of every entry (Pydantic).
  2. Internal consistency: refusals carry no expectations, answerable questions
     carry source, page and keywords; lists are paired; ids are unique.
  3. Labels against the PDFs: the expected file exists, the page exists, and
     EVERY must_contain keyword really appears in the extracted text of at
     least one of the expected pages (using pypdf).
  4. Category counts match the planned mix (reported as warnings, not errors).

Run:  python eval/validate_dataset.py
Exit code 0 = no errors, 1 = labelling problems found.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "golden_dataset.json"
PDF_DIR = HERE / "sample_pdfs"

CATEGORIES = {"single_fact", "multi_fact", "out_of_scope", "injection", "vague"}
PLANNED_MIX = {"single_fact": 13, "multi_fact": 3, "out_of_scope": 5, "injection": 3, "vague": 1}


class Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    question: str
    expected_source: Optional[Union[str, list[str]]]
    expected_page: Optional[Union[int, list[int]]]
    must_contain: Optional[list[str]]
    should_refuse: bool
    notes: str

    @field_validator("category")
    @classmethod
    def _category(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"category must be one of {sorted(CATEGORIES)}")
        return v

    @field_validator("question", "notes")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace, same idea as the PDF generator."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_pages(pdf_dir: Path = PDF_DIR) -> dict[str, list[str]]:
    """file name -> list of normalised page texts (index 0 = page 1)."""
    pages: dict[str, list[str]] = {}
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf))
        pages[pdf.name] = [normalise(p.extract_text() or "") for p in reader.pages]
    return pages


def expected_pairs(entry: Entry) -> list[tuple[str, int]]:
    """Expected (source, page) pairs, paired by position. Empty for refusals."""
    if entry.expected_source is None or entry.expected_page is None:
        return []
    sources = entry.expected_source if isinstance(entry.expected_source, list) else [entry.expected_source]
    pages = entry.expected_page if isinstance(entry.expected_page, list) else [entry.expected_page]
    return list(zip(sources, pages))


def validate(raw_entries: list[dict], pages: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Pure function, so it can be unit-tested."""
    errors: list[str] = []
    warnings: list[str] = []
    entries: list[Entry] = []
    short_kw: dict[str, list[str]] = {}

    for i, raw in enumerate(raw_entries):
        label = raw.get("id", f"#{i}") if isinstance(raw, dict) else f"#{i}"
        try:
            entries.append(Entry(**raw))
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                errors.append(f"{label}: schema: {loc}: {err['msg']}")

    ids = Counter(e.id for e in entries)
    for dup, n in ids.items():
        if n > 1:
            errors.append(f"{dup}: id used {n} times")

    for e in entries:
        # --- consistency -----------------------------------------------------
        if e.should_refuse:
            for field in ("expected_source", "expected_page", "must_contain"):
                if getattr(e, field) is not None:
                    errors.append(f"{e.id}: should_refuse=true but {field} is not null")
            continue

        if e.expected_source is None or e.expected_page is None:
            errors.append(f"{e.id}: answerable question needs expected_source and expected_page")
            continue
        if not e.must_contain:
            errors.append(f"{e.id}: answerable question needs a non-empty must_contain list")
            continue

        n_src = len(e.expected_source) if isinstance(e.expected_source, list) else 1
        n_pg = len(e.expected_page) if isinstance(e.expected_page, list) else 1
        if n_src != n_pg:
            errors.append(f"{e.id}: expected_source has {n_src} item(s) but expected_page has {n_pg}")
            continue
        if e.category == "multi_fact" and n_src < 2:
            errors.append(f"{e.id}: multi_fact needs at least 2 expected (source, page) pairs")

        # --- labels vs PDF text ----------------------------------------------
        pair_texts: list[str] = []
        for source, page in expected_pairs(e):
            if source not in pages:
                errors.append(f"{e.id}: expected_source '{source}' not found in sample_pdfs")
                continue
            if not 1 <= page <= len(pages[source]):
                errors.append(f"{e.id}: {source} has {len(pages[source])} pages, expected_page={page}")
                continue
            pair_texts.append(pages[source][page - 1])

        for kw in e.must_contain:
            if kw != kw.lower():
                errors.append(f"{e.id}: keyword '{kw}' must be lowercase")
            if kw.strip() != kw or not kw:
                errors.append(f"{e.id}: keyword '{kw}' has stray whitespace or is empty")
            if pair_texts and not any(kw.lower() in t for t in pair_texts):
                errors.append(
                    f"{e.id}: keyword '{kw}' NOT found on the expected page(s) "
                    f"{expected_pairs(e)}"
                )

        for kw in e.must_contain:
            if len(kw) <= 2:
                short_kw.setdefault(e.id, []).append(kw)

    if short_kw:
        listing = ", ".join(f"{i}={kws}" for i, kws in short_kw.items())
        warnings.append(
            "short numeric keywords (1-2 characters) could match a wrong answer by chance; "
            "they are the right fact to check but a weak signal on their own: " + listing
        )

    counts = Counter(e.category for e in entries)
    for cat, planned in PLANNED_MIX.items():
        if counts.get(cat, 0) != planned:
            warnings.append(f"category '{cat}': {counts.get(cat, 0)} questions, planned {planned}")

    return errors, warnings


def main() -> int:
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return 1
    if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
        print("No PDFs in eval/sample_pdfs. Run: python eval/make_sample_pdfs.py")
        return 1

    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    pages = load_pages()
    errors, warnings = validate(raw, pages)

    counts = Counter(r.get("category") for r in raw if isinstance(r, dict))
    print(f"Dataset: {len(raw)} questions  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"PDFs:    " + ", ".join(f"{k} ({len(v)} pages)" for k, v in pages.items()))

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print("  -", w)
    if errors:
        print("\nLABELLING PROBLEMS:")
        for err in errors:
            print("  -", err)
        print(f"\n{len(errors)} problem(s). Fix them before running the evaluation.")
        return 1
    print("\nOK: schema valid, all must_contain keywords found on the expected pages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
