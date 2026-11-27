"""Fast tests (no API calls): the golden dataset is valid and the validator catches bad labels."""
import copy
import json

from eval.validate_dataset import DATASET_PATH, load_pages, validate


def _raw():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_golden_dataset_has_no_labelling_errors():
    errors, _ = validate(_raw(), load_pages())
    assert errors == []


def test_validator_catches_keyword_not_on_page():
    raw = _raw()
    bad = copy.deepcopy(raw)
    bad[0]["must_contain"] = ["definitely-not-in-the-pdf"]
    errors, _ = validate(bad, load_pages())
    assert any("q01" in e and "NOT found" in e for e in errors)


def test_validator_catches_wrong_page():
    raw = _raw()
    bad = copy.deepcopy(raw)
    bad[4]["expected_page"] = 4  # 27 days of leave is on page 2, not page 4
    errors, _ = validate(bad, load_pages())
    assert any("q05" in e and "NOT found" in e for e in errors)


def test_validator_catches_inconsistent_refusal_and_duplicate_id():
    raw = _raw()
    bad = copy.deepcopy(raw)
    bad[16]["must_contain"] = ["canberra"]  # q17 is a refusal and must carry no keywords
    bad[1]["id"] = "q01"                    # duplicate id
    errors, _ = validate(bad, load_pages())
    assert any("q17" in e and "should_refuse" in e for e in errors)
    assert any("q01" in e and "used 2 times" in e for e in errors)
