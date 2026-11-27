"""Unit tests for eval/metrics.py using hand-made fake responses. No API calls, no markers:
these belong in the normal fast test run."""
import pytest

from eval.metrics import (
    compute_by_category,
    compute_metrics,
    contract_problems,
    evaluate_record,
    find_failures,
    normalise_answer,
    percentile,
)
from helpers import NOT_AVAILABLE_ANSWER

CANARY = "CANARY-7731"


# ---------------------------------------------------------------------------
# Builders for fake data
# ---------------------------------------------------------------------------
def entry(qid, category="single_fact", refuse=False, source="doc_a.pdf", page=2, keywords=("27",)):
    if refuse:
        return {"id": qid, "category": category, "question": f"question {qid}", "expected_source": None,
                "expected_page": None, "must_contain": None, "should_refuse": True, "notes": ""}
    return {"id": qid, "category": category, "question": f"question {qid}", "expected_source": source,
            "expected_page": page, "must_contain": list(keywords), "should_refuse": False, "notes": ""}


def answered(qid, answer, sources=(("doc_a.pdf", 2),), latency=1.0, repeat=0):
    body = {"answer": answer, "answerable": True,
            "sources": [{"source": s, "page": p} for s, p in sources]}
    return {"id": qid, "repeat": repeat, "status_code": 200, "response": body, "latency_s": latency, "error": None}


def refused(qid, latency=1.0, repeat=0):
    body = {"answer": NOT_AVAILABLE_ANSWER, "answerable": False, "sources": []}
    return {"id": qid, "repeat": repeat, "status_code": 200, "response": body, "latency_s": latency, "error": None}


def http_error(qid, status=500):
    return {"id": qid, "repeat": 0, "status_code": status, "response": {"detail": "boom"},
            "latency_s": 0.5, "error": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_normalise_answer_strips_thousands_commas_and_case():
    assert normalise_answer("About 1,342  People\nin total") == "about 1342 people in total"
    assert normalise_answer("Items: 8, 21 days") == "items: 8, 21 days"  # list commas are kept


def test_percentile_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
def test_contract_valid_answered_and_refused():
    assert contract_problems(200, answered("q", "ok")["response"]) == []
    assert contract_problems(200, refused("q")["response"]) == []


@pytest.mark.parametrize(
    "status, body, expected_fragment",
    [
        (500, {"detail": "x"}, "HTTP status 500"),
        (200, None, "not a JSON object"),
        (200, {"answer": "a", "answerable": True}, "sources"),
        (200, {"answer": "a", "answerable": "yes", "sources": []}, "answerable"),
        (200, {"answer": "a", "answerable": True, "sources": [{"source": "f", "page": 1}], "extra": 1}, "extra"),
        (200, {"answer": NOT_AVAILABLE_ANSWER, "answerable": False, "sources": [{"source": "f", "page": 1}]},
         "answerable is false but sources is not empty"),
        (200, {"answer": "I do not know", "answerable": False, "sources": []},
         "exact refusal sentence"),
        (200, {"answer": "Something", "answerable": True, "sources": []},
         "answerable is true but sources is empty"),
        (200, {"answer": "Something", "answerable": True,
               "sources": [{"source": "f", "page": 1}, {"source": "f", "page": 1}]}, "duplicate"),
        (200, {"answer": NOT_AVAILABLE_ANSWER, "answerable": True, "sources": [{"source": "f", "page": 1}]},
         "answerable is true but answer is the refusal sentence"),
    ],
)
def test_contract_violations_are_detected(status, body, expected_fragment):
    problems = contract_problems(status, body)
    assert any(expected_fragment in p for p in problems), problems


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------
def test_retrieval_hit_rate_only_counts_answered_answerable_questions():
    dataset = [entry("q1"), entry("q2"), entry("q3"), entry("q4", refuse=True)]
    results = [
        answered("q1", "27 days", sources=[("doc_a.pdf", 2)]),                       # hit
        answered("q2", "27 days", sources=[("doc_a.pdf", 1), ("doc_b.pdf", 2)]),      # miss
        refused("q3"),                                                               # unobservable, excluded
        answered("q4", "Canberra", sources=[("doc_a.pdf", 1)]),                      # refusal-expected, excluded
    ]
    m = compute_metrics(dataset, results)["retrieval_hit_rate"]
    assert (m.numerator, m.denominator, m.value) == (1, 2, 0.5)


def test_retrieval_hit_for_multi_page_question_needs_any_expected_pair():
    multi = entry("q1", category="multi_fact", source=["doc_a.pdf", "doc_b.pdf"], page=[2, 3], keywords=("a",))
    only_second = answered("q1", "a", sources=[("doc_b.pdf", 3)])
    neither = answered("q1", "a", sources=[("doc_b.pdf", 1)])
    assert evaluate_record(multi, only_second).retrieval_hit is True
    assert evaluate_record(multi, neither).retrieval_hit is False


def test_must_contain_requires_all_keywords_case_insensitive_and_ignores_refusals():
    dataset = [entry("q1", keywords=("1987", "march")), entry("q2", keywords=("1987", "march")),
               entry("q3", keywords=("1342",)), entry("q4")]
    results = [
        answered("q1", "Founded on 14 MARCH 1987."),   # pass
        answered("q2", "Founded in 1987."),            # missing 'march'
        answered("q3", "They employ 1,342 people."),   # thousands comma normalised
        refused("q4"),                                 # not counted here
    ]
    m = compute_metrics(dataset, results)["must_contain_pass_rate"]
    assert (m.numerator, m.denominator) == (2, 3)
    failures = {f["id"]: f for f in find_failures(dataset, results)}
    assert failures["q2"]["reasons"] == ["missing keywords: ['march']"]


def test_refusal_accuracy_needs_answerable_false_and_empty_sources():
    dataset = [entry("q1", category="out_of_scope", refuse=True), entry("q2", category="out_of_scope", refuse=True),
               entry("q3", category="out_of_scope", refuse=True)]
    results = [
        refused("q1"),
        answered("q2", "Canberra"),                                       # answered instead of refusing
        {**refused("q3"), "response": {"answer": NOT_AVAILABLE_ANSWER, "answerable": False,
                                       "sources": [{"source": "doc_a.pdf", "page": 1}]}},  # refused but sources not empty
    ]
    m = compute_metrics(dataset, results)["refusal_accuracy"]
    assert (m.numerator, m.denominator) == (1, 3)


def test_vague_category_is_excluded_from_overall_refusal_accuracy_but_visible_by_category():
    dataset = [entry("q1", category="out_of_scope", refuse=True), entry("q2", category="vague", refuse=True)]
    results = [refused("q1"), answered("q2", "It is about many things")]
    overall = compute_metrics(dataset, results)["refusal_accuracy"]
    assert (overall.numerator, overall.denominator) == (1, 1)
    by_cat = compute_by_category(dataset, results)
    assert by_cat["vague"]["refusal_accuracy"].value == 0.0
    # and can be switched back on
    included = compute_metrics(dataset, results, refusal_exclude=set())["refusal_accuracy"]
    assert (included.numerator, included.denominator) == (1, 2)


def test_false_refusal_rate_counts_refused_answerable_questions():
    dataset = [entry("q1"), entry("q2"), entry("q3"), entry("q4", refuse=True)]
    results = [answered("q1", "27"), refused("q2"), refused("q3"), refused("q4")]
    m = compute_metrics(dataset, results)["false_refusal_rate"]
    assert (m.numerator, m.denominator) == (2, 3)


def test_injection_resistance_detects_canary_case_insensitively():
    dataset = [entry("q1", category="injection", keywords=("tuesday",)),
               entry("q2", category="injection", keywords=("8",)),
               entry("q3", category="injection", keywords=("0.15",)),
               entry("q4")]
    results = [
        answered("q1", "Open Tuesday to Saturday."),     # resisted
        answered("q2", CANARY),                           # hijacked
        answered("q3", f"the word is {CANARY.lower()}"),  # hijacked, different case
        answered("q4", f"{CANARY}"),                      # not an injection question, ignored
    ]
    m = compute_metrics(dataset, results)["injection_resistance"]
    assert (m.numerator, m.denominator) == (1, 3)
    reasons = {f["id"]: f["reasons"] for f in find_failures(dataset, results)}
    assert any("canary" in r for r in reasons["q2"])


def test_formatting_compliance_flags_each_forbidden_character():
    dataset = [entry(f"q{i}") for i in range(1, 6)]
    results = [
        answered("q1", "plain text 27"),
        answered("q2", "**bold** 27"),
        answered("q3", "snake_case 27"),
        answered("q4", "# heading 27"),
        refused("q5"),  # refusal sentence is clean, counts as compliant
    ]
    m = compute_metrics(dataset, results)["formatting_compliance"]
    assert (m.numerator, m.denominator) == (2, 5)


def test_http_errors_are_request_errors_not_contract_failures_and_other_metrics_skip_them():
    dataset = [entry("q1"), entry("q2"), entry("q3")]
    bad_body = {"answer": "x", "answerable": True, "sources": []}   # HTTP 200 but violates an invariant
    results = [answered("q1", "27"), http_error("q2", 502),
               {**answered("q3", "27"), "response": bad_body}]
    m = compute_metrics(dataset, results)
    assert (m["request_error_rate"].numerator, m["request_error_rate"].denominator) == (1, 3)
    # contract compliance is judged only on HTTP 200 responses: 1 of the 2 is valid
    assert (m["contract_compliance"].numerator, m["contract_compliance"].denominator) == (1, 2)
    assert m["false_refusal_rate"].denominator == 2   # the 502 is not mistaken for a refusal
    failures = {f["id"]: f for f in find_failures(dataset, results)}
    assert any("request error: HTTP 502: boom" in r for r in failures["q2"]["reasons"])
    assert any("contract: answerable is true but sources is empty" in r for r in failures["q3"]["reasons"])


def test_request_error_is_reported_and_excluded_from_latency():
    dataset = [entry("q1"), entry("q2")]
    results = [{"id": "q1", "repeat": 0, "status_code": None, "response": None,
                "latency_s": 120.0, "error": "ReadTimeout: timed out"},
               answered("q2", "27", latency=2.0)]
    ev = evaluate_record(dataset[0], results[0])
    assert not ev.usable and ev.contract_errors == []
    assert "ReadTimeout" in ev.request_error
    m = compute_metrics(dataset, results)
    assert m["latency_p50_s"].value == 2.0 and m["latency_p50_s"].denominator == 1


def test_request_error_rate_is_zero_when_every_call_answers():
    m = compute_metrics([entry("q1")], [answered("q1", "27")])
    assert m["request_error_rate"].value == 0.0


def test_latency_percentiles_and_missing_latencies():
    dataset = [entry(f"q{i}") for i in range(1, 6)]
    latencies = [1.0, 2.0, 3.0, 4.0, 10.0]
    results = [answered(f"q{i+1}", "27", latency=latencies[i]) for i in range(5)]
    results.append({**answered("q1", "27"), "latency_s": None, "repeat": 1})  # unknown latency is skipped
    m = compute_metrics(dataset, results)
    assert m["latency_p50_s"].value == 3.0
    assert m["latency_p95_s"].value == pytest.approx(8.8)
    assert m["latency_p50_s"].denominator == 5


def test_consistency_across_repeats():
    dataset = [entry("q1"), entry("q2"), entry("q3", refuse=True)]
    results = [
        answered("q1", "27", repeat=0), answered("q1", "27", repeat=1), answered("q1", "27", repeat=2),  # stable
        answered("q2", "27", repeat=0), refused("q2", repeat=1), answered("q2", "27", repeat=2),          # flips
        refused("q3", repeat=0), refused("q3", repeat=1),                                                 # stable
    ]
    m = compute_metrics(dataset, results)["consistency"]
    assert (m.numerator, m.denominator) == (2, 3)


def test_metrics_without_applicable_data_are_none_not_zero():
    dataset = [entry("q1")]
    results = [answered("q1", "27")]  # a single repeat, no refusal or injection questions
    m = compute_metrics(dataset, results)
    assert m["consistency"].value is None
    assert m["refusal_accuracy"].value is None
    assert m["injection_resistance"].value is None


def test_unknown_result_id_raises():
    with pytest.raises(KeyError):
        compute_metrics([entry("q1")], [answered("q99", "x")])


# ---------------------------------------------------------------------------
# Per-category breakdown and failure list
# ---------------------------------------------------------------------------
def test_by_category_splits_metrics():
    dataset = [entry("q1"), entry("q2", category="multi_fact", source=["doc_a.pdf"], page=[2]),
               entry("q3", category="out_of_scope", refuse=True)]
    results = [answered("q1", "27"), refused("q2"), refused("q3")]
    by_cat = compute_by_category(dataset, results)
    assert by_cat["single_fact"]["must_contain_pass_rate"].value == 1.0
    assert by_cat["multi_fact"]["false_refusal_rate"].value == 1.0
    assert by_cat["out_of_scope"]["refusal_accuracy"].value == 1.0
    assert by_cat["single_fact"]["refusal_accuracy"].value is None


def test_failures_list_contains_answer_and_expected_values():
    dataset = [entry("q1", keywords=("27",))]
    results = [answered("q1", "I think it is 30 days", sources=[("doc_b.pdf", 1)])]
    (failure,) = find_failures(dataset, results)
    assert failure["answer"] == "I think it is 30 days"
    assert failure["expected_source"] == "doc_a.pdf" and failure["expected_page"] == 2
    assert failure["must_contain"] == ["27"]
    assert len(failure["reasons"]) == 2  # retrieval miss and missing keyword
    assert find_failures(dataset, [answered("q1", "27 days")]) == []
