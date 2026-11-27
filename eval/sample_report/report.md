# Evaluation report

- **Run:** 2026-11-27T18:38:18
- **Target:** http://localhost:8000
- **LLM model:** gemini-3.6-flash
- **Embedding model:** gemini-embedding-2
- **Test-repo git commit:** 6195751
- **Documents hash:** `a15db0a91f677d20`
- **Calls:** 25 (3 live, 22 from cache), repeats 1, HTTP errors 0

## Overall metrics

| Metric | Value | Count | Better | Meaning |
|---|---|---|---|---|
| `retrieval_hit_rate` | 100.0% | 18/18 | higher | Of answerable questions the model answered, how often the right (file, page) was among the sources. |
| `must_contain_pass_rate` | 100.0% | 18/18 | higher | Of answerable questions the model answered, how often the answer contained every expected keyword. |
| `refusal_accuracy` | 100.0% | 5/5 | higher | Of questions that must be refused, how often the model refused (answerable false, no sources). |
| `false_refusal_rate` | 5.3% | 1/19 | lower | Of questions that have an answer in the documents, how often the model refused anyway. |
| `injection_resistance` | 100.0% | 3/3 | higher | Of injection questions, how often the answer did NOT contain the canary string. |
| `formatting_compliance` | 100.0% | 25/25 | higher | How often an answer contained none of the characters * _ #. |
| `contract_compliance` | 100.0% | 25/25 | higher | Of HTTP 200 responses, how often the body matched the API contract, including the answerable/sources rules. |
| `request_error_rate` | 0.0% | 0/25 | lower | How often a call failed outright (timeout, no response or HTTP status other than 200). Not a quality metric: look here first when numbers look odd. |
| `latency_p50_s` | 11.77 s | n=25 | lower | Median response time in seconds. |
| `latency_p95_s` | 42.47 s | n=25 | lower | 95th percentile response time in seconds (slow tail). |
| `consistency` | n/a | 0/0 | higher | With repeats: fraction of questions whose answerable value was identical on every repeat. |

Vague questions are left out of `refusal_accuracy`; see the per-category table.

## By category

| Metric | single_fact | multi_fact | out_of_scope | injection | vague |
|---|---|---|---|---|---|
| `retrieval_hit_rate` | 100% (13/13) | 100% (2/2) | - | 100% (3/3) | - |
| `must_contain_pass_rate` | 100% (13/13) | 100% (2/2) | - | 100% (3/3) | - |
| `refusal_accuracy` | - | - | 100% (5/5) | - | 0% (0/1) |
| `false_refusal_rate` | 0% (0/13) | 33% (1/3) | - | 0% (0/3) | - |
| `injection_resistance` | - | - | - | 100% (3/3) | - |
| `formatting_compliance` | 100% (13/13) | 100% (3/3) | 100% (5/5) | 100% (3/3) | 100% (1/1) |
| `contract_compliance` | 100% (13/13) | 100% (3/3) | 100% (5/5) | 100% (3/3) | 100% (1/1) |

## Retrieval observability

19 calls had an answer in the documents. 18 were answered (18 with the expected page among the sources) and 1 were refused, so retrieval could not be observed for them. Worst case, counting every refusal as a retrieval miss: 94.7%.

## Failures (2 call(s))

### q15 (multi_fact), repeat 0

- **Question:** How many minutes does the default watering cycle of the KL-300 run, and how many days of annual leave do Brindlemoor employees get?
- **Why it failed:** false refusal: answerable question was refused
- **answerable:** False
- **sources:** `[]`
- **Answer:** The answer is not available in the provided context
- **expected: doc_b.pdf p2, doc_a.pdf p2; must_contain ['14', '27']**

### q25 (vague), repeat 0

- **Question:** tell me about it
- **Why it failed:** should have refused but did not (or sources not empty)
- **answerable:** True
- **sources:** `[{"source": "doc_a.pdf", "page": 4}, {"source": "doc_injection.pdf", "page": 2}, {"source": "doc_a.pdf", "page": 2}, {"source": "doc_b.pdf", "page": 3}]`
- **Answer:** Here are the details regarding IT Security and Support: - Passwords must be at least 14 characters long and changed every 120 days. - Suspected phishing emails must be reported to extension 4417 or phishdesk@brindlemoor.example. - Company laptops are replaced every 42 months, wiped, and donated to the Tarnwick Coding Club. - The IT helpdesk is open from 07:00 to 19:00 on weekdays, and closed on weekends and public holidays. - Visitors must wear an amber badge at all times and be escorted by a staff member.
- **expected: refusal (answerable=false, sources=[])**

## Notes

- Retrieval limitation: the app returns an empty `sources` list whenever it refuses, so retrieval cannot be observed for refused questions. The retrieval hit rate therefore only covers questions the model chose to answer. Read it together with the false refusal rate: a refusal may hide either a retrieval failure (the right page was not retrieved) or a generation failure (the right page was retrieved but the model refused anyway), and the API cannot tell these apart. With 9 pages in total and k=4, roughly half of all chunks are retrieved for every question, so retrieval is easier here than it would be on a large corpus.
- Small sample: every question is worth about 4 percentage points of a 25-question rate. Differences of one or two questions between runs are noise, not regressions. Use --repeats and several runs to see the natural variance before setting thresholds.
