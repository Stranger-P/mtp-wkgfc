# Error adjudication (20 errors, FEVER-200)

Claude's hand reading of each error, for the user to confirm or override. The `agree?`
column compares the LLM classifier's de-biased label with this adjudication.

Classifier agreement with this adjudication: **10/20 (50%)**

| id | dir | adjudicated cause | system right? | certainty | classifier said | agree? | note |
|---|---|---|---|---|---|---|---|
| fever-70784 | FR | **entity_linking** | no | certain | retrieval_miss | NO | Several films are titled Daag; system took the 1973 Hindi one. |
| fever-196759 | MR | **entity_linking** | no | certain | entity_linking | yes | 'When Marnie Was There' (2014) vs the Marnie FEVER means. |
| fever-32429 | FR | **label_noise** | yes | certain | label_noise | yes | Franklin born 17 Jan 1706, died 17 Apr 1790. Gold is indefensible. |
| fever-91498 | FR | **label_noise** | yes | certain | strict_semantic | NO | Tiber is a deepwater Gulf of Mexico field; 'landlocked' is false. |
| fever-142286 | FR | **label_noise** | yes | certain | label_noise | yes | Season 2 did premiere 2 Jan 2017, so the negated claim is false. |
| fever-17054 | MR | **label_noise** | yes | uncertain | reasoning_error | NO | NEOs are small-body class; 'large' is vague, so gold is arguable. |
| fever-20603 | FR | **reasoning_error** | no | certain | strict_numeric | NO | Negation plus tolerance applied the wrong way round. |
| fever-106996 | MR | **reasoning_error** | no | certain | strict_semantic | NO | Nominated-but-not-won does not settle 'passed over by'. |
| fever-27416 | FR | **retrieval_miss** | no | uncertain | temporal_staleness | NO | System's source describes a different metric than the claim's. |
| fever-104930 | FR | **retrieval_miss** | no | certain | retrieval_miss | yes | Funder is the Office of Naval Research; system found NIMH. |
| fever-161902 | FR | **strict_numeric** | no | certain | strict_numeric | yes | July vs 1 August 1969: one month, not a substantive contradiction. |
| fever-18030 | FR | **strict_numeric** | no | certain | strict_semantic | NO | 30 vs 27 knockouts; the claim's qualifier may change the count. |
| fever-35237 | FR | **strict_numeric** | no | certain | strict_semantic | NO | Seven vs eight nominations, and 'awards' is ambiguous. |
| fever-100083 | FR | **strict_semantic** | no | certain | label_noise | NO | FEVER conflates born/baptized; system demanded a baptism record. |
| fever-18079 | FR | **strict_semantic** | no | certain | strict_semantic | yes | Norway was still a kingdom in personal union with Sweden in 1902. |
| fever-142863 | FR | **strict_semantic** | no | certain | strict_semantic | yes | A gluten-free diet is an instance of a healthy diet. |
| fever-82973 | FR | **strict_semantic** | no | certain | strict_semantic | yes | Netflix original = commissioned for Netflix; system read 'created for' narrowly. |
| fever-130248 | FR | **strict_semantic** | no | certain | strict_semantic | yes | 'Barely' is a vague quantifier the evidence does not clearly defeat. |
| fever-66539 | FR | **temporal_staleness** | no | certain | temporal_staleness | yes | Eleven solo albums was right in 2017; thirteen now. |
| fever-93546 | MR | **vague_claim** | no | certain | reasoning_error | NO | 'A way of doing things' is not settleable from evidence. |

## Adjudicated totals

| cause | n | % |
|---|---|---|
| strict_semantic | 5 | 25% |
| label_noise | 4 | 20% |
| strict_numeric | 3 | 15% |
| retrieval_miss | 2 | 10% |
| reasoning_error | 2 | 10% |
| entity_linking | 2 | 10% |
| temporal_staleness | 1 | 5% |
| vague_claim | 1 | 5% |

- strict-match refutations (semantic + numeric): **8/20 (40%)** -> the primary target
- dataset's fault: **4/20**, so true accuracy on this sample is **92.0%**, not 90.0%
- temporal staleness: **1/20** -> cannot be shown on FEVER
