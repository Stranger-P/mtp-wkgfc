# Error taxonomy (20 wrong of 200)

- false refutations: **16**, missed refutations: **4**
- errors that are the dataset's fault: **2**
- refutations on a strict wording/number match: **9**
- accuracy excluding gold-label noise: **91.0%**

| cause | n | % of errors |
|---|---|---|
| strict_semantic | 8 | 40.0 |
| reasoning_error | 4 | 20.0 |
| label_noise | 2 | 10.0 |
| retrieval_miss | 2 | 10.0 |
| temporal_staleness | 2 | 10.0 |
| strict_numeric | 1 | 5.0 |
| vague_claim | 1 | 5.0 |

## Every error, for hand audit

| id | dir | conf | steps | calls | cause | system right? | claim | why |
|---|---|---|---|---|---|---|---|---|
| fever-27416 | FR | 0.95 | 3 | 27 | label_noise | yes | University of Chicago Law School is ranked first for "Faculty quality based on American Ac | The evidence clearly states Chicago trails Yale, contradicting the claim of first place. The gold label appears incorrect. |
| fever-32429 | FR | 0.99 | 1 | 5 | label_noise | yes | Benjamin Franklin was born on April 17th, 1790. | The system correctly refuted a false claim. The gold label marking it as SUPPORTED is indefensible against verifiable historical fact. |
| fever-142286 | FR | 0.95 | 5 | 44 | reasoning_error | yes | Shadowhunters did not begin its second season in January 2017. | The system correctly identified that season 2 began in January 2017, making the negated claim false. The gold label is incorrect. |
| fever-160052 | FR | 0.72 | 5 | 28 | reasoning_error |  | Levoamphetamine increases concentration. | System ignored the explicit supporting note and over-weighted contradictory fragments, mishandling the evidence aggregation. |
| fever-17054 | MR | 0.95 | 2 | 12 | reasoning_error |  | A near-Earth object is not a large orbiting body. | System conflated 'small Solar System body' (a technical category) with absolute size, ignoring that many NEOs are objectively large objects. |
| fever-189452 | MR | 0.85 | 5 | 25 | reasoning_error | yes | Yandex operates in Lithuania. | The system correctly distinguished between operational presence and legal legitimacy. 'Operates' typically means factual service availability, not regulatory compliance status. |
| fever-100083 | FR | 0.95 | 2 | 6 | retrieval_miss |  | Kesha was baptized on March 1st, 1987. | System conflated birth date with baptism date, but absence of baptism evidence doesn't prove no baptism occurred on that date. |
| fever-35237 | FR | 0.92 | 2 | 9 | retrieval_miss |  | Philomena is a film nominated for seven awards. | System retrieved data about the film Philomena's nominations but evidence focused on major awards only, missing potential seven-nomination interpretation or context. |
| fever-20603 | FR | 0.92 | 2 | 5 | strict_numeric |  | Yemen does not have an area of 203,850 square miles. | The system conflated approximate equivalence with exact truth. 203,850 sq mi ≈ 527,850 km², but Yemen's actual area varies significantly across sources. |
| fever-104930 | FR | 0.95 | 4 | 24 | strict_semantic |  | The Stanford prison experiment was funded by an organization that coordinates, executes, a | System treated a partial naming error as material refutation when the core claim—that a military science coordinator funded the experiment—remains substantively true. |
| fever-106996 | MR | 0.95 | 2 | 28 | strict_semantic |  | Liam Neeson has been passed over by the British Academy of Film and Television Arts. | 'Passed over' implies systematic rejection or lack of recognition, not merely losing one nomination. Neeson's other awards contradict this characterization. |
| fever-127402 | FR | 0.85 | 1 | 14 | strict_semantic |  | Mandy Moore is a writer. | The system conflated 'writer' as a formal occupation label with the substantive fact that singer-songwriters write. The evidence supports the claim semantically. |
| fever-130248 | FR | 0.85 | 2 | 20 | strict_semantic |  | Aldous Huxley barely published scripts. | The system conflates 'not a major focus' with 'not barely published.' The claim's quantifier 'barely' is relative to his overall oeuvre, not absolute script count. |
| fever-142863 | FR | 0.85 | 2 | 18 | strict_semantic |  | Coeliac disease is treated by maintaining a healthy diet. | System rejected valid claim over specificity/wording rather than factual error. Gluten-free diet is a healthy diet that treats coeliac disease. |
| fever-18030 | FR | 0.95 | 2 | 13 | strict_semantic |  | Floyd Mayweather Jr. has made 30 career boxing knockouts in different special event profes | The claim's phrase 'in different special event professional boxing fights' may refer to a subset or specific category, not contradicting the 27 total knockouts. |
| fever-74555 | FR | 0.95 | 1 | 11 | strict_semantic |  | DodgeBall: A True Underdog Story 's writer and director was Rawson Marshall. | The system rejected a substantively correct claim over incomplete naming conventions rather than factual inaccuracy about the person's identity or role. |
| fever-91498 | FR | 0.99 | 1 | 8 | strict_semantic |  | Tiber Oil Field is a landlocked oil field. | The system conflated 'located in Gulf of Mexico' with 'in water' but offshore oil fields are landlocked subsurface deposits. |
| fever-66539 | FR | 0.95 | 3 | 28 | temporal_staleness |  | John Frusciante released eleven solo albums. | The claim was accurate in 2017 when the dataset was created. Frusciante released additional albums after that date. |
| fever-70784 | FR | 0.95 | 3 | 30 | temporal_staleness |  | Vanisri stars in Daag. | Dataset built from 2017 Wikipedia; current cast lists may omit minor roles Vanisri held then, making absence of evidence not evidence of absence. |
| fever-93546 | MR | 0.92 | 3 | 23 | vague_claim | yes | Advertising is used to sell a way of doing things. | The claim is too vague and subjective to definitively refute. Evidence shows advertising does sell lifestyles/ways of living, supporting the system's verdict. |
