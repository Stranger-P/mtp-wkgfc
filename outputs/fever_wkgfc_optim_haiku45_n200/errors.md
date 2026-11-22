# Error taxonomy (20 wrong of 200)

- false refutations: **16**, missed refutations: **4**
- errors that are the dataset's fault: **3**
- refutations on a strict wording/number match: **10**
- accuracy excluding gold-label noise: **91.5%**

| cause | n | % of errors |
|---|---|---|
| strict_semantic | 8 | 40.0 |
| label_noise | 3 | 15.0 |
| strict_numeric | 2 | 10.0 |
| retrieval_miss | 2 | 10.0 |
| temporal_staleness | 2 | 10.0 |
| reasoning_error | 2 | 10.0 |
| entity_linking | 1 | 5.0 |

## Every error, for hand audit

| id | dir | conf | steps | calls | cause | system right? | claim | why |
|---|---|---|---|---|---|---|---|---|
| fever-196759 | MR | 0.95 | 2 | 16 | entity_linking |  | Marnie was created in 2014. | The system conflated Marnie the 2014 animated film with another entity the dataset's gold label refers to, likely the 1964 Hitchcock thriller or the given name. |
| fever-100083 | FR | 0.95 | 4 | 10 | label_noise | yes | Kesha was baptized on March 1st, 1987. | Birth and baptism are distinct events. No evidence supports a baptism on that date. The system correctly distinguished them; the gold label appears erroneous. |
| fever-142286 | FR | 0.95 | 2 | 17 | label_noise | yes | Shadowhunters did not begin its second season in January 2017. | The system correctly identified that the negated claim contradicts the evidence. The gold label SUPPORTED is indefensible; the system's REFUTED verdict is correct. |
| fever-32429 | FR | 0.99 | 1 | 5 | label_noise | yes | Benjamin Franklin was born on April 17th, 1790. | The system correctly refuted a false claim. The gold label marking it as SUPPORTED is indefensible against verifiable historical fact. |
| fever-17054 | MR | 0.92 | 3 | 15 | reasoning_error |  | A near-Earth object is not a large orbiting body. | System confused 'small in size' with 'not an orbiting body.' NEOs are definitionally orbiting bodies; size classification doesn't negate orbital status. |
| fever-93546 | MR | 0.78 | 4 | 27 | reasoning_error |  | Advertising is used to sell a way of doing things. | System conflated advertising's incidental cultural effects with its primary commercial purpose of selling products/services, committing a category error. |
| fever-104930 | FR | 0.95 | 6 | 28 | retrieval_miss |  | The Stanford prison experiment was funded by an organization that coordinates, executes, a | System retrieved NIMH as funder but missed ONR, the actual funder that matches the organizational description in the claim. |
| fever-70784 | FR | 0.95 | 2 | 18 | retrieval_miss |  | Vanisri stars in Daag. | Absence of evidence in retrieved cast lists is not evidence of absence; minor/uncredited roles often omitted from documentation. |
| fever-161902 | FR | 0.95 | 5 | 45 | strict_numeric |  | From November 1963 to July 1969 Alan Shepard was Chief of the Astronaut Office. | A one-month discrepancy between July and August 1969 is minor; the system over-applied precision standards to a date range claim. |
| fever-20603 | FR | 0.95 | 2 | 5 | strict_numeric | yes | Yemen does not have an area of 203,850 square miles. | The 54 sq mi difference (0.03%) is within normal rounding tolerance, so the claim's negation is refuted by the evidence. |
| fever-106996 | MR | 0.92 | 4 | 25 | strict_semantic |  | Liam Neeson has been passed over by the British Academy of Film and Television Arts. | The system conflated 'nominated but not won' with 'passed over by,' misinterpreting the claim's semantic meaning about institutional treatment. |
| fever-130248 | FR | 0.85 | 6 | 39 | strict_semantic |  | Aldous Huxley barely published scripts. | 'Barely published' means few were produced/released, not that none existed. Disney rejected Huxley's Alice screenplay; Madame Curie credit was uncredited. |
| fever-142863 | FR | 0.92 | 2 | 18 | strict_semantic |  | Coeliac disease is treated by maintaining a healthy diet. | System conflated 'healthy diet' (genus) with 'gluten-free diet' (species), treating semantic specificity as factual contradiction. |
| fever-18030 | FR | 0.98 | 3 | 16 | strict_semantic |  | Floyd Mayweather Jr. has made 30 career boxing knockouts in different special event profes | The system ignored the claim's qualifier 'in different special event professional boxing fights,' which narrows the scope and may make 30 accurate for that subset. |
| fever-18079 | FR | 0.92 | 6 | 39 | strict_semantic |  | Scandinavia includes the three kingdoms of Denmark, Norway, and Sweden as of 1902. | The claim's substance—Scandinavia comprises Denmark, Norway, Sweden—is correct. The system conflated regional geography with political sovereignty, a semantic distinction. |
| fever-35237 | FR | 0.95 | 6 | 31 | strict_semantic |  | Philomena is a film nominated for seven awards. | The system interpreted 'seven awards' as seven nominations total, but the claim likely means seven award ceremonies/categories, which the evidence supports. |
| fever-82973 | FR | 0.85 | 6 | 36 | strict_semantic |  | Netflix is who Trollhunters was created for. | The system conflates 'created for' (Netflix as commissioning entity) with 'created by' (Guillermo del Toro). Netflix originals are created for Netflix. |
| fever-91498 | FR | 0.99 | 1 | 8 | strict_semantic |  | Tiber Oil Field is a landlocked oil field. | The system conflated 'located in Gulf of Mexico' (administrative/geographic designation) with being in open water, missing that offshore oil fields are subsurface and landlocked. |
| fever-27416 | FR | 0.95 | 5 | 32 | temporal_staleness |  | University of Chicago Law School is ranked first for "Faculty quality based on American Ac | The dataset was built from 2017 Wikipedia; rankings may have changed since then. The system's evidence may reflect a later state. |
| fever-66539 | FR | 0.92 | 5 | 40 | temporal_staleness |  | John Frusciante released eleven solo albums. | The gold label reflects 2017 Wikipedia state; the system's evidence shows a later count of thirteen albums, indicating post-2017 releases. |
