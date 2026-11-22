"""All prompts in one place. AGENT_POLICY is the trainable policy text θ that TextGrad
optimises (paper Sec. 4.4); everything else stays fixed."""

LABELS = ("SUPPORTED", "REFUTED")

# ---------------------------------------------------------------------------
# KG retrieval: expand-and-prune (adapted from Think-on-Graph, the paper's ref. [35])
# ---------------------------------------------------------------------------
RELATION_PRUNE = """You are exploring the Wikidata knowledge graph to find evidence for verifying a claim.
Select the {k} relations of the topic entity that are most useful for verifying the claim, and score each
from 0 to 1 (scores of the selected relations sum to 1). "(incoming)" means other entities point to this one
with that relation.

Claim: {claim}
Topic entity: {entity}
Relations:
{relations}

Return JSON only: {{"relations": [{{"id": <number from the list>, "score": <float>}}, ...]}}"""

ENTITY_SCORE = """Score how useful each candidate entity is for verifying the claim, from 0 to 1
(scores sum to 1).

Claim: {claim}
Path: {head} --[{relation}]--> candidates
Candidates:
{candidates}

Return JSON only: {{"scores": [<float for candidate 1>, <float for candidate 2>, ...]}}"""

ENTITY_DISAMBIGUATION = """Pick the Wikidata item that the mention refers to in this claim.

Claim: {claim}
Mention: {mention}
Candidates:
{candidates}

Return JSON only: {{"choice": <candidate number, or 0 if none fits>}}"""

ENTITY_EXTRACTION_FALLBACK = """List the named entities (people, organisations, places, works, events, concepts)
in the claim that should be looked up in Wikidata.

Claim: {claim}

Return JSON only: {{"entities": ["...", "..."]}}"""

# ---------------------------------------------------------------------------
# Agentic module (POMDP policy, paper Sec. 4.1). This text is θ.
# ---------------------------------------------------------------------------
AGENT_POLICY = """You are a fact-checking detective. You verify a claim by gathering evidence step by step from
a knowledge graph (Wikidata) and, when needed, the web. At each step you see the claim, the evidence gathered
so far and your previous actions, and you choose ONE action:

- "expandKG": explore the knowledge graph one hop further from the current frontier entities. Use it when the
  evidence is missing a relational link that Wikidata is likely to contain (people, places, dates, roles,
  works, organisations).
- "webSearch": search the web with a focused query. Use it when the knowledge graph lacks the needed fact
  (recent events, quotes, statistics, obscure details) or after further hops stop adding useful evidence.
  Write the query to target exactly the missing information.
- "verdict": stop and decide. Use it only when the evidence is sufficient to support or refute EVERY part of
  the claim.

Guidelines:
- Check each part of the claim (entities, relations, numbers, dates) against the evidence before concluding.
- Knowledge-graph triples are high-precision anchors; web evidence is useful but may be noisy or wrong.
  If they conflict, prefer the knowledge graph unless the web evidence is clearly more specific and recent.
- A claim is REFUTED if any essential part is contradicted or false. Partly correct claims are REFUTED.
- Do not stop early just because the claim sounds plausible; do not keep searching when the evidence is
  already conclusive."""

AGENT_STEP = """Claim: {claim}

Evidence so far:
{evidence}

Previous actions: {history}
Steps used: {step} of {max_steps}. Remaining KG hops: {hops_left}.{web_note}

Decide the next action. Return JSON only:
{{"thought": "<what is established, what is missing>",
  "sufficient": <true|false>,
  "action": "expandKG" | "webSearch" | "verdict",
  "query": "<web query if action is webSearch, else empty>",
  "verdict": "SUPPORTED" | "REFUTED" | "",
  "confidence": <0-1>,
  "justification": "<if verdict: cite the triples that decide it>"}}"""

FORCED_VERDICT = """You have used all allowed retrieval steps. Based ONLY on the evidence below (and careful
reasoning), decide whether the claim is SUPPORTED or REFUTED.

Claim: {claim}

Evidence:
{evidence}

Return JSON only: {{"verdict": "SUPPORTED" | "REFUTED", "confidence": <0-1>, "justification": "<cite evidence>"}}"""

# Fixed-pipeline verdict (ablation variants without the agent)
PIPELINE_VERDICT = """Decide whether the claim is SUPPORTED or REFUTED by the evidence. A claim is REFUTED if any
essential part is contradicted or false. If the evidence does not settle it, use careful reasoning.

Claim: {claim}

Evidence:
{evidence}

Return JSON only: {{"verdict": "SUPPORTED" | "REFUTED", "confidence": <0-1>, "justification": "<cite evidence>"}}"""

# Closed-book LLM baseline (paper Table 1 "Large-scale LLMs")
DIRECT_VERDICT = """Is the following claim factually SUPPORTED or REFUTED? Use your own knowledge.

Claim: {claim}
{gold}
Return JSON only: {{"verdict": "SUPPORTED" | "REFUTED", "confidence": <0-1>, "justification": "..."}}"""

# ---------------------------------------------------------------------------
# Web retrieval (paper Sec. 4.3)
# ---------------------------------------------------------------------------
WEB_QUERY_FROM_CLAIM = """Write one concise web search query (max 12 words) that would find evidence to verify
this claim.

Claim: {claim}

Return JSON only: {{"query": "..."}}"""

WEB_FILTER = """You are filtering web passages as evidence for a claim. For each passage, judge whether it is
relevant to verifying the claim, and how confident you are that its factual content is reliable and
consistent (0-1). Copy the 1-3 sentences that matter.

Claim: {claim}

Passages:
{passages}

Return JSON only:
{{"passages": [{{"id": <number>, "relevant": <true|false>, "confidence": <0-1>, "key_sentences": "..."}}, ...]}}"""

WEB_TO_TRIPLES = """Convert the evidence sentences into knowledge triples (subject, relation, object) aligned with
the knowledge-graph schema below: reuse the existing entity names and relation labels whenever they fit,
otherwise use short Wikidata-style relation labels (e.g. "position held", "date of birth", "member of").
If a sentence adds detail about an existing entity that does not fit a triple, return it as an annotation.

Claim: {claim}

Existing entities: {entities}
Existing relation labels: {relations}

Evidence sentences (numbered by source):
{sentences}

Return JSON only:
{{"triples": [{{"subject": "...", "relation": "...", "object": "...", "source": <number>}}],
  "annotations": [{{"entity": "...", "text": "...", "source": <number>}}]}}"""

# ---------------------------------------------------------------------------
# Self-reflection (paper Sec. 4.4)
# ---------------------------------------------------------------------------
SELF_CRITIQUE = """You are reviewing a fact-checking agent's decision trajectory.

Claim: {claim}
Gold label: {gold}
Predicted: {pred}  ({outcome})
Trajectory:
{trajectory}

Final evidence:
{evidence}

Diagnose the retrieval decisions. Use these error types where they apply:
insufficient_coverage, premature_termination, redundant_expansion, contradictory_evidence_mishandled,
bad_web_query, reasoning_error, none.

Return JSON only: {{"error_types": ["..."], "critique": "<2-4 sentences on what the policy should have done>"}}"""


# --- Extensions beyond the base paper -----------------------------------------------------------
# Motivation (measured on FEVER-200, both runs): 16 of 20 errors are false refutations, and about
# half of those are refutations on a wording, specificity or numeric-precision mismatch rather than
# a substantive contradiction. A confidence threshold cannot separate them (good and bad
# refutations both sit at 0.95-0.99), so the mismatch has to be re-examined on its content.

TOLERANCE_NOTE = """
A claim is REFUTED only when the evidence contradicts something essential to it. These are NOT
refutations:
- the evidence states the same fact in different words;
- the claim is more general and the evidence gives a specific instance of it;
- the claim turns on a vague word ("barely", "many", "a few") that the evidence does not clearly defeat;
- a number or date differs only by rounding, unit conversion or a trivial margin;
- the evidence simply does not mention the claim (absence of evidence is not contradiction).
Take care with claims that contain a negation ("did not", "does not", "is not"): there, the
evidence contradicting the inner statement SUPPORTS the claim."""

REFUTATION_AUDIT = """A fact-checking system has decided to REFUTE this claim. Your job is to check
whether the mismatch it found is substantive, or only a difference of wording or precision.

Claim: {claim}
System's reasoning for refuting: {justification}

Evidence:
{evidence}

First write the strongest case FOR the claim that a careful annotator would give from this evidence.
Then classify the mismatch the system relied on:
- substantive: the evidence contradicts something essential to the claim
- paraphrase: the evidence states the same fact in different words
- specificity: the claim is more general; the evidence gives an instance of it
- vague_quantifier: the claim turns on a vague word the evidence does not clearly defeat
- numeric_tolerance: the number or date differs only by rounding, units or a trivial margin
- missing_evidence: the refutation rests on the evidence not mentioning the claim
- temporal: the evidence describes a later state of the world than the claim refers to

Keep REFUTED unless you can name a positive reason from the list above to overturn it. If the claim
contains a negation, re-read it before deciding: overturning a negated claim is easy to get wrong.

Return JSON only:
{{"case_for_claim": "<under 25 words>",
  "mismatch": "<one label from the list>",
  "verdict": "SUPPORTED" | "REFUTED",
  "confidence": <0-1>,
  "justification": "<one sentence>"}}"""
