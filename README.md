# WKGFC reimplementation (base paper of the MTP)

Reimplementation of **Gong et al., "Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking", SIGIR '26**
(arXiv 2603.00267). The authors released no code; every module below follows the paper's Section 4.

| Paper section | File | What it does |
|---|---|---|
| 4.1 Agentic module (POMDP) | `wkgfc/agent.py` | initKGRetrieval -> loop {expandKG, webSearch, verdict} -> forced verdict at max steps; trajectory + error flags |
| 4.2 KG retrieval | `wkgfc/entities.py`, `wkgfc/wikidata.py`, `wkgfc/kg_retrieval.py` | spaCy NER -> `wbsearchentities` -> SPARQL in/out templates -> LLM relation + entity pruning (beam k, hops N), adapted from Think-on-Graph (paper ref. 35) |
| 4.3 Web retrieval | `wkgfc/web_retrieval.py` | Serper -> BM25 coarse filter -> LLM consistency filter -> web triples aligned to KG schema (dupes skipped, details as annotations); fact-check sites blocked |
| 4.4 Policy improvement | `wkgfc/optimise.py` | composite reward, self-critique, experience buffer, TextGrad-style prompt optimisation (100 train / 50 val) |
| 5.x Evaluation | `wkgfc/data.py`, `wkgfc/metrics.py`, `wkgfc/run.py` | Table 2 datasets, balanced accuracy, macro-F1, Table 4 error analysis (IK/EM/OC), cost per claim, McNemar test |
| Prompts | `wkgfc/prompts.py` | all prompts; `AGENT_POLICY` is the optimised policy θ |

## Setup (on your machine)

```bash
cd base_paper_impl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf     # or en_core_web_sm for speed
cp .env.example .env                          # then put your OPENAI_API_KEY and SERPER_API_KEY in .env
python tests/test_offline.py                  # offline tests (no keys needed)
python -m wkgfc.run datasets                  # checks all 7 datasets against Table 2
```

## Running experiments

```bash
# 1. One claim, full trace (sanity check, ~20-40 LLM calls)
python -m wkgfc.run check "Douglas Adams was educated at St John's College, Cambridge."

# 2. Develop on a small slice with a cheap model (results cached in .cache/, runs resume)
python -m wkgfc.run evaluate --dataset fever --variant kg_only --limit 50 --model gpt-4o-mini
python -m wkgfc.run evaluate --dataset fever --variant wkgfc   --limit 50 --model gpt-4o-mini

# 3. Prompt optimisation (paper: 150 FEVER+HOVER non-test claims, 20 epochs) - start small
python -m wkgfc.run optimise --epochs 3 --batch 10 --model gpt-4o-mini
python -m wkgfc.run evaluate --dataset fever --variant wkgfc --policy-prompt outputs/policy_gpt-4o-mini/best_policy.json

# 4. Baselines and ablations (paper Tables 1 and 3)
python -m wkgfc.run direct   --dataset fever                               # closed-book LLM
python -m wkgfc.run evaluate --dataset fever --variant web_only            # Variant 2
python -m wkgfc.run evaluate --dataset fever --variant kg_agent            # Variant 3
python -m wkgfc.run evaluate --dataset fever --variant kg_web_no_agent     # Variant 4
python -m wkgfc.run evaluate --dataset fever --variant wkgfc_no_optim      # Variant 5

# 5. Parameter study (paper Fig. 3, on kg_only)
python -m wkgfc.run evaluate --dataset hover --variant kg_only --k 3 --hops 2

# 6. Significance between two runs
python -m wkgfc.run compare outputs/<runA> outputs/<runB>
```

Each run writes `outputs/<name>/predictions.jsonl` (verdict, justification, trajectory, evidence, cost per
claim) and `summary.json` (balanced accuracy, macro-F1, forced-verdict rate, IK/EM/OC error shares, average
LLM calls / tokens / SPARQL queries / web searches per claim).

Local models: any OpenAI-compatible server works, e.g. Ollama
`--base-url http://localhost:11434/v1 --model llama3.1:8b` (set `OPENAI_API_KEY=ollama`).

## Choices the paper leaves open (state these in the thesis)

| Item | Our choice | Where |
|---|---|---|
| Hops done by initKGRetrieval vs later expandKG | 2 initial hops, total budget N=4 | `Config.init_hops`, `max_hops_N` |
| Max agent steps | 6 | `Config.max_steps` |
| Entity linking | top `wbsearchentities` hit (optional LLM disambiguation `--disambiguate`) | `entities.py` |
| Relations treated as noise | IDs, URLs, Commons/Wikimedia meta (from Think-on-Graph's list) | `wikidata.useless_relation` |
| Web passages | Serper title+snippet (full pages optional: `fetch_full_pages`) | `web_retrieval.py` |
| Fact-checking sites | blocked to avoid verdict leakage (paper silent) | `Config.blocked_domains` |
| Reward weights | 0.7 correctness + 0.2 sufficiency + 0.1 efficiency | `optimise.composite_reward` |
| TextGrad | same procedure (critique -> textual gradient -> prompt update), implemented directly | `optimise.py` |
| Test samples | Table 2 sizes and refuted rates, drawn with seed 42 | `data.py` |
| LIAR-New binarisation | true/mostly-true = SUPPORTED; possibility "i" removed | `data.py` |

## Not yet verified live
Offline tests cover all logic with a fake Wikidata, fake search and scripted LLM. The live SPARQL, Serper
and OpenAI calls have not been run yet (no keys / network in the build environment): run the `check`
command first and fix anything Wikidata-specific before large runs.

## Extensions beyond the base paper

All extensions default to **off**, so `--variant wkgfc` remains the paper's system.

| command / flag | what it does |
|---|---|
| `run errors <run_dir>` | classifies a finished run's wrong predictions by cause (`error_taxonomy.json`, `errors.md`) |
| `--tolerance` / `--variant wkgfc_tol` | tells the verdict prompt what does *not* count as a refutation |
| `--audit` / `--variant wkgfc_audit` | re-examines every REFUTED verdict with one adversarial pass |
| `tools/replay_audit.py <run_dir>` | applies the audit offline to a finished run's saved evidence, one call per refusal |
| `tools/audit_rules.py <run_dir>` | sweeps overturn rules over a saved replay, no API calls |

### What we measured on FEVER-200 (Haiku 4.5)

1. **Errors are asymmetric and reproducible.** 16 false refutations against 4 missed refutations,
   identical in the unoptimised and optimised runs. REFUTED precision 85.7% at 96.0% recall; the
   closed-book baseline is 81.9% at 95.0%. The framework's whole gain is fewer false refutations.
2. **Confidence cannot find them.** Correct refutations average 0.961, false ones 0.939, and the
   distributions overlap; every threshold breaks more correct refutations than it fixes.
3. **Prompt optimisation does not transfer.** Validation 82.7% → 86.1%, test flat at 90.0%: it
   changed 8 of 200 predictions, fixed 4, broke 4, and raised cost (0.77 → 0.96 web searches).
4. **The strict-match signature is not discriminative.** Hand adjudication of the 20 errors puts 8
   of them on a wording or precision mismatch, but replaying the audit over all 112 refusals shows
   the same signature fires just as often on correct refutations: as an error detector,
   `specificity` scores 12.5% precision — the same as `substantive` — and `temporal` scores 0%
   (8 flagged, none were errors). Classifying only the errors was selection bias.
5. **One rule survives.** Overturning on `numeric_tolerance` alone gives +2 claims (90.0 → 91.0)
   with recall unchanged, so `audit_overturn_on` defaults to that single label, plus a guard that
   never overturns a claim containing a negation.
6. **Gold-label noise is real.** 4 of 20 errors are the dataset's fault (Franklin's birth date is
   his death date; Tiber Oil Field is in the Gulf of Mexico), putting the true ceiling near 92%.

Taken together: FEVER has little headroom left for this framework, and the temporal hypothesis
cannot be tested on it. Run LIAR-New and AVeriTeC before building further mechanisms.

### Caveat on the error taxonomy

`run errors` is LLM-assisted triage, not ground truth. It agreed with hand adjudication on only
**10 of 20** errors, and its label-noise count moved from 7 to 3 when the prompt was changed to
state the case for the gold label first. Audit `errors.md` by hand; `errors_adjudicated.md` holds
one such adjudication.
