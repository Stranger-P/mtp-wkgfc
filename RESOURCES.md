# Base paper (WKGFC) - resources collected

Paper: Gong et al., "Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking", SIGIR '26 (arXiv 2603.00267).
No official code released. Collected on 2026-11-17.

## 1. Datasets (datasets/)

| Dataset | File(s) | What we have | Paper uses (Table 2) | Status |
|---|---|---|---|---|
| FEVER | fever/ (empty) | - | 1,332 claims, 50% refuted | **MANUAL DOWNLOAD NEEDED** (see below) |
| HOVER | hover/hover_dev_release_v1.1.json | 4,000 dev claims (2,000 SUPPORTED / 2,000 NOT_SUPPORTED; 2-4 hops). Test file has no labels. | 1,332 claims, 50% refuted | OK - sample 1,332 balanced from dev |
| LIAR-New | liar_new/LIAR-New.jsonl (+ articles) | 1,957 claims, 6-way PolitiFact labels + date + possibility_label | 200 claims, 76% refuted | OK - binarise (true/mostly-true = Supported, rest = Refuted), sample 200 |
| AVeriTeC | averitec/dev.json (+ train.json) | 500 dev claims: 305 Refuted, 122 Supported, 38 Conflicting, 35 NEE | 465 claims, 73.7% refuted | OK - drop 35 NEE, merge Conflicting into Refuted = 465 (73.8%) - matches paper |
| SummEval | gold_evidence/summeval.pkl | 1,600 rows (doc_text, claim_text, label, + KGs) | 1,600 | OK - exact match |
| AggreFact-CNN | gold_evidence/AggreFact-CNN.pkl | 558 rows | 558 | OK - exact match |
| PubHealth | gold_evidence/pubhealth.pkl | 1,231 rows | 1,232 | OK (1 row difference) |

Gold-evidence .pkl files come from the GraphCheck repo (pandas DataFrames).

### FEVER - download yourself
Easiest: `https://fever.ai/download/fever/shared_task_dev.jsonl` (labelled dev, 19,998 claims; ~4 MB).
Mirror: `https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl`
Or Hugging Face: `copenlu/fever_gold_evidence`. Put the file in datasets/fever/.
Then: drop NOT ENOUGH INFO, sample 1,332 balanced SUPPORTS/REFUTES.

### Exact samples
The paper does not publish which 1,332 / 200 claims it used -> ask the authors, or fix a random seed and report it.

## 2. Reference code (references/) - what to reuse

| Repo | Why it matters | Reuse |
|---|---|---|
| ToG (Think-on-Graph, ICLR'24) | **The paper's expand-and-prune is taken from this [ref 35].** `ToG/wiki_func.py`, `ToG/utils.py`: relation_search_prune, entity_search, entity_score, entity_prune; `prompt_list.py` pruning prompts | Core of initKGRetrieval/expandKG. Swap its local Wikidata server for public SPARQL |
| ToG-2 (ICLR'25) | KG + text hybrid retrieval [ref 25] | Ideas for KG<->web interleaving |
| fire (FIRE, NAACL'25) | Iterative retrieve-or-verdict agent with Serper; same group (Nakov, Xie) as WKGFC | Agent loop, confidence-based stopping, Serper wrapper |
| textgrad | Prompt optimisation used by the paper | Policy-improvement step |
| GraphCheck (ACL'25) | Baseline + gold-evidence datasets | Datasets; baseline numbers |
| ClaimPKG (ACL'25 Findings) | KG claim verification, pseudo-subgraphs | Entity/subgraph retrieval ideas |
| HerO2 (AVeriTeC 2025) | Strong web-RAG baseline (HerO) | Baseline, web pipeline |
| reasoning-on-graphs (RoG) | KG reasoning baseline [ref 24] | Optional |

Note: each repo folder has a leftover partial `.git` folder from a failed clone - safe to delete.

## 3. APIs / keys needed (copy .env.example to .env)

| Service | Needed for | Key? | Cost |
|---|---|---|---|
| OpenAI API | All LLM steps (pruning, agent, web filter, verdict, TextGrad) | Yes | Paid per token |
| Serper (serper.dev) | webSearch() | Yes | 2,500 free queries |
| Wikidata wbsearchentities + SPARQL | Entity linking, KG expansion | No (set a User-Agent with your email) | Free, 60 s/query limit |

## 4. Python setup
pip install -r requirements.txt
python -m spacy download en_core_web_trf
