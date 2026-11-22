"""Central configuration. Every experiment is a Config; ablation variants just flip flags."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a hard dependency on python-dotenv)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


@dataclass
class Config:
    # --- LLM -------------------------------------------------------------
    llm_backend: str = "openai"          # "openai" | "bedrock" | "gemini" (free, dev only) | "mock"
    llm_model: str = "gpt-4o"            # paper setting; use a cheaper model while developing
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    openai_base_url: str | None = None   # set to an OpenAI-compatible server (Ollama/vLLM) for local models
    llm_fallback_models: list = field(default_factory=list)  # tried in order if the main model is busy/missing.
                                         # Testing only: keep EMPTY for thesis runs (one model per reported number).

    # --- KG retrieval (paper Sec. 4.2, 5.6) --------------------------------
    beam_size_k: int = 4                 # top-k relations/entities kept per hop
    init_hops: int = 2                   # hops performed by initKGRetrieval
    max_hops_N: int = 4                  # total hop budget (initial + expandKG)
    max_relations_per_entity: int = 60   # candidate relations shown to the pruning LLM
    max_objects_per_relation: int = 10   # values fetched per kept relation
    language: str = "en"
    kg_parallel_requests: int = 4        # concurrent Wikidata/LLM requests per hop; 1 = original sequential order

    # --- Web retrieval (paper Sec. 4.3) ------------------------------------
    search_backend: str = "serper"       # "serper" | "mock"
    web_results_per_query: int = 10
    bm25_top_passages: int = 8           # coarse filter keeps this many passages
    min_web_confidence: float = 0.5      # fine LLM filter threshold
    fetch_full_pages: bool = False       # paper uses snippets; True = download page text
    blocked_domains: tuple = (
        "politifact.com", "snopes.com", "factcheck.org", "fullfact.org",
        "leadstories.com", "checkyourfact.com", "africacheck.org",
    )

    # --- Agent (paper Sec. 4.1) --------------------------------------------
    use_kg: bool = True
    use_web: bool = True
    use_agent: bool = True               # False = fixed pipeline (ablation Variants 1/2/4)

    # --- extensions beyond the base paper (all default off, so "wkgfc" stays the paper's system) ---
    verdict_tolerance: bool = False       # tell the verdict prompt what does NOT count as refutation
    audit_refutations: bool = False       # re-examine every REFUTED verdict with one adversarial pass
    # Which mismatch labels may overturn a refutation. Measured on FEVER-200 by replaying the audit
    # over all 112 refusals: numeric_tolerance fires 2/2 on true errors, while specificity (1 fix /
    # 7 breaks), vague_quantifier (2/9), missing_evidence (3/16) and temporal (0/8) are net-negative.
    audit_overturn_on: tuple = ("numeric_tolerance",)
    audit_skip_negated: bool = True       # never overturn a claim containing a negation
    max_steps: int = 6                   # agent actions before the forced verdict
    policy_prompt_path: str | None = None  # optimised prompt from TextGrad (Sec. 4.4)

    # --- Infra ---------------------------------------------------------------
    cache_dir: str = ".cache"
    output_dir: str = "outputs"
    wikidata_user_agent: str = "MTP-WKGFC/0.1 (student research; contact in .env)"
    request_timeout: int = 60
    seed: int = 42
    verbose: bool = False

    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | None = None, **overrides) -> "Config":
        _load_dotenv(Path(".env"))
        data: dict = {}
        if path:
            data = json.loads(Path(path).read_text())
        data.update({k: v for k, v in overrides.items() if v is not None})
        cfg = cls(**data)
        ua = os.environ.get("WIKIDATA_USER_AGENT")
        if ua:
            cfg.wikidata_user_agent = ua
        cache_dir = os.environ.get("WKGFC_CACHE_DIR")  # e.g. a local disk when the project is on a shared drive
        if cache_dir:
            cfg.cache_dir = cache_dir
        return cfg

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=list)


# Ablation variants from paper Table 3.
VARIANTS = {
    "kg_only": dict(use_kg=True, use_web=False, use_agent=False),            # Variant 1
    "web_only": dict(use_kg=False, use_web=True, use_agent=False),           # Variant 2
    "kg_agent": dict(use_kg=True, use_web=False, use_agent=True),            # Variant 3
    "kg_web_no_agent": dict(use_kg=True, use_web=True, use_agent=False),     # Variant 4
    "wkgfc_no_optim": dict(use_kg=True, use_web=True, use_agent=True, policy_prompt_path=None),  # Variant 5
    "wkgfc": dict(use_kg=True, use_web=True, use_agent=True),                # full (pass --policy-prompt)
    # --- our extensions, as ablatable variants ---
    "wkgfc_tol": dict(use_kg=True, use_web=True, use_agent=True, verdict_tolerance=True),
    "wkgfc_audit": dict(use_kg=True, use_web=True, use_agent=True, audit_refutations=True),
    "wkgfc_tol_audit": dict(use_kg=True, use_web=True, use_agent=True, verdict_tolerance=True,
                            audit_refutations=True),
}
