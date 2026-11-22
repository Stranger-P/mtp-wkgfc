"""Dataset loaders reproducing the paper's benchmark (Table 2): binary labels SUPPORTED / REFUTED,
samples without verifiable evidence removed, partially-correct / conflicting merged into REFUTED.

Each example: {"id", "claim", "label", "dataset", "date"?, "gold_evidence"?}.
Sizes and refuted rates follow Table 2; the paper does not publish its exact samples, so we draw
them with a fixed seed (record the seed in your thesis)."""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

# name -> (sample size, refuted ratio) from paper Table 2 (None = use all)
TABLE2 = {
    "fever": (1332, 0.50),
    "hover": (1332, 0.50),
    "liar_new": (200, 0.76),
    "averitec": (None, None),       # 465 after filtering
    "summeval": (None, None),       # 1600
    "aggrefact_cnn": (None, None),  # 558
    "pubhealth": (None, None),      # 1231 (paper: 1232)
}


def _stratified(examples: list[dict], n: int | None, neg_rate: float | None, seed: int) -> list[dict]:
    if n is None or n >= len(examples):
        return examples
    rng = random.Random(seed)
    neg = [e for e in examples if e["label"] == "REFUTED"]
    pos = [e for e in examples if e["label"] == "SUPPORTED"]
    n_neg = round(n * neg_rate)
    n_pos = n - n_neg
    if n_neg > len(neg) or n_pos > len(pos):
        raise ValueError(f"not enough examples: need {n_pos}+/{n_neg}-, have {len(pos)}+/{len(neg)}-")
    out = rng.sample(pos, n_pos) + rng.sample(neg, n_neg)
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
def load_fever(root: Path) -> list[dict]:
    path = root / "fever" / "shared_task_dev.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: download https://fever.ai/download/fever/shared_task_dev.jsonl")
    out = []
    for line in path.open():
        r = json.loads(line)
        lab = {"SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED"}.get(r["label"])
        if lab:  # NOT ENOUGH INFO removed
            out.append({"id": f"fever-{r['id']}", "claim": r["claim"], "label": lab, "dataset": "fever"})
    return out


def load_hover(root: Path, split: str = "dev") -> list[dict]:
    data = json.loads((root / "hover" / f"hover_{split}_release_v1.1.json").read_text())
    return [{"id": f"hover-{r['uid']}", "claim": r["claim"], "dataset": "hover", "num_hops": r.get("num_hops"),
             "label": "SUPPORTED" if r["label"] == "SUPPORTED" else "REFUTED"} for r in data if "label" in r]


def load_liar_new(root: Path) -> list[dict]:
    out = []
    for line in (root / "liar_new" / "LIAR-New.jsonl").open():
        r = json.loads(line)
        if r.get("possibility_label") == "i":  # impossible to verify -> removed
            continue
        lab = "SUPPORTED" if r["label"] in ("true", "mostly-true") else "REFUTED"
        out.append({"id": f"liarnew-{r['example_id']}", "claim": r["statement"].strip("“”\" "), "label": lab,
                    "dataset": "liar_new", "date": r.get("date", "")})
    return out


def load_averitec(root: Path) -> list[dict]:
    out = []
    for i, r in enumerate(json.loads((root / "averitec" / "dev.json").read_text())):
        if r["label"] == "Not Enough Evidence":
            continue
        lab = "SUPPORTED" if r["label"] == "Supported" else "REFUTED"  # Conflicting -> REFUTED
        out.append({"id": f"averitec-{i}", "claim": r["claim"], "label": lab, "dataset": "averitec",
                    "date": r.get("claim_date", ""), "speaker": r.get("speaker", "")})
    return out


def _load_gold(root: Path, fname: str, name: str) -> list[dict]:
    with open(root / "gold_evidence" / fname, "rb") as f:
        df = pickle.load(f)  # pandas DataFrame from the GraphCheck release
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        out.append({"id": f"{name}-{i}", "claim": str(row.claim_text), "dataset": name,
                    "label": "SUPPORTED" if int(row.label) == 1 else "REFUTED",
                    "gold_evidence": str(row.doc_text)})
    return out


LOADERS = {
    "fever": load_fever,
    "hover": load_hover,
    "liar_new": load_liar_new,
    "averitec": load_averitec,
    "summeval": lambda root: _load_gold(root, "summeval.pkl", "summeval"),
    "aggrefact_cnn": lambda root: _load_gold(root, "AggreFact-CNN.pkl", "aggrefact_cnn"),
    "pubhealth": lambda root: _load_gold(root, "pubhealth.pkl", "pubhealth"),
}


def load_benchmark(name: str, data_root: str, seed: int = 42, limit: int | None = None,
                   exclude_ids: set | None = None) -> list[dict]:
    """Paper-sized test set for `name` (Table 2), optionally truncated to `limit` for development."""
    root = Path(data_root)
    examples = LOADERS[name](root)
    if exclude_ids:
        examples = [e for e in examples if e["id"] not in exclude_ids]
    n, neg = TABLE2[name]
    sample = _stratified(examples, n, neg, seed)
    if limit:  # keep the label balance when truncating
        sample = _stratified(sample, limit, sum(e["label"] == "REFUTED" for e in sample) / len(sample), seed)
    return sample


def load_optimisation_pool(data_root: str, seed: int = 42, n: int = 150) -> tuple[list[dict], list[dict]]:
    """150 non-test claims from FEVER + HOVER for prompt optimisation (paper Sec. 4.4): 100 train / 50 val.
    HOVER claims come from the HOVER *train* split; FEVER claims from dev claims NOT in the FEVER test sample."""
    root = Path(data_root)
    rng = random.Random(seed + 1)
    hover = _stratified(load_hover(root, "train"), n // 2, 0.5, seed + 1)
    try:
        test_ids = {e["id"] for e in load_benchmark("fever", data_root, seed)}
        fever_pool = [e for e in load_fever(root) if e["id"] not in test_ids]
        fever = _stratified(fever_pool, n - len(hover), 0.5, seed + 1)
    except FileNotFoundError:
        fever = []
        hover = _stratified(load_hover(root, "train"), n, 0.5, seed + 1)
    pool = hover + fever
    rng.shuffle(pool)
    n_train = round(len(pool) * 2 / 3)
    return pool[:n_train], pool[n_train:]
