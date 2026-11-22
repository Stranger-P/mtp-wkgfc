"""Replay the refutation audit over a finished run, without re-running retrieval.

Every prediction record stores the evidence graph the agent actually saw, so the audit pass can be
applied offline to each REFUTED verdict. That measures the audit's two effects separately -
how many false refutations it overturns, and how many correct ones it breaks - for one LLM call
per refusal instead of a full re-run.

  python tools/replay_audit.py outputs/<run> --backend bedrock --fallback-models ""
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wkgfc.agent import FactChecker                      # noqa: E402
from wkgfc.cache import Cache                            # noqa: E402
from wkgfc.config import Config, _load_dotenv            # noqa: E402
from wkgfc.kg import EvidenceGraph, Triple               # noqa: E402
from wkgfc.llm import make_llm                           # noqa: E402
from wkgfc.metrics import balanced_accuracy, per_class   # noqa: E402

TRIPLE_FIELDS = ("subject", "relation", "obj", "source", "subject_id", "relation_id", "obj_id", "url", "confidence")


def graph_from_record(rec: dict) -> EvidenceGraph:
    ev = rec.get("evidence", {}) or {}
    g = EvidenceGraph()
    for t in ev.get("triples", []):
        g.add(Triple(**{k: t[k] for k in TRIPLE_FIELDS if k in t}))
    for entity, notes in (ev.get("annotations") or {}).items():
        for note in notes or []:
            g.annotate(entity, note if isinstance(note, str) else str(note))
    g.topic_entities = dict(ev.get("topic_entities") or {})
    return g


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--backend", default="bedrock")
    p.add_argument("--model", default=None)
    p.add_argument("--fallback-models", default="")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--cache-dir", default=None)
    a = p.parse_args(argv)

    _load_dotenv(Path(".env"))
    import os
    model = a.model or os.environ.get(f"{a.backend.upper()}_MODEL") or os.environ.get("LLM_MODEL")
    cfg = Config(llm_backend=a.backend, llm_model=model, audit_refutations=True,
                 llm_fallback_models=[m.strip() for m in a.fallback_models.split(",") if m.strip()])
    if a.cache_dir:
        cfg.cache_dir = a.cache_dir
    run = Path(a.run_dir)
    recs = [json.loads(l) for l in (run / "predictions.jsonl").open() if l.strip()]
    refusals = [r for r in recs if r["pred"] == "REFUTED"]
    print(f"{len(recs)} claims, {len(refusals)} predicted REFUTED -> {len(refusals)} audit calls")

    cache = Cache(cfg.cache_dir, "llm")
    checker = FactChecker(cfg, make_llm(cfg, cache))

    def one(rec):
        label, conf, just, audit = checker._audit_refutation(
            rec["claim"], graph_from_record(rec), "REFUTED", rec.get("confidence", 0.0),
            rec.get("justification", ""))
        return rec["id"], label, conf, just, audit or {}

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(one, refusals))

    by_id = {r[0]: r for r in results}
    rows, new_pred = [], {}
    for rec in recs:
        new_pred[rec["id"]] = rec["pred"]
    for rid, label, conf, just, audit in results:
        new_pred[rid] = label
    for rec in refusals:
        rid, label, conf, just, audit = by_id[rec["id"]]
        rows.append({"id": rid, "claim": rec["claim"], "gold": rec["gold"], "was": "REFUTED", "now": label,
                     "overturned": label != "REFUTED", "mismatch": audit.get("mismatch", ""),
                     "case_for_claim": audit.get("case_for_claim", ""), "justification": just,
                     "correct_before": rec["gold"] == "REFUTED", "correct_after": rec["gold"] == label})

    fixed = [r for r in rows if r["overturned"] and r["correct_after"]]
    broken = [r for r in rows if r["overturned"] and not r["correct_after"]]
    kept_bad = [r for r in rows if not r["overturned"] and not r["correct_before"]]

    gold = [r["gold"] for r in recs]
    before = [r["pred"] for r in recs]
    after = [new_pred[r["id"]] for r in recs]

    report = {
        "refusals_audited": len(rows),
        "overturned": sum(1 for r in rows if r["overturned"]),
        "false_refutations_fixed": len(fixed),
        "correct_refutations_broken": len(broken),
        "false_refutations_missed": len(kept_bad),
        "net_claims": len(fixed) - len(broken),
        "mismatch_counts": {m: sum(1 for r in rows if r["mismatch"] == m)
                            for m in sorted({r["mismatch"] for r in rows})},
        "before": {"balanced_accuracy": round(100 * balanced_accuracy(gold, before), 2),
                   "per_class": per_class(gold, before)},
        "after": {"balanced_accuracy": round(100 * balanced_accuracy(gold, after), 2),
                  "per_class": per_class(gold, after)},
    }
    (run / "audit_replay.json").write_text(json.dumps({"report": report, "rows": rows}, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {run / 'audit_replay.json'}")
    return report


if __name__ == "__main__":
    main()
