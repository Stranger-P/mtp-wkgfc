"""Sweep overturn rules over a saved audit replay, to pick which mismatch labels may flip a verdict.

The audit's mismatch label is a diagnosis; whether it should overturn the verdict is an empirical
question. This sweeps candidate rules over an existing audit_replay.json - no further API calls -
and reports, per rule, how many false refutations it fixes against how many correct ones it breaks.

  python tools/audit_rules.py outputs/<run>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wkgfc.metrics import balanced_accuracy, per_class   # noqa: E402

NEGATION = re.compile(r"\b(not|n't|never|no longer|without|fails? to|neither|nor)\b", re.I)
LABELS = ("specificity", "vague_quantifier", "numeric_tolerance", "temporal", "missing_evidence")


def sweep(run_dir: Path) -> dict:
    rows = json.load(open(run_dir / "audit_replay.json"))["rows"]
    recs = [json.loads(l) for l in (run_dir / "predictions.jsonl").open() if l.strip()]
    gold = {r["id"]: r["gold"] for r in recs}

    def score(name, fn):
        pred = {r["id"]: r["pred"] for r in recs}
        fix = brk = 0
        for row in rows:
            if fn(row):
                pred[row["id"]] = "SUPPORTED"
                if gold[row["id"]] == "SUPPORTED":
                    fix += 1
                else:
                    brk += 1
        g = [gold[r["id"]] for r in recs]
        p = [pred[r["id"]] for r in recs]
        pc = per_class(g, p)
        return {"rule": name, "fixed": fix, "broken": brk, "net": fix - brk,
                "balanced_accuracy": round(100 * balanced_accuracy(g, p), 2),
                "refuted_precision": pc["REFUTED"]["precision"], "refuted_recall": pc["REFUTED"]["recall"]}

    results = [score("none (baseline)", lambda r: False),
               score("model's own verdict", lambda r: r["overturned"]),
               score("any non-substantive", lambda r: r["mismatch"] != "substantive")]
    for m in LABELS:
        results.append(score(f"only {m}", lambda r, m=m: r["mismatch"] == m))
        results.append(score(f"only {m}, skip negated",
                             lambda r, m=m: r["mismatch"] == m and not NEGATION.search(r["claim"])))
    # per-label base rates: how discriminative is each diagnosis?
    base = {}
    for m in sorted({r["mismatch"] for r in rows}):
        sub = [r for r in rows if r["mismatch"] == m]
        s = sum(1 for r in sub if gold[r["id"]] == "SUPPORTED")
        base[m] = {"n": len(sub), "would_fix": s, "would_break": len(sub) - s,
                   "precision_as_error_detector": round(100 * s / len(sub), 1) if sub else 0.0}
    out = {"rules": sorted(results, key=lambda r: -r["net"]), "mismatch_base_rates": base}
    (run_dir / "audit_rules.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    o = sweep(Path(a.run_dir))
    print(f"{'rule':38} {'fix':>4} {'brk':>4} {'net':>4} {'bAcc':>6} {'R-prec':>7} {'R-rec':>6}")
    for r in o["rules"]:
        print(f"{r['rule']:38} {r['fixed']:>4} {r['broken']:>4} {r['net']:>+4} {r['balanced_accuracy']:>6} "
              f"{r['refuted_precision']:>7} {r['refuted_recall']:>6}")
    print("\nIs each diagnosis a useful error detector?")
    for m, b in sorted(o["mismatch_base_rates"].items(), key=lambda kv: -kv[1]["precision_as_error_detector"]):
        print(f"  {m:20} n={b['n']:>3}  fixes={b['would_fix']:>2}  breaks={b['would_break']:>3}  "
              f"precision={b['precision_as_error_detector']:>5}%")
