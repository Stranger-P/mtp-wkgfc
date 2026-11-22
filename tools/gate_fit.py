#!/usr/bin/env python3
"""Fit the flip gate for the confirmation audit on a dev split, report on test.

The auditor flags far more inconsistencies than exist, so the flip must be
gated.  This sweeps gates over the audit's own signals (category, confidence),
chooses the best on a dev half, and reports it on a held-out half, so the gain
is not fitted on the numbers it is reported against.

  python tools/gate_fit.py --audit outputs/pact/summeval_confirm.json \
      --run outputs/summeval_direct_gold
"""
from __future__ import annotations
import argparse, json, math, random
from itertools import combinations
from pathlib import Path

TYPES = ["entity", "added", "number", "relation", "overstatement"]


def bacc(rows):
    labs = sorted({g for g, _ in rows})
    if not labs:
        return 0.0
    t = 0.0
    for l in labs:
        s = [p for g, p in rows if g == l]
        t += sum(1 for p in s if p == l) / len(s) if s else 0
    return 100 * t / len(labs)


def mcnemar(a, b):
    bb = sum(1 for (g, x), (_, y) in zip(a, b) if x == g and y != g)
    cc = sum(1 for (g, x), (_, y) in zip(a, b) if x != g and y == g)
    if bb + cc == 0:
        return bb, cc, 1.0
    chi = (abs(bb - cc) - 1) ** 2 / (bb + cc)
    return bb, cc, round(math.erfc(math.sqrt(chi / 2)), 4)


def evaluate(keys, P, A, allow, thr):
    before, after = [], []
    flips = 0
    good = 0
    for k in keys:
        g, p = P[k]["gold"], P[k]["pred"]
        before.append((g, p))
        a = A.get(k) or {}
        fire = (a.get("found") and a.get("verdict") == "INCONSISTENT"
                and str(a.get("quote_from_claim", "")).strip()
                and a.get("type") in allow
                and (a.get("confidence") or 0) >= thr)
        if fire and p == "SUPPORTED":
            after.append((g, "REFUTED"))
            flips += 1
            good += (g == "REFUTED")
        else:
            after.append((g, p))
    return before, after, flips, good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    res = json.loads(Path(a.audit).read_text())
    A = res["audits"]
    P = {}
    for line in (Path(a.run) / "predictions.jsonl").read_text(errors="ignore").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        P[r["id"]] = r

    # Split the WHOLE dataset in half, so each half carries both classes and
    # both verdicts.  Measuring only on audited claims is meaningless: they
    # were all predicted SUPPORTED, so balanced accuracy there is 50.0 by
    # construction and any flip looks like a huge gain.
    allk = sorted(P)
    rng = random.Random(a.seed)
    rng.shuffle(allk)
    half = len(allk) // 2
    dev, test = sorted(allk[:half]), sorted(allk[half:])
    cov = 100 * sum(1 for k in allk if k in A) / max(len(allk), 1)
    print(f"{len(allk)} claims -> dev {len(dev)}, test {len(test)}; "
          f"{len(A)} audited ({cov:.0f}% coverage)")

    best = None
    print(f"\n{'gate':44s} {'dev bAcc':>9s} {'flips':>6s} {'prec':>6s}")
    print("-" * 70)
    base = bacc([(P[k]['gold'], P[k]['pred']) for k in dev])
    print(f"{'(no audit)':44s} {base:9.2f} {0:6d} {'-':>6s}")
    for r_ in range(1, len(TYPES) + 1):
        for allow in combinations(TYPES, r_):
            for thr in (0.0, 0.8, 0.9, 0.95):
                b, af, fl, gd = evaluate(dev, P, A, set(allow), thr)
                s = bacc(af)
                prec = 100 * gd / fl if fl else 0
                if best is None or s > best[0]:
                    best = (s, allow, thr, fl, prec)
                if fl and s > base + 0.5:
                    print(f"{'+'.join(allow)+' @'+str(thr):44s} {s:9.2f} "
                          f"{fl:6d} {prec:5.1f}%")

    s, allow, thr, fl, prec = best
    print(f"\nbest on dev: types={allow} confidence>={thr} "
          f"({fl} flips, precision {prec:.1f}%, dev {base:.2f} -> {s:.2f})")

    b, af, fl, gd = evaluate(test, P, A, set(allow), thr)
    bb = bacc(b)
    aa = bacc(af)
    mn = mcnemar(b, af)
    print(f"\nHELD-OUT TEST: balanced accuracy {bb:.2f} -> {aa:.2f} ({aa-bb:+.2f})")
    print(f"  {fl} flips, {gd} correct (precision {100*gd/fl if fl else 0:.1f}%)")
    print(f"  McNemar b={mn[0]} c={mn[1]} p={mn[2]}")


if __name__ == "__main__":
    main()
