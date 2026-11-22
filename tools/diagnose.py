#!/usr/bin/env python3
"""Consolidated diagnostic across every finished run in outputs/.

Prints one row per run (accuracy and the direction of its errors) and one row
per direct/agent pair (how often they disagree, and the oracle ceiling), so it
is obvious which dataset has headroom and what kind of error dominates there.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

L = ("SUPPORTED", "REFUTED")


def load(d: Path):
    p = d / "predictions.jsonl"
    if not p.exists():
        return {}
    out = {}
    try:
        text = p.read_text(errors="ignore")
    except OSError:
        # the Mac is writing this file right now; the mount refuses a read
        return {"__locked__": True}
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("pred") in L:
            out[r["id"]] = r
    return out


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
    return bb, cc, math.erfc(math.sqrt(chi / 2))


def main(root="outputs"):
    root = Path(root)
    runs = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in ("_to_delete", "pact"):
            continue
        P = load(d)
        if P.get("__locked__"):
            print(f"{d.name:34s} (still running - file locked by the Mac)")
            continue
        if len(P) < 20:
            continue
        runs[d.name] = P

    print(f"{'run':34s} {'n':>4s} {'bAcc':>6s} {'acc':>6s} "
          f"{'false-SUP':>10s} {'false-REF':>10s} {'forced%':>8s}")
    print("-" * 84)
    for name, P in runs.items():
        rows = [(r["gold"], r["pred"]) for r in P.values()]
        fs = sum(1 for g, p in rows if g == "REFUTED" and p == "SUPPORTED")
        fr = sum(1 for g, p in rows if g == "SUPPORTED" and p == "REFUTED")
        acc = 100 * sum(1 for g, p in rows if g == p) / len(rows)
        forced = 100 * sum(1 for r in P.values() if r.get("forced")) / len(P)
        print(f"{name:34s} {len(rows):4d} {bacc(rows):6.1f} {acc:6.1f} "
              f"{fs:10d} {fr:10d} {forced:8.1f}")

    pairs = [("fever", "fever_direct_haiku45_n200", "fever_wkgfc_haiku45_n200"),
             ("hover", "hover_direct_n200", "hover_wkgfc_n200"),
             ("liar_new", "liar_direct_n200", "liar_wkgfc_n200"),
             ("averitec", "averitec_direct_n200", "averitec_wkgfc_n200")]
    print(f"\n{'pair':12s} {'n':>4s} {'direct':>7s} {'agent':>7s} {'oracle':>7s} "
          f"{'gap':>6s} {'disag':>6s} {'D-only':>7s} {'A-only':>7s} {'p':>7s}")
    print("-" * 84)
    for tag, a, b in pairs:
        A, B = runs.get(a, {}), runs.get(b, {})
        ks = [k for k in A if k in B]
        if len(ks) < 20:
            print(f"{tag:12s} {'(incomplete)':>20s}")
            continue
        g = {k: A[k]["gold"] for k in ks}
        ra = [(g[k], A[k]["pred"]) for k in ks]
        rb = [(g[k], B[k]["pred"]) for k in ks]
        ro = [(g[k], g[k] if g[k] in (A[k]["pred"], B[k]["pred"]) else B[k]["pred"])
              for k in ks]
        dis = [k for k in ks if A[k]["pred"] != B[k]["pred"]]
        bb, cc, p = mcnemar(ra, rb)
        print(f"{tag:12s} {len(ks):4d} {bacc(ra):7.1f} {bacc(rb):7.1f} {bacc(ro):7.1f} "
              f"{bacc(ro)-max(bacc(ra),bacc(rb)):6.1f} "
              f"{100*len(dis)/len(ks):5.0f}% {bb:7d} {cc:7d} {p:7.3f}")
    print("\ngap = points the oracle has over the better single system "
          "(the headroom a combiner could win)")
    print("D-only / A-only = claims only direct / only the agent gets right")

    # tribunal results, where present
    tri = sorted((root / "pact").glob("*_full.json")) if (root / "pact").is_dir() else []
    if tri:
        print(f"\n{'tribunal':12s} {'disputed':>9s} {'right':>7s} {'PACT':>7s} "
              f"{'agent':>7s} {'direct':>7s} {'oracle':>7s} {'sided-agent':>12s}")
        print("-" * 84)
        for f in tri:
            r = json.loads(f.read_text())
            b = r["balanced_accuracy"]
            s = r["sided_with"]
            print(f"{r['dataset']:12s} {r['disputed']:9d} "
                  f"{r['tribunal_right_on_disputed']:7d} {b['pact']:7.1f} "
                  f"{b['wkgfc']:7.1f} {b['direct']:7.1f} {b['oracle']:7.1f} "
                  f"{s['wkgfc']:5d}/{r['disputed']:<6d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs")
