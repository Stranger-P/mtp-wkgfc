"""Evaluation metrics: balanced accuracy (paper's primary metric), macro-F1, error analysis (Table 4)."""
from __future__ import annotations

LABELS = ("SUPPORTED", "REFUTED")


def balanced_accuracy(gold: list[str], pred: list[str]) -> float:
    recalls = []
    for c in LABELS:
        idx = [i for i, g in enumerate(gold) if g == c]
        if idx:
            recalls.append(sum(pred[i] == c for i in idx) / len(idx))
    return sum(recalls) / len(recalls) if recalls else 0.0


def macro_f1(gold: list[str], pred: list[str]) -> float:
    f1s = []
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def per_class(gold: list[str], pred: list[str]) -> dict:
    """Precision/recall per label. On a two-class task balanced accuracy hides the asymmetry
    between crying wolf (low REFUTED precision) and missing refutations (low REFUTED recall)."""
    out = {}
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        out[c] = {
            "precision": round(100 * tp / (tp + fp), 2) if tp + fp else 0.0,
            "recall": round(100 * tp / (tp + fn), 2) if tp + fn else 0.0,
            "support": tp + fn,
        }
    return out


def error_analysis(records: list[dict]) -> dict:
    """Paper Sec. 5.7 / Table 4, over WRONG predictions only:
    IK = insufficient KG (web requested after the KG was enhanced),
    EM = exceeded maximum steps (forced verdict),
    OC = over-confidence (verdict straight after the initial KG, no extra retrieval).
    Categories can overlap, as in the paper."""
    wrong = [r for r in records if r["pred"] != r["gold"]]
    n = len(wrong)
    if n == 0:
        return {"wrong": 0}
    ik = sum(bool(r.get("flags", {}).get("insufficient_kg")) for r in wrong)
    em = sum(bool(r.get("forced")) for r in wrong)
    oc = sum(bool(r.get("flags", {}).get("over_confidence_pattern")) for r in wrong)
    exec_err = sum(bool(r.get("error")) for r in wrong)
    return {"wrong": n, "IK_%": round(100 * ik / n, 1), "EM_%": round(100 * em / n, 1),
            "OC_%": round(100 * oc / n, 1), "execution_error_%": round(100 * exec_err / n, 1)}


def summarise(records: list[dict]) -> dict:
    gold = [r["gold"] for r in records]
    pred = [r["pred"] or "REFUTED" for r in records]
    n = len(records) or 1
    cost_keys = ("llm_calls", "llm_calls_uncached", "prompt_tokens", "completion_tokens", "sparql_queries",
                 "web_searches")
    avg_cost = {k: round(sum(r.get("cost", {}).get(k, 0) for r in records) / n, 2) for k in cost_keys}
    return {
        "n": len(records),
        "balanced_accuracy": round(100 * balanced_accuracy(gold, pred), 2),
        "macro_f1": round(100 * macro_f1(gold, pred), 2),
        "per_class": per_class(gold, pred),
        "accuracy": round(100 * sum(g == p for g, p in zip(gold, pred)) / n, 2),
        "forced_verdict_%": round(100 * sum(bool(r.get("forced")) for r in records) / n, 1),
        "avg_seconds": round(sum(r.get("seconds", 0) for r in records) / n, 2),
        "avg_cost_per_claim": avg_cost,
        "errors": error_analysis(records),
    }


def mcnemar(gold: list[str], pred_a: list[str], pred_b: list[str]) -> dict:
    """McNemar test (continuity-corrected chi-square) for comparing two systems on the same claims."""
    import math

    b = sum(pa == g and pb != g for g, pa, pb in zip(gold, pred_a, pred_b))
    c = sum(pa != g and pb == g for g, pa, pb in zip(gold, pred_a, pred_b))
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))  # survival function of chi2 with 1 dof
    return {"b": b, "c": c, "chi2": round(chi2, 3), "p": round(p, 4)}
