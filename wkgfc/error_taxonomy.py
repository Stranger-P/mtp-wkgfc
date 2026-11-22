"""Error taxonomy for wrong predictions.

The paper's Table 4 buckets (IK / EM / OC) are defined by agent behaviour and leave most
errors unlabelled: on our FEVER-200 runs they cover only half of them. This module adds a
*cause* taxonomy over the same wrong records, so every error lands in exactly one bucket.

Each wrong record is classified by an LLM from the claim, the gold label, the system's own
justification and a sample of the evidence it used. The categories below are mutually
exclusive and ordered: the first that applies wins. `label_noise` comes first on purpose,
because a dataset error must not be counted against the system.

The classification is LLM-assisted, not ground truth. 20 errors is small enough to audit by
hand: `errors.md` is written for exactly that, and `--no-llm` gives the deterministic part only.
"""
from __future__ import annotations

import json

CATEGORIES: dict[str, str] = {
    "strict_semantic": "The evidence supports the substance of the claim, but the system refuted it over "
                       "wording, specificity, a paraphrase, or a vague quantifier (e.g. 'barely', 'created for').",
    "strict_numeric": "The system refuted over a number or date that differs only slightly from the claim, "
                      "or over a rounding, unit or precision difference.",
    "temporal_staleness": "The evidence describes a later state of the world than the claim's reference time: "
                          "the claim was true when the dataset was built and the retrieved evidence reflects now.",
    "entity_linking": "The system resolved the claim to the wrong entity (wrong film, wrong person, wrong "
                      "work of the same name).",
    "retrieval_miss": "The evidence needed to decide was never retrieved, or the wrong fact was retrieved, so "
                      "the verdict rested on an absence or on an unrelated fact.",
    "reasoning_error": "The retrieved evidence was adequate but the inference was wrong: a mishandled negation, "
                       "a faulty aggregation, or a non-sequitur.",
    "vague_claim": "The claim is too vague, subjective or underspecified to be settled by any evidence.",
    "label_noise": "Last resort. The dataset's gold label is indefensible on a verifiable external fact "
                   "(not merely debatable, and not just because the system argued well), and the system's "
                   "verdict was the correct one.",
}

ERROR_CLASSIFY = """You are auditing the errors of an automated fact-checking system.

For this claim the system's verdict disagreed with the dataset's gold label. Decide WHY.

Claim: {claim}
Gold label (dataset): {gold}
System verdict: {pred}
System's justification: {justification}

Evidence the system used (sample):
{evidence}

The system's justification is an argument for its own verdict, so do not treat its fluency as
proof. First write the strongest case FOR the gold label that a careful annotator would give,
then classify.

Two things to weigh before you choose:
- The dataset was built from Wikipedia in 2017. If the claim was plausibly true then and the
  retrieved evidence reflects a later state of the world (a count that has since grown, a role
  someone has since left, a ranking that has since moved), that is `temporal_staleness`, not a
  dataset error.
- If the claim names a work, person or place whose name is shared by others, and the system may
  have resolved a different one, that is `entity_linking`.

Choose exactly one category, the first in this list that applies:
{categories}

Reply with JSON only:
{{"case_for_gold": "<the strongest case for the gold label, under 25 words>",
  "category": "<one key from the list>",
  "system_actually_right": true/false,
  "reason": "<one sentence, under 30 words>"}}"""


def _evidence_sample(rec: dict, n: int = 12) -> str:
    triples = rec.get("evidence", {}).get("triples", [])
    lines = []
    for t in triples[:n]:
        src = t.get("source", "?")
        lines.append(f"- ({t.get('subject')}, {t.get('relation')}, {t.get('obj')}) [{src}]")
    anns = rec.get("evidence", {}).get("annotations", {})
    for ent, notes in list(anns.items())[:4]:
        for note in (notes or [])[:2]:
            text = note if isinstance(note, str) else note.get("text", str(note))
            lines.append(f"- note on {ent}: {str(text)[:200]}")
    return "\n".join(lines) or "(no evidence retrieved)"


def _signals(rec: dict) -> dict:
    """Deterministic facts about the error, available without an LLM."""
    f = rec.get("flags", {})
    triples = rec.get("evidence", {}).get("triples", [])
    return {
        "id": rec.get("id"),
        "direction": "false_refutation" if rec["gold"] == "SUPPORTED" else "missed_refutation",
        "confidence": rec.get("confidence"),
        "forced": bool(rec.get("forced")),
        "steps": f.get("steps"),
        "retrieval_actions": f.get("retrieval_actions"),
        "requested_web": bool(f.get("requested_web")),
        "insufficient_kg": bool(f.get("insufficient_kg")),
        "over_confidence_pattern": bool(f.get("over_confidence_pattern")),
        "llm_calls": rec.get("cost", {}).get("llm_calls"),
        "kg_triples": sum(1 for t in triples if t.get("source") == "kg"),
        "web_triples": sum(1 for t in triples if t.get("source") == "web"),
    }


def classify_errors(records: list[dict], llm=None) -> list[dict]:
    """One row per wrong prediction: deterministic signals plus (if llm given) a cause category."""
    cats = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    rows = []
    for rec in records:
        if rec["pred"] == rec["gold"]:
            continue
        row = _signals(rec)
        row["claim"] = rec["claim"]
        row["gold"], row["pred"] = rec["gold"], rec["pred"]
        if llm is not None:
            out = llm.complete_json(
                ERROR_CLASSIFY.format(claim=rec["claim"], gold=rec["gold"], pred=rec["pred"],
                                      justification=(rec.get("justification") or "")[:1500],
                                      evidence=_evidence_sample(rec), categories=cats),
                purpose="error_taxonomy", default={})
            cat = (out or {}).get("category")
            row["category"] = cat if cat in CATEGORIES else "unclassified"
            row["system_actually_right"] = bool((out or {}).get("system_actually_right"))
            row["reason"] = (out or {}).get("reason", "")
            row["case_for_gold"] = (out or {}).get("case_for_gold", "")
        rows.append(row)
    return rows


def taxonomy_report(records: list[dict], rows: list[dict]) -> dict:
    n_wrong = len(rows) or 1
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r.get("category", "unclassified")] = by_cat.get(r.get("category", "unclassified"), 0) + 1
    directions = {d: sum(1 for r in rows if r["direction"] == d) for d in ("false_refutation", "missed_refutation")}
    dataset_fault = sum(1 for r in rows if r.get("category") == "label_noise")
    strict = sum(1 for r in rows if r.get("category") in ("strict_semantic", "strict_numeric"))
    return {
        "n": len(records),
        "wrong": len(rows),
        "direction": directions,
        "by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1])),
        "by_category_%": {k: round(100 * v / n_wrong, 1)
                          for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "dataset_fault": dataset_fault,
        "strict_match_refutations": strict,
        "accuracy_excluding_label_noise": round(
            100 * (len(records) - (len(rows) - dataset_fault)) / (len(records) or 1), 2),
        "mean_llm_calls_by_direction": {
            d: round(sum(r["llm_calls"] or 0 for r in rows if r["direction"] == d) / max(1, directions[d]), 1)
            for d in directions},
    }


def report_markdown(report: dict, rows: list[dict]) -> str:
    out = [f"# Error taxonomy ({report['wrong']} wrong of {report['n']})", ""]
    out.append(f"- false refutations: **{report['direction']['false_refutation']}**, "
               f"missed refutations: **{report['direction']['missed_refutation']}**")
    out.append(f"- errors that are the dataset's fault: **{report['dataset_fault']}**")
    out.append(f"- refutations on a strict wording/number match: **{report['strict_match_refutations']}**")
    out.append(f"- accuracy excluding gold-label noise: **{report['accuracy_excluding_label_noise']}%**")
    out += ["", "| cause | n | % of errors |", "|---|---|---|"]
    for k, v in report["by_category"].items():
        out.append(f"| {k} | {v} | {report['by_category_%'][k]} |")
    out += ["", "## Every error, for hand audit", "",
            "| id | dir | conf | steps | calls | cause | system right? | claim | why |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r.get("category", ""), r["id"])):
        claim = r["claim"].replace("|", "/")
        out.append(f"| {r['id']} | {'FR' if r['direction'] == 'false_refutation' else 'MR'} | {r['confidence']} "
                   f"| {r['steps']} | {r['llm_calls']} | {r.get('category', '-')} "
                   f"| {'yes' if r.get('system_actually_right') else ''} | {claim[:90]} "
                   f"| {r.get('reason', '')} |")
    return "\n".join(out) + "\n"


def load_predictions(run_dir) -> list[dict]:
    from pathlib import Path
    return [json.loads(l) for l in open(Path(run_dir) / "predictions.jsonl") if l.strip()]
