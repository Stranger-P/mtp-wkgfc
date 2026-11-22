#!/usr/bin/env python3
"""Contested-case tribunal -- PACT module M3.

Adjudicates only those claims where the closed-book LLM (`direct`) and the
WKGFC agent disagree.  It replays over saved predictions, so no retrieval is
repeated and no Wikidata / Serper call is made.

Flags --provenance (M1) and --temporal (M2) switch the extra context on and
off, so the three modules can be ablated independently.

  python tools/tribunal.py --dataset liar_new \
      --direct outputs/liar_direct_n200 --wkgfc outputs/liar_wkgfc_n200 \
      --out outputs/tribunal_liar.json --provenance --temporal
"""
from __future__ import annotations

import argparse, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
LABELS = ("SUPPORTED", "REFUTED")

# ---------------------------------------------------------------- data

def load_preds(d: Path) -> dict:
    out = {}
    for line in (d / "predictions.jsonl").open():
        r = json.loads(line)
        out[r["id"]] = r
    return out


def load_dates(dataset: str, root: Path) -> dict:
    """Claim dates, where the dataset carries them."""
    dates = {}
    if dataset == "liar_new":
        p = root / "liar_new" / "LIAR-New.jsonl"
        for line in p.open():
            r = json.loads(line)
            dates[f"liarnew-{r['example_id']}"] = r.get("date", "")
    elif dataset == "averitec":
        p = root / "averitec" / "dev.json"
        for i, r in enumerate(json.loads(p.read_text())):
            dates[f"averitec-{i}"] = r.get("claim_date", "")
    return dates


def domain(url: str) -> str:
    try:
        h = urlparse(url).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def render_evidence(rec: dict, provenance: bool, max_triples: int = 30,
                    max_annot_chars: int = 2200) -> str:
    ev = rec.get("evidence") or {}
    triples = ev.get("triples") or []
    web = [t for t in triples if t.get("source") != "kg"]
    kg = [t for t in triples if t.get("source") == "kg"]
    web.sort(key=lambda t: -(t.get("confidence") or 0))
    lines = []
    if web:
        lines.append("WEB EVIDENCE")
        for t in web[:max_triples]:
            tag = ""
            if provenance:
                d = domain(t.get("url", ""))
                tag = f"  [source: {d or 'unknown'}]"
            lines.append(
                f"- {t['subject']} -- {t['relation']} -> {t['obj']}"
                f"  (confidence {t.get('confidence', 0):.2f}){tag}")
    if kg:
        lines.append("")
        lines.append("KNOWLEDGE-GRAPH EVIDENCE (Wikidata)")
        for t in kg[:max_triples]:
            lines.append(f"- {t['subject']} -- {t['relation']} -> {t['obj']}")
    annots = ev.get("annotations") or {}
    if annots:
        lines.append("")
        lines.append("PASSAGES RETRIEVED")
        used = 0
        for key, vals in annots.items():
            for v in (vals if isinstance(vals, list) else [vals]):
                s = f"- ({key}) {v}"
                if used + len(s) > max_annot_chars:
                    break
                lines.append(s)
                used += len(s)
    return "\n".join(lines) if lines else "(no evidence was retrieved)"


# ---------------------------------------------------------------- prompt

CHECKS = """Work through these three checks before ruling. They are the failure
modes measured on this system, so answer each one explicitly.

1. ECHO CHECK. For each piece of evidence that would support the claim, decide
   whether it STATES the fact or merely REPORTS THAT SOMEONE ASSERTED it. A
   campaign page, party site, advocacy group, opinion piece or social post
   repeating the claim is evidence that the claim was MADE, not that it is
   TRUE. Evidence of this kind cannot by itself support a verdict of SUPPORTED.

2. TEMPORAL CHECK. Decide whether the evidence describes the period the claim
   is about. Evidence about a later period does not refute a claim that was
   accurate when it was made, and a figure that has since changed is not a
   refutation of the earlier figure.

3. MISMATCH CHECK. If the case for REFUTED rests on a difference between the
   claim and the evidence, decide whether that difference is SUBSTANTIVE, or
   whether it is only paraphrase, extra specificity, a vague quantifier, or
   rounding within a trivial margin. Only a substantive difference refutes."""

PROMPT = """You are adjudicating a disputed fact-check. Two independent checkers
examined the same claim and reached opposite verdicts. Decide which verdict is
correct.

CLAIM: {claim}
{dateline}
POSITION 1 -- {v1}
{j1}

POSITION 2 -- {v2}
{j2}

EVIDENCE AVAILABLE
{evidence}

{checks}

Then rule. A claim is SUPPORTED only if the evidence establishes it; if the
evidence merely echoes the claim, or does not settle it, say so in your reason
and rule on the balance of what the evidence does establish.

Reply with JSON only:
{{"echo": "<one sentence>",
  "temporal": "<one sentence>",
  "mismatch": "<one sentence, or 'n/a'>",
  "verdict": "SUPPORTED" or "REFUTED",
  "decided_by": 1 or 2,
  "confidence": <0.0-1.0>,
  "reason": "<one or two sentences>"}}"""


def build_prompt(rec_d, rec_w, date, provenance, temporal, order):
    a = ("direct", rec_d)
    b = ("wkgfc", rec_w)
    first, second = (a, b) if order == 0 else (b, a)
    dateline = ""
    if temporal and date:
        dateline = (f"\nTHE CLAIM WAS MADE ON: {date}\nJudge the claim as of "
                    f"that date, using what was true then.\n")
    return PROMPT.format(
        claim=rec_w["claim"],
        dateline=dateline,
        v1=first[1]["pred"], j1=(first[1].get("justification") or "").strip()[:1800],
        v2=second[1]["pred"], j2=(second[1].get("justification") or "").strip()[:1800],
        evidence=render_evidence(rec_w, provenance),
        checks=CHECKS,
    ), {1: first[0], 2: second[0]}


# ---------------------------------------------------------------- llm

def load_dotenv(path=".env"):
    """Bedrock credentials live in .env (AWS_BEARER_TOKEN_BEDROCK)."""
    f = Path(path)
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k.strip(), v)


def make_client(region):
    import boto3
    from botocore.config import Config
    return boto3.client("bedrock-runtime", region_name=region,
                        config=Config(retries={"max_attempts": 8, "mode": "adaptive"},
                                      read_timeout=120))


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return json.loads(m.group(0)[:m.group(0).rindex("}") + 1])
        except Exception:
            return None


def call(client, prompt, model, max_tokens=900):
    last = None
    for attempt in range(5):
        try:
            r = client.converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0})
            return r["output"]["message"]["content"][0]["text"]
        except Exception as e:  # throttling and transient faults
            last = e
            time.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"bedrock failed: {last}")


# ---------------------------------------------------------------- metrics

def bacc(rows):
    labs = sorted({g for g, _ in rows})
    if not labs:
        return 0.0
    tot = 0.0
    for l in labs:
        sup = [p for g, p in rows if g == l]
        tot += (sum(1 for p in sup if p == l) / len(sup)) if sup else 0.0
    return 100 * tot / len(labs)


def mcnemar(rows_a, rows_b):
    """Continuity-corrected McNemar over two aligned prediction lists."""
    import math
    b = sum(1 for (g, x), (_, y) in zip(rows_a, rows_b) if x == g and y != g)
    c = sum(1 for (g, x), (_, y) in zip(rows_a, rows_b) if x != g and y == g)
    if b + c == 0:
        return b, c, 1.0
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, math.erfc(math.sqrt(chi / 2))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--direct", required=True)
    ap.add_argument("--wkgfc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="datasets")
    ap.add_argument("--provenance", action="store_true", help="M1: show source domains")
    ap.add_argument("--temporal", action="store_true", help="M2: show claim date, anchor to it")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--region", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    load_dotenv()

    D = load_preds(Path(a.direct))
    W = load_preds(Path(a.wkgfc))
    dates = load_dates(a.dataset, Path(a.data_root))
    keys = [k for k in W if k in D and W[k]["pred"] in LABELS and D[k]["pred"] in LABELS]
    disputed = [k for k in keys if D[k]["pred"] != W[k]["pred"]]
    if a.limit:
        disputed = disputed[:a.limit]
    rng = random.Random(a.seed)
    orders = {k: rng.randint(0, 1) for k in disputed}

    # resume: rulings already made are kept, so a run can be done in chunks
    cache_path = Path(a.out).with_suffix(".rulings.jsonl")
    rulings_done = {}
    if cache_path.exists():
        for line in cache_path.open():
            try:
                r = json.loads(line)
                rulings_done[r["id"]] = r["ruling"]
            except Exception:
                pass
    todo = [k for k in disputed if k not in rulings_done]
    if rulings_done:
        print(f"[tribunal] resuming: {len(rulings_done)} already ruled, "
              f"{len(todo)} to do", flush=True)

    print(f"[tribunal] {a.dataset}: {len(keys)} claims, {len(disputed)} disputed "
          f"({100*len(disputed)/max(len(keys),1):.0f}%), provenance={a.provenance} "
          f"temporal={a.temporal}", flush=True)

    client = make_client(a.region or os.environ.get("AWS_REGION", "us-east-1"))
    rulings = dict(rulings_done)

    def work(k):
        prompt, who = build_prompt(D[k], W[k], dates.get(k, ""), a.provenance,
                                   a.temporal, orders[k])
        txt = call(client, prompt, a.model)
        j = parse_json(txt) or {}
        v = j.get("verdict")
        if v not in LABELS:
            v = W[k]["pred"]          # unparseable -> keep the agent's verdict
            j["parse_failed"] = True
        dec = j.get("decided_by")
        j["sided_with"] = who.get(dec) if dec in (1, 2) else None
        j["verdict"] = v
        return k, j

    t0 = time.time()
    with cache_path.open("a") as cf, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(work, k): k for k in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            k, j = fut.result()
            rulings[k] = j
            cf.write(json.dumps({"id": k, "ruling": j}) + "\n")
            cf.flush()
            ok = "ok " if j["verdict"] == W[k]["gold"] else "ERR"
            print(f"  {n:3d}/{len(todo)} {ok} gold={W[k]['gold']:9s} "
                  f"tribunal={j['verdict']:9s} sided={j.get('sided_with')}", flush=True)

    gold = {k: W[k]["gold"] for k in keys}
    rows_direct = [(gold[k], D[k]["pred"]) for k in keys]
    rows_wkgfc = [(gold[k], W[k]["pred"]) for k in keys]
    rows_pact = [(gold[k], rulings[k]["verdict"] if k in rulings else W[k]["pred"])
                 for k in keys]
    rows_oracle = [(gold[k], gold[k] if gold[k] in (D[k]["pred"], W[k]["pred"])
                    else W[k]["pred"]) for k in keys]

    sided = [j.get("sided_with") for j in rulings.values()]
    res = {
        "dataset": a.dataset, "n": len(keys), "disputed": len(disputed),
        "provenance": a.provenance, "temporal": a.temporal, "model": a.model,
        "balanced_accuracy": {
            "direct": round(bacc(rows_direct), 2),
            "wkgfc": round(bacc(rows_wkgfc), 2),
            "pact": round(bacc(rows_pact), 2),
            "oracle": round(bacc(rows_oracle), 2)},
        "tribunal_right_on_disputed": sum(
            1 for k in rulings if rulings[k]["verdict"] == gold[k]),
        "sided_with": {"direct": sided.count("direct"), "wkgfc": sided.count("wkgfc"),
                       "unstated": sided.count(None)},
        "parse_failures": sum(1 for j in rulings.values() if j.get("parse_failed")),
        "mcnemar_vs_wkgfc": dict(zip(("b", "c", "p"), mcnemar(rows_wkgfc, rows_pact))),
        "mcnemar_vs_direct": dict(zip(("b", "c", "p"), mcnemar(rows_direct, rows_pact))),
        "seconds": round(time.time() - t0, 1),
        "rulings": {k: rulings[k] for k in rulings},
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    b = res["balanced_accuracy"]
    print(f"\n[tribunal] direct {b['direct']}  wkgfc {b['wkgfc']}  "
          f"PACT {b['pact']}  oracle {b['oracle']}")
    print(f"[tribunal] tribunal correct on {res['tribunal_right_on_disputed']}"
          f"/{len(disputed)} disputed; sided {res['sided_with']}")
    print(f"[tribunal] McNemar vs wkgfc: {res['mcnemar_vs_wkgfc']}")
    print(f"[tribunal] -> {a.out}  ({res['seconds']}s)")


if __name__ == "__main__":
    sys.exit(main())
