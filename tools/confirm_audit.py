#!/usr/bin/env python3
"""Confirmation audit -- the mirror of the refutation audit.

Measured bias: on document-grounded datasets the verifier under-refutes
(SummEval misses 240 inconsistent summaries of 1600).  This pass re-examines
only the claims already judged SUPPORTED, hunting for a concrete inconsistency,
and flips the verdict only when one is quoted.

  python tools/confirm_audit.py --dataset summeval \
      --run outputs/summeval_direct_gold --out outputs/pact/summeval_confirm.json \
      --limit 400
"""
from __future__ import annotations
import argparse, json, math, os, pickle, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
L = ("SUPPORTED", "REFUTED")
PKL = {"summeval": "summeval.pkl", "aggrefact_cnn": "AggreFact-CNN.pkl",
       "pubhealth": "pubhealth.pkl"}

AUDIT = """You are re-checking a claim against the source document it must be
grounded in. A first pass judged the claim CONSISTENT with the document. That
pass is known to miss inconsistencies, so your task is to look specifically for
one it may have overlooked.

DOCUMENT
{doc}

CLAIM
{claim}

Check each of these in order, against the document above:
1. ENTITY -- is every person, place and organisation the same one the document
   puts in that role? A swapped name, or the right name in the wrong role, is
   an inconsistency.
2. NUMBER OR DATE -- does every figure, quantity, age and date match?
3. RELATION -- does the document actually state the relation the claim asserts
   (who did what, to whom, in which direction)?
4. ADDED FACT -- does the claim assert something the document does not contain
   at all?
5. OVERSTATEMENT -- does the claim turn something alleged, reported, hedged or
   possible into a plain fact?

A paraphrase, a compression, a pronoun resolved to its referent, or an
inference the document directly licenses is NOT an inconsistency. Do not
manufacture one: most claims are consistent, and a false alarm is as costly as
a miss.

Reply with JSON only:
{{"found": true or false,
  "type": "entity" | "number" | "relation" | "added" | "overstatement" | "none",
  "quote_from_claim": "<the exact words at fault, or empty>",
  "document_says": "<what the document says instead, or empty>",
  "verdict": "CONSISTENT" or "INCONSISTENT",
  "confidence": <0.0-1.0>}}"""


def load_preds(d: Path):
    out = {}
    for line in (d / "predictions.jsonl").read_text(errors="ignore").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("pred") in L:
            out[r["id"]] = r
    return out


def load_docs(dataset: str, root: Path):
    with open(root / "gold_evidence" / PKL[dataset], "rb") as f:
        df = pickle.load(f)
    return {f"{dataset}-{i}": str(row.doc_text)
            for i, row in enumerate(df.itertuples(index=False))}


def load_dotenv(path=".env"):
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


def call(client, prompt, model, max_tokens=500):
    last = None
    for attempt in range(5):
        try:
            r = client.converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0})
            return r["output"]["message"]["content"][0]["text"]
        except Exception as e:
            last = e
            time.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"bedrock failed: {last}")


def parse_json(t):
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def bacc(rows):
    labs = sorted({g for g, _ in rows})
    if not labs:
        return 0.0
    tot = 0.0
    for l in labs:
        s = [p for g, p in rows if g == l]
        tot += sum(1 for p in s if p == l) / len(s) if s else 0
    return 100 * tot / len(labs)


def mcnemar(a, b):
    bb = sum(1 for (g, x), (_, y) in zip(a, b) if x == g and y != g)
    cc = sum(1 for (g, x), (_, y) in zip(a, b) if x != g and y == g)
    if bb + cc == 0:
        return bb, cc, 1.0
    chi = (abs(bb - cc) - 1) ** 2 / (bb + cc)
    return bb, cc, round(math.erfc(math.sqrt(chi / 2)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(PKL))
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="datasets")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--region", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="audit at most this many")
    ap.add_argument("--max-doc-chars", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    load_dotenv()

    P = load_preds(Path(a.run))
    docs = load_docs(a.dataset, Path(a.data_root))
    keys = sorted(P)
    rng = random.Random(a.seed)
    cand = [k for k in keys if P[k]["pred"] == "SUPPORTED" and k in docs]
    if a.limit and len(cand) > a.limit:
        cand = rng.sample(cand, a.limit)
        cand.sort()
    scope = set(cand)

    cache = Path(a.out).with_suffix(".audits.jsonl")
    done = {}
    if cache.exists():
        for line in cache.open():
            try:
                r = json.loads(line)
                done[r["id"]] = r["audit"]
            except Exception:
                pass
    todo = [k for k in cand if k not in done]
    print(f"[confirm] {a.dataset}: {len(keys)} predictions, "
          f"{sum(1 for k in keys if P[k]['pred']=='SUPPORTED')} judged SUPPORTED, "
          f"auditing {len(cand)} ({len(done)} already done, {len(todo)} to do)",
          flush=True)

    client = make_client(a.region or os.environ.get("AWS_REGION", "us-east-1"))
    audits = dict(done)

    def work(k):
        prompt = AUDIT.format(doc=docs[k][:a.max_doc_chars], claim=P[k]["claim"])
        j = parse_json(call(client, prompt, a.model)) or {}
        flip = bool(j.get("found")) and j.get("verdict") == "INCONSISTENT" \
            and bool(str(j.get("quote_from_claim", "")).strip())
        j["flip"] = flip
        return k, j

    t0 = time.time()
    with cache.open("a") as cf, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(work, k): k for k in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            k, j = fut.result()
            audits[k] = j
            cf.write(json.dumps({"id": k, "audit": j}) + "\n")
            cf.flush()
            if n % 25 == 0 or n == len(todo):
                nf = sum(1 for v in audits.values() if v.get("flip"))
                print(f"  {n}/{len(todo)} audited, {nf} flipped "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # evaluate on the audited subset only, so before/after are comparable
    sub = [k for k in keys if k in scope or P[k]["pred"] == "REFUTED"]
    before = [(P[k]["gold"], P[k]["pred"]) for k in sub]
    after = [(P[k]["gold"],
              "REFUTED" if audits.get(k, {}).get("flip") else P[k]["pred"])
             for k in sub]
    flips = [k for k in scope if audits.get(k, {}).get("flip")]
    good = sum(1 for k in flips if P[k]["gold"] == "REFUTED")
    types = {}
    for k in flips:
        t = audits[k].get("type", "?")
        types[t] = types.get(t, 0) + 1

    res = {"dataset": a.dataset, "run": a.run, "model": a.model,
           "n_evaluated": len(sub), "n_audited": len(scope), "n_flipped": len(flips),
           "flips_correct": good, "flips_wrong": len(flips) - good,
           "flip_precision": round(100 * good / len(flips), 1) if flips else None,
           "flip_types": types,
           "balanced_accuracy": {"before": round(bacc(before), 2),
                                 "after": round(bacc(after), 2)},
           "accuracy": {
               "before": round(100 * sum(1 for g, p in before if g == p) / len(before), 2),
               "after": round(100 * sum(1 for g, p in after if g == p) / len(after), 2)},
           "mcnemar": dict(zip(("b", "c", "p"), mcnemar(before, after))),
           "seconds": round(time.time() - t0, 1)}
    Path(a.out).write_text(json.dumps({**res, "audits": audits}, indent=2))
    b = res["balanced_accuracy"]
    print(f"\n[confirm] balanced accuracy {b['before']} -> {b['after']} "
          f"({b['after']-b['before']:+.2f})")
    print(f"[confirm] flipped {res['n_flipped']} of {res['n_audited']}; "
          f"{good} were truly REFUTED (precision {res['flip_precision']}%)")
    print(f"[confirm] flip types: {types}")
    print(f"[confirm] McNemar: {res['mcnemar']}")
    print(f"[confirm] -> {a.out}  ({res['seconds']}s)")


if __name__ == "__main__":
    sys.exit(main())
