"""Command-line entry point.

  python -m wkgfc.run datasets                                   # check data + Table 2 sizes
  python -m wkgfc.run check "Barack Obama was born in Kenya."    # one claim, full trace
  python -m wkgfc.run evaluate --dataset fever --variant wkgfc --limit 100 --model gpt-4o-mini
  python -m wkgfc.run direct   --dataset fever --limit 100        # closed-book LLM baseline
  python -m wkgfc.run optimise --epochs 20 --batch 10             # TextGrad-style policy optimisation
  python -m wkgfc.run compare outputs/runA outputs/runB          # McNemar significance test
  python -m wkgfc.run errors  outputs/runA                      # error taxonomy over wrong predictions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import DirectLLM, FactChecker
from .cache import Cache
from .config import VARIANTS, Config
from .data import LOADERS, TABLE2, load_benchmark, load_optimisation_pool
from .llm import make_llm
from .metrics import mcnemar, summarise
from .web_retrieval import make_search
from .wikidata import WikidataClient


_CACHES: dict = {}
_CACHES_LOCK = threading.Lock()


def _cache(cfg: Config, name: str) -> Cache:
    """One cache object per (dir, name) in the process, shared by all claim workers (it is lock-protected)."""
    with _CACHES_LOCK:
        key = (cfg.cache_dir, name)
        if key not in _CACHES:
            _CACHES[key] = Cache(cfg.cache_dir, name)
        return _CACHES[key]


def build(cfg: Config):
    llm = make_llm(cfg, _cache(cfg, "llm"))
    kg = WikidataClient(cfg, _cache(cfg, "wikidata")) if cfg.use_kg else None
    search = make_search(cfg, _cache(cfg, "search")) if cfg.use_web else None
    return llm, FactChecker(cfg, llm, kg, search)


def _resolve_models(a) -> None:
    """Pick the model and fallbacks for the chosen backend, so `--backend bedrock` alone is enough."""
    b = a.backend.upper()
    a.model = a.model or os.environ.get(f"{b}_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    if a.fallback_models is None:
        a.fallback_models = os.environ.get(f"{b}_FALLBACK_MODELS", os.environ.get("LLM_FALLBACK_MODELS", ""))


def _cfg_from_args(a) -> Config:
    _resolve_models(a)
    overrides = dict(VARIANTS.get(getattr(a, "variant", None) or "wkgfc", {}))
    overrides.update(llm_backend=a.backend, llm_model=a.model, beam_size_k=a.k, max_hops_N=a.hops, max_steps=a.max_steps,
                     policy_prompt_path=a.policy_prompt, openai_base_url=a.base_url, seed=a.seed, kg_parallel_requests=a.parallel,
                     llm_fallback_models=[m.strip() for m in (a.fallback_models or "").split(",") if m.strip()])
    cfg = Config.load(a.config, **overrides)
    if a.disambiguate:
        cfg.extra["llm_entity_disambiguation"] = True
    # Extension flags OR with the variant, so --variant wkgfc_audit and --audit both work.
    if getattr(a, "verdict_tolerance", False):
        cfg.verdict_tolerance = True
    if getattr(a, "audit_refutations", False):
        cfg.audit_refutations = True
    return cfg


def _run_dir(a, kind: str) -> Path:
    name = a.name or f"{a.dataset}_{kind}_{a.model}_k{a.k}_N{a.hops}" + (f"_n{a.limit}" if a.limit else "")
    d = Path(a.output_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _evaluate(a, make_checker_fn, kind: str, cfg: Config):
    """make_checker_fn() builds a fresh `ex -> result` function. Each worker thread gets its own (own LLM,
    Wikidata and search objects, so per-claim cost counts stay exact); caches are shared."""
    data = load_benchmark(a.dataset, a.data_root, seed=a.seed, limit=a.limit)
    out_dir = _run_dir(a, kind)
    (out_dir / "config.json").write_text(cfg.to_json())
    pred_path = out_dir / "predictions.jsonl"
    done = {}
    if pred_path.exists():  # resume
        for line in pred_path.open():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    todo = [ex for ex in data if ex["id"] not in done]
    workers = max(1, getattr(a, "workers", 1))
    print(f"[{kind}] {a.dataset}: {len(data)} claims ({len(done)} already done, {len(todo)} to do, "
          f"{workers} at a time) -> {out_dir}")

    local = threading.local()

    def run_one(ex):
        if not hasattr(local, "fn"):
            local.fn = make_checker_fn()
        return ex, local.fn(ex)

    n_done = len(done)
    with pred_path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, ex) for ex in todo]
        try:
            for fut in as_completed(futures):
                ex, res = fut.result()
                rec = {"id": ex["id"], "claim": ex["claim"], "gold": ex["label"], "pred": res.verdict,
                       "confidence": res.confidence, "forced": res.forced, "flags": res.flags, "cost": res.cost,
                       "seconds": res.seconds, "error": res.error, "justification": res.justification,
                       "trajectory": res.trajectory, "evidence": res.evidence}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[ex["id"]] = rec
                n_done += 1
                mark = "ok " if res.verdict == ex["label"] else "ERR"
                print(f"  {n_done:4d}/{len(data)} {mark} gold={ex['label']:9s} pred={res.verdict or '-':9s} "
                      f"calls={res.cost.get('llm_calls', 0):3d} {res.seconds:6.1f}s {res.error[:60]}", flush=True)
        except KeyboardInterrupt:
            print("\nStopping: finished claims are saved; rerun the same command to continue.")
            for fu in futures:
                fu.cancel()
            raise
    records = [done[e["id"]] for e in data if e["id"] in done]
    summary = summarise(records)
    summary.update(dataset=a.dataset, kind=kind, model=a.model)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def cmd_evaluate(a):
    cfg = _cfg_from_args(a)

    def make():
        _, checker = build(cfg)
        return lambda ex: checker.check(ex["claim"])

    _evaluate(a, make, a.variant, cfg)


def cmd_direct(a):
    a.variant = "direct"
    cfg = _cfg_from_args(a)

    def make():
        d = DirectLLM(cfg, make_llm(cfg, _cache(cfg, "llm")))
        return lambda ex: d.check(ex["claim"], ex.get("gold_evidence", "") if a.gold_evidence else "")

    _evaluate(a, make, "direct", cfg)


def cmd_check(a):
    cfg = _cfg_from_args(a)
    _, checker = build(cfg)
    res = checker.check(a.claim)
    print(json.dumps({k: v for k, v in res.to_dict().items() if k != "evidence"}, indent=2, ensure_ascii=False))
    print("\nEvidence:\n" + "\n".join(f"  ({t['subject']}, {t['relation']}, {t['obj']}) [{t['source']}]"
                                      for t in res.evidence.get("triples", [])[:80]))


def cmd_optimise(a):
    from .optimise import PolicyOptimiser

    a.variant = "wkgfc"
    cfg = _cfg_from_args(a)
    llm, checker = build(cfg)
    train, val = load_optimisation_pool(a.data_root, seed=a.seed, n=a.pool)
    print(f"[optimise] train={len(train)} val={len(val)}")
    out = Path(a.output_dir) / (a.name or f"policy_{a.model}")
    res = PolicyOptimiser(cfg, checker, llm, str(out), make_checker=lambda: build(cfg)[1],
                          workers=a.workers).run(train, val, epochs=a.epochs, batch_size=a.batch)
    print(json.dumps(res, indent=2))
    print(f"Best policy saved to {out / 'best_policy.json'}; use it with --policy-prompt")


def cmd_datasets(a):
    for name in LOADERS:
        try:
            full = LOADERS[name](Path(a.data_root))
            test = load_benchmark(name, a.data_root, seed=a.seed)
            neg = sum(e["label"] == "REFUTED" for e in test) / len(test)
            print(f"{name:14s} available={len(full):6d}  test={len(test):5d}  refuted={100 * neg:5.1f}%  "
                  f"(paper Table 2: {TABLE2[name]})")
        except FileNotFoundError as e:
            print(f"{name:14s} MISSING: {e}")


def cmd_ping(a):
    """Step-1 connectivity test: one tiny LLM call, no Wikidata / search."""
    import time

    cfg = _cfg_from_args(a)
    llm = make_llm(cfg)  # no cache: always hits the API
    print(f"backend={cfg.llm_backend} model={cfg.llm_model} fallbacks={cfg.llm_fallback_models or 'none'}")
    t0 = time.time()
    try:
        out = llm.complete_json('Is the claim "Paris is the capital of France" SUPPORTED or REFUTED? '
                                'Return JSON only: {"verdict": "...", "confidence": <0-1>}', purpose="ping")
    except Exception as e:
        print(f"LLM connection FAILED: {e}")
        return 1
    print(f"reply: {out}")
    print(f"answered by: {llm.last_model}  tokens in/out: {llm.usage.prompt_tokens}/{llm.usage.completion_tokens}  "
          f"time: {time.time() - t0:.1f}s")
    print("LLM connection OK" if isinstance(out, dict) and out.get("verdict") else "Got a reply but not valid JSON")


def cmd_models(a):
    """List model names available to your key (gemini / openai-compatible backends)."""
    cfg = _cfg_from_args(a)
    if cfg.llm_backend == "bedrock":
        print("For Bedrock, copy the inference-profile ID from the AWS console (Bedrock > Inference profiles).")
        return 0
    llm = make_llm(cfg)
    try:
        names = sorted(m.id for m in llm._get_client().models.list())
    except Exception as e:
        print(f"Could not list models: {e}")
        return 1
    for n in names:
        print(n)
    print(f"\n{len(names)} models. Put one in LLM_MODEL in .env (for Gemini, drop the 'models/' prefix).")
    return 0


def cmd_compare(a):
    def load(d):
        return {json.loads(l)["id"]: json.loads(l) for l in open(Path(d) / "predictions.jsonl")}

    A, B = load(a.run_a), load(a.run_b)
    ids = sorted(set(A) & set(B))
    gold = [A[i]["gold"] for i in ids]
    print(f"{len(ids)} shared claims")
    print(json.dumps(mcnemar(gold, [A[i]["pred"] or "REFUTED" for i in ids],
                             [B[i]["pred"] or "REFUTED" for i in ids]), indent=2))


def cmd_errors(a):
    """Classify the wrong predictions of a finished run by cause, not by agent behaviour."""
    from .error_taxonomy import classify_errors, load_predictions, report_markdown, taxonomy_report

    run = Path(a.run_dir)
    records = load_predictions(run)
    llm = None
    if not a.no_llm:
        cfg = _cfg_from_args(a)
        llm = make_llm(cfg, _cache(cfg, "llm"))
    rows = classify_errors(records, llm)
    report = taxonomy_report(records, rows)
    (run / "error_taxonomy.json").write_text(json.dumps({"report": report, "errors": rows}, indent=2))
    (run / "errors.md").write_text(report_markdown(report, rows))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {run / 'error_taxonomy.json'} and {run / 'errors.md'} ({len(rows)} errors) - audit errors.md by hand.")


def main(argv=None):
    from .config import _load_dotenv

    _load_dotenv(Path(".env"))  # so LLM_BACKEND / LLM_MODEL in .env become the defaults below
    p = argparse.ArgumentParser(prog="wkgfc")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data-root", default="datasets")
        sp.add_argument("--output-dir", default="outputs")
        sp.add_argument("--config", default=None)
        sp.add_argument("--backend", default=os.environ.get("LLM_BACKEND", "openai"),
                        choices=["openai", "bedrock", "gemini"], help="LLM provider (default from LLM_BACKEND in .env)")
        sp.add_argument("--model", default=None,
                        help="model / Bedrock inference-profile ID (default: <BACKEND>_MODEL, then LLM_MODEL in .env)")
        sp.add_argument("--base-url", default=None, help="OpenAI-compatible server (Ollama/vLLM)")
        sp.add_argument("--fallback-models", default=None,
                        help="comma-separated models to try if the main one fails (default: <BACKEND>_FALLBACK_MODELS "
                             "in .env; pass \"\" to turn off - do this for thesis runs)")
        sp.add_argument("--k", type=int, default=4)
        sp.add_argument("--parallel", type=int, default=4,
                        help="concurrent Wikidata/LLM requests per hop (1 = original one-at-a-time; same results)")
        sp.add_argument("--hops", type=int, default=4)
        sp.add_argument("--max-steps", type=int, default=6)
        sp.add_argument("--policy-prompt", default=None)
        sp.add_argument("--disambiguate", action="store_true", help="LLM entity disambiguation (not in paper)")
        sp.add_argument("--tolerance", dest="verdict_tolerance", action="store_true",
                        help="extension: tell the verdict prompt what does NOT count as a refutation")
        sp.add_argument("--audit", dest="audit_refutations", action="store_true",
                        help="extension: re-examine every REFUTED verdict with one adversarial pass")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--name", default=None)

    sp = sub.add_parser("evaluate"); common(sp)
    sp.add_argument("--dataset", required=True, choices=list(LOADERS))
    sp.add_argument("--workers", type=int, default=3,
                    help="claims checked at the same time (1 = one by one; results are the same)")
    sp.add_argument("--variant", default="wkgfc", choices=list(VARIANTS))
    sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(fn=cmd_evaluate)

    sp = sub.add_parser("direct"); common(sp)
    sp.add_argument("--dataset", required=True, choices=list(LOADERS))
    sp.add_argument("--workers", type=int, default=3,
                    help="claims checked at the same time (1 = one by one; results are the same)")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--gold-evidence", action="store_true", help="give gold evidence (SummEval-series)")
    sp.set_defaults(fn=cmd_direct)

    sp = sub.add_parser("check"); common(sp)
    sp.add_argument("claim")
    sp.add_argument("--variant", default="wkgfc", choices=list(VARIANTS))
    sp.set_defaults(fn=cmd_check)

    sp = sub.add_parser("optimise"); common(sp)
    sp.add_argument("--epochs", type=int, default=20)
    sp.add_argument("--batch", type=int, default=10)
    sp.add_argument("--pool", type=int, default=150)
    sp.add_argument("--workers", type=int, default=3, help="claims checked at the same time")
    sp.set_defaults(fn=cmd_optimise)

    sp = sub.add_parser("datasets"); common(sp)
    sp.set_defaults(fn=cmd_datasets)

    sp = sub.add_parser("ping"); common(sp)
    sp.set_defaults(fn=cmd_ping)

    sp = sub.add_parser("models"); common(sp)
    sp.set_defaults(fn=cmd_models)

    sp = sub.add_parser("compare")
    sp.add_argument("run_a"); sp.add_argument("run_b")
    sp.set_defaults(fn=cmd_compare)

    sp = sub.add_parser("errors"); common(sp)
    sp.add_argument("run_dir")
    sp.add_argument("--no-llm", action="store_true",
                    help="deterministic signals only, no LLM cause classification")
    sp.set_defaults(fn=cmd_errors)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
