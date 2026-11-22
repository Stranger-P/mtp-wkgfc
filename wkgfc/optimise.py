"""Policy improvement via self-reflection and prompt optimisation (paper Sec. 4.4).

1. Run the agent on training claims -> trajectories H_n.
2. Composite reward r in [0, 1]: correctness, evidence sufficiency, retrieval efficiency.
3. Self-critique each trajectory (error types: insufficient coverage, premature termination,
   redundant expansion, mishandled contradictions, ...) -> experience buffer D.
4. Textual gradient descent on the policy prompt θ (the TextGrad procedure): an LLM turns a batch of
   critiques into a "textual gradient" (what to change), then applies it to θ.
5. Keep the θ with the best validation balanced accuracy (paper: 100 train / 50 val, 20 epochs).

The base LLM is never fine-tuned; only the prompt changes."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .agent import FactChecker
from .metrics import balanced_accuracy
from .parallel import pmap
import threading

GRADIENT_PROMPT = """You are optimising the SYSTEM PROMPT of a fact-checking agent that decides when to expand a
knowledge graph, search the web, or stop and give a verdict.

Current system prompt:
<<<
{policy}
>>>

Reviews of the agent's recent decisions (reward in [0,1]; low = bad):
{feedback}

Write the textual gradient: concise, concrete criticism of the system prompt explaining which instructions
caused the bad decisions (e.g. stopping too early, searching redundantly, mishandling conflicting
evidence) and what should change. Do not rewrite the prompt yet. Return plain text."""

UPDATE_PROMPT = """Improve the SYSTEM PROMPT of a fact-checking agent using the feedback.

Current system prompt:
<<<
{policy}
>>>

Feedback (textual gradient):
{gradient}

Rules: keep the three actions (expandKG, webSearch, verdict) and their names exactly; keep it general (no
claim-specific facts); keep it under 450 words; change only what the feedback justifies.
Return JSON only: {{"policy": "<the full new system prompt>"}}"""


@dataclass
class Experience:
    claim: str
    gold: str
    pred: str
    reward: float
    actions: list
    error_types: list = field(default_factory=list)
    critique: str = ""


def composite_reward(result, gold: str, max_steps: int) -> float:
    """r = 0.7*correct + 0.2*sufficiency + 0.1*efficiency (weights are ours; the paper names the terms
    but not the weights)."""
    correct = float(result.verdict == gold)
    has_evidence = bool(result.evidence.get("triples"))
    sufficiency = float(has_evidence and not result.forced)
    n_retrieval = sum(1 for s in result.trajectory if s.get("action") in ("expandKG", "webSearch"))
    efficiency = max(0.0, 1.0 - n_retrieval / max(1, max_steps))
    return round(0.7 * correct + 0.2 * sufficiency + 0.1 * efficiency, 3)


def self_critique(llm, result, gold: str) -> tuple[list, str]:
    traj = "\n".join(f"{i}. {json.dumps(s, ensure_ascii=False)[:400]}" for i, s in enumerate(result.trajectory, 1))
    evidence = "\n".join(f"({t['subject']}, {t['relation']}, {t['obj']}) [{t['source']}]"
                         for t in result.evidence.get("triples", [])[:60])
    out = llm.complete_json(prompts.SELF_CRITIQUE.format(
        claim=result.claim, gold=gold, pred=result.verdict or "(none)",
        outcome="correct" if result.verdict == gold else "WRONG", trajectory=traj, evidence=evidence or "(none)"),
        purpose="self_critique", default={})
    return out.get("error_types", []) if isinstance(out, dict) else [], (out.get("critique", "") if isinstance(out, dict) else "")


class PolicyOptimiser:
    def __init__(self, cfg, checker: FactChecker, llm, out_dir: str, make_checker=None, workers: int = 1):
        """make_checker() builds an extra FactChecker per worker thread (own counters, shared caches);
        results are collected in input order, so workers only change wall-clock time."""
        self.cfg, self.checker, self.llm = cfg, checker, llm
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.buffer: list[Experience] = []
        self.workers = max(1, workers) if make_checker else 1
        self._make_checker = make_checker
        self._local = threading.local()

    def _check(self, claim: str, policy: str):
        if self.workers == 1:
            return self.checker.check(claim, policy=policy)
        if not hasattr(self._local, "checker"):
            self._local.checker = self._make_checker()
        return self._local.checker.check(claim, policy=policy)

    def _evaluate(self, policy: str, data: list[dict]) -> tuple[float, list]:
        results = list(zip(data, pmap(lambda ex: self._check(ex["claim"], policy), data, self.workers)))
        preds = [r.verdict or "REFUTED" for _, r in results]
        golds = [ex["label"] for ex, _ in results]
        return balanced_accuracy(golds, preds), results

    def run(self, train: list[dict], val: list[dict], epochs: int = 20, batch_size: int = 10,
            feedback_size: int = 8) -> dict:
        rng = random.Random(self.cfg.seed)
        policy = self.checker.policy
        best_policy, (best_val, _) = policy, self._evaluate(policy, val)
        history = [{"epoch": 0, "val_bacc": best_val}]
        print(f"[optimise] epoch 0 val balanced accuracy = {best_val:.3f}")

        for epoch in range(1, epochs + 1):
            batch = rng.sample(train, min(batch_size, len(train)))
            def one(ex):
                r = self._check(ex["claim"], policy)
                reward = composite_reward(r, ex["label"], self.cfg.max_steps)
                errs, crit = self_critique(self.llm, r, ex["label"])
                return Experience(ex["claim"], ex["label"], r.verdict, reward,
                                  [s.get("action") for s in r.trajectory], errs, crit)

            exps = pmap(one, batch, self.workers)
            self.buffer.extend(exps)
            worst = sorted(exps, key=lambda e: e.reward)[:feedback_size]
            feedback = "\n".join(
                f"- reward={e.reward} gold={e.gold} pred={e.pred} actions={'->'.join(a for a in e.actions if a)} "
                f"errors={e.error_types}: {e.critique}" for e in worst)
            gradient = self.llm.complete(GRADIENT_PROMPT.format(policy=policy, feedback=feedback),
                                         purpose="textgrad_gradient")
            upd = self.llm.complete_json(UPDATE_PROMPT.format(policy=policy, gradient=gradient),
                                         purpose="textgrad_update", default={})
            candidate = upd.get("policy") if isinstance(upd, dict) else None
            if candidate and all(a in candidate for a in ("expandKG", "webSearch", "verdict")):
                policy = candidate
            val_bacc, _ = self._evaluate(policy, val)
            history.append({"epoch": epoch, "val_bacc": val_bacc,
                            "mean_train_reward": sum(e.reward for e in exps) / len(exps)})
            print(f"[optimise] epoch {epoch} val balanced accuracy = {val_bacc:.3f}")
            if val_bacc > best_val:
                best_val, best_policy = val_bacc, policy
                (self.out / "best_policy.json").write_text(json.dumps(
                    {"policy": best_policy, "val_bacc": best_val, "epoch": epoch}, indent=2))
            (self.out / "history.json").write_text(json.dumps(history, indent=2))
            with open(self.out / "experience_buffer.jsonl", "a") as f:
                for e in exps:
                    f.write(json.dumps(e.__dict__, ensure_ascii=False) + "\n")

        if not (self.out / "best_policy.json").exists():
            (self.out / "best_policy.json").write_text(json.dumps(
                {"policy": best_policy, "val_bacc": best_val, "epoch": 0}, indent=2))
        return {"best_val_bacc": best_val, "history": history}
