"""Agentic module (paper Sec. 4.1): evidence acquisition as a POMDP.

a0 = initKGRetrieval(claim) -> o0 = initial sub-KG. Then, at each step the policy π_θ (the LLM with
policy prompt θ) observes (claim, evidence graph, history) and picks expandKG / webSearch / verdict.
If max_steps is reached, a special prompt forces a verdict (the paper's "Exceeding Maximum Steps").

Fixed-pipeline variants (no agent) reproduce ablation Variants 1, 2 and 4 of Table 3."""
from __future__ import annotations

import re

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .kg import EvidenceGraph
from .kg_retrieval import KGRetriever
from .web_retrieval import WebRetriever

ACTIONS = ("expandKG", "webSearch", "verdict")


@dataclass
class Step:
    action: str
    detail: dict = field(default_factory=dict)


@dataclass
class CheckResult:
    claim: str
    verdict: str
    confidence: float
    justification: str
    forced: bool = False
    trajectory: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_NEGATION = re.compile(
    r"\b(not|n't|never|no longer|without|fails? to|neither|nor)\b", re.I)


def normalise_label(v) -> str:
    s = str(v or "").strip().upper()
    if s.startswith("SUPPORT") or s in ("TRUE", "YES"):
        return "SUPPORTED"
    if s.startswith("REFUT") or s.startswith("NOT") or s in ("FALSE", "NO"):
        return "REFUTED"
    return ""


class FactChecker:
    def __init__(self, cfg, llm, kg_client=None, search=None):
        self.cfg = cfg
        self.llm = llm
        self.kg_client = kg_client
        self.kg = KGRetriever(cfg, kg_client, llm) if kg_client is not None else None
        self.web = WebRetriever(cfg, search, llm) if search is not None else None
        self.policy = self._load_policy()

    def _load_policy(self) -> str:
        p = self.cfg.policy_prompt_path
        if p and Path(p).exists():
            text = Path(p).read_text()
            try:  # allow the JSON produced by optimise.py
                text = json.loads(text)["policy"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            return text
        return prompts.AGENT_POLICY

    # ------------------------------------------------------------------
    def check(self, claim: str, policy: str | None = None) -> CheckResult:
        t0 = time.time()
        usage0 = self.llm.usage.snapshot()
        q0 = getattr(self.kg_client, "queries", 0)
        s0 = getattr(getattr(self.web, "search", None), "calls", 0)
        graph = EvidenceGraph()
        try:
            if self.cfg.use_agent:
                res = self._run_agent(claim, graph, policy or self.policy)
            else:
                res = self._run_pipeline(claim, graph)
        except Exception as e:  # execution errors are a category in the paper's error analysis
            res = CheckResult(claim, "", 0.0, "", error=f"{type(e).__name__}: {e}")
        res.evidence = graph.to_dict()
        u1 = self.llm.usage.snapshot()
        res.cost = {
            "llm_calls": u1["calls"] - usage0["calls"],
            "llm_calls_uncached": (u1["calls"] - u1["cached_calls"]) - (usage0["calls"] - usage0["cached_calls"]),
            "prompt_tokens": u1["prompt_tokens"] - usage0["prompt_tokens"],
            "completion_tokens": u1["completion_tokens"] - usage0["completion_tokens"],
            "models": {m: n - usage0.get("by_model", {}).get(m, 0) for m, n in u1.get("by_model", {}).items()
                       if n - usage0.get("by_model", {}).get(m, 0) > 0},
            "sparql_queries": getattr(self.kg_client, "queries", 0) - q0,
            "web_searches": getattr(getattr(self.web, "search", None), "calls", 0) - s0,
        }
        res.seconds = round(time.time() - t0, 2)
        return res

    # ------------------------------------------------------------------
    def _run_agent(self, claim: str, graph: EvidenceGraph, policy: str) -> CheckResult:
        traj: list[dict] = []
        if self.cfg.use_kg and self.kg:
            self.kg.init_retrieval(claim, graph)
            traj.append({"action": "initKGRetrieval", "entities": graph.topic_entities, **graph.stats()})

        retrieval_actions = 0
        expanded_once = False
        flags = {"requested_web": False, "insufficient_kg": False}

        for step in range(1, self.cfg.max_steps + 1):
            decision = self._decide(claim, graph, traj, step, policy)
            action = decision.get("action")
            if action not in ACTIONS:
                action = "verdict" if normalise_label(decision.get("verdict")) else self._fallback_action(graph)

            # Redirect impossible actions (no hops/frontier left, or source disabled)
            if action == "expandKG" and not self._can_expand(graph):
                action = "webSearch" if self._can_web() else "verdict"
            if action == "webSearch" and not self._can_web():
                action = "expandKG" if self._can_expand(graph) else "verdict"

            if action == "verdict":
                label = normalise_label(decision.get("verdict"))
                if not label:  # policy wanted to stop but gave no label -> forced verdict
                    break
                traj.append({"action": "verdict", "thought": decision.get("thought", "")})
                label, conf, just, audit = self._audit_refutation(
                    claim, graph, label, _f(decision.get("confidence")), decision.get("justification", ""))
                if audit:
                    traj.append({"action": "refutationAudit", **audit})
                return CheckResult(
                    claim, label, conf, just,
                    trajectory=traj,
                    flags={**flags, "over_confidence_pattern": retrieval_actions == 0,
                           "steps": step, "retrieval_actions": retrieval_actions,
                           "audit": audit})

            retrieval_actions += 1
            if action == "expandKG":
                new = self.kg.expand_hop(claim, graph)
                expanded_once = True
                traj.append({"action": "expandKG", "new_triples": new, "thought": decision.get("thought", ""),
                             **graph.stats()})
            else:
                query = (decision.get("query") or "").strip() or self.web.make_query(claim)
                flags["requested_web"] = True
                if expanded_once or (self.kg and self.kg.hops_left(graph) == 0):
                    flags["insufficient_kg"] = True
                report = self.web.retrieve(claim, query, graph)
                traj.append({"action": "webSearch", "thought": decision.get("thought", ""), **report})

        # Exceeded maximum steps -> enforced verdict
        out = self.llm.complete_json(prompts.FORCED_VERDICT.format(claim=claim, evidence=graph.render()) + self._tolerance(),
                                     purpose="forced_verdict", default={})
        traj.append({"action": "forcedVerdict"})
        label, conf, just, audit = self._audit_refutation(
            claim, graph, normalise_label(out.get("verdict")) or "REFUTED", _f(out.get("confidence")),
            out.get("justification", ""))
        if audit:
            traj.append({"action": "refutationAudit", **audit})
        return CheckResult(
            claim, label, conf, just, forced=True, trajectory=traj,
            flags={**flags, "over_confidence_pattern": False, "steps": self.cfg.max_steps,
                   "retrieval_actions": retrieval_actions, "audit": audit})

    def _tolerance(self) -> str:
        """Appended to a verdict prompt: what does NOT count as a refutation (extension, off by default)."""
        return prompts.TOLERANCE_NOTE if self.cfg.verdict_tolerance else ""

    def _audit_refutation(self, claim: str, graph: EvidenceGraph, label: str, conf: float, just: str):
        """One adversarial pass over a REFUTED verdict (extension, off by default).

        Returns (label, confidence, justification, audit). Measured on FEVER-200, about half of the
        false refutations rest on a wording or precision mismatch rather than a contradiction, and
        confidence cannot tell those apart - so the mismatch is re-examined on its content.
        """
        if not self.cfg.audit_refutations or label != "REFUTED":
            return label, conf, just, None
        out = self.llm.complete_json(
            prompts.REFUTATION_AUDIT.format(claim=claim, justification=just or "(none given)",
                                            evidence=graph.render()),
            purpose="refutation_audit", default={}) or {}
        new = normalise_label(out.get("verdict"))
        mismatch = out.get("mismatch", "")
        # The audit's own verdict is not trusted: on FEVER-200 it labelled 48 of 112 refusals
        # non-substantive yet overturned only 2, and most non-substantive labels are net-negative.
        # So overturning is gated on the mismatch label, by a rule measured offline.
        allowed = mismatch in self.cfg.audit_overturn_on
        negated = bool(_NEGATION.search(claim)) and self.cfg.audit_skip_negated
        overturn = bool(new) and new != label and allowed and not negated
        audit = {"mismatch": mismatch, "case_for_claim": out.get("case_for_claim", ""),
                 "model_verdict": new, "allowed_by_rule": allowed, "negated_claim": negated,
                 "overturned": overturn}
        if not overturn:
            return label, conf, just, audit
        return new, _f(out.get("confidence")) or conf, out.get("justification", just), audit

    def _decide(self, claim, graph, traj, step, policy) -> dict:
        history = " -> ".join(t["action"] + (f"('{t.get('query')}')" if t.get("query") else "") for t in traj) or "none"
        hops_left = self.kg.hops_left(graph) if self.kg and self.cfg.use_kg else 0
        notes = []
        if not self.cfg.use_web:
            notes.append(" Web search is disabled.")
        if not self._can_expand(graph):
            notes.append(" KG expansion is not possible (no frontier or hop budget left).")
        prompt = prompts.AGENT_STEP.format(claim=claim, evidence=graph.render(), history=history, step=step,
                                           max_steps=self.cfg.max_steps, hops_left=hops_left,
                                           web_note="".join(notes))
        out = self.llm.complete_json(prompt, system=policy, purpose="agent_step", default={})
        return out if isinstance(out, dict) else {}

    def _can_expand(self, graph) -> bool:
        return bool(self.cfg.use_kg and self.kg and graph.frontier and self.kg.hops_left(graph) > 0)

    def _can_web(self) -> bool:
        return bool(self.cfg.use_web and self.web)

    def _fallback_action(self, graph) -> str:
        return "expandKG" if self._can_expand(graph) else ("webSearch" if self._can_web() else "verdict")

    # ------------------------------------------------------------------
    def _run_pipeline(self, claim: str, graph: EvidenceGraph) -> CheckResult:
        """Variants 1 (kg_only), 2 (web_only), 4 (kg_web_no_agent): fixed retrieval, then one verdict."""
        traj = []
        if self.cfg.use_kg and self.kg:
            self.kg.init_retrieval(claim, graph)
            traj.append({"action": "initKGRetrieval", "entities": graph.topic_entities, **graph.stats()})
            while self._can_expand(graph):
                new = self.kg.expand_hop(claim, graph)
                traj.append({"action": "expandKG", "new_triples": new, **graph.stats()})
        if self.cfg.use_web and self.web:
            q = self.web.make_query(claim)
            traj.append({"action": "webSearch", **self.web.retrieve(claim, q, graph)})
        out = self.llm.complete_json(
            prompts.PIPELINE_VERDICT.format(claim=claim, evidence=graph.render()) + self._tolerance(),
            purpose="verdict", default={})
        label, conf, just, audit = self._audit_refutation(
            claim, graph, normalise_label(out.get("verdict")) or "REFUTED", _f(out.get("confidence")),
            out.get("justification", ""))
        if audit:
            traj.append({"action": "refutationAudit", **audit})
        return CheckResult(claim, label, conf, just, trajectory=traj, flags={"audit": audit})


class DirectLLM:
    """Closed-book baseline: the LLM judges the claim with no retrieval (paper Table 1, LLM rows)."""

    def __init__(self, cfg, llm):
        self.cfg, self.llm = cfg, llm

    def check(self, claim: str, gold_evidence: str = "") -> CheckResult:
        t0 = time.time()
        u0 = self.llm.usage.snapshot()
        gold = f"\nEvidence:\n{gold_evidence}\n" if gold_evidence else ""
        out = self.llm.complete_json(prompts.DIRECT_VERDICT.format(claim=claim, gold=gold),
                                     purpose="direct", default={})
        r = CheckResult(claim, normalise_label(out.get("verdict")) or "REFUTED", _f(out.get("confidence")),
                        out.get("justification", ""))
        u1 = self.llm.usage.snapshot()
        r.cost = {k: u1[k] - u0[k] for k in ("calls", "prompt_tokens", "completion_tokens")}
        r.cost["llm_calls"] = r.cost.pop("calls")
        r.cost["llm_calls_uncached"] = (u1["calls"] - u1["cached_calls"]) - (u0["calls"] - u0["cached_calls"])
        r.cost["models"] = {m: n - u0["by_model"].get(m, 0) for m, n in u1["by_model"].items()
                            if n - u0["by_model"].get(m, 0) > 0}
        r.seconds = round(time.time() - t0, 2)
        return r


def _f(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0
