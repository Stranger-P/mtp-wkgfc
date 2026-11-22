"""Offline end-to-end tests: fake Wikidata graph, fake search, scripted LLM.
Run:  python -m pytest -q tests   (or  python tests/test_offline.py)"""
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wkgfc.agent import FactChecker, normalise_label  # noqa: E402
from wkgfc.cache import Cache  # noqa: E402
from wkgfc.config import VARIANTS, Config  # noqa: E402
from wkgfc.kg import EvidenceGraph, Triple  # noqa: E402
from wkgfc.llm import MockLLM, parse_json  # noqa: E402
from wkgfc.metrics import balanced_accuracy, error_analysis, macro_f1, mcnemar, summarise  # noqa: E402
from wkgfc.optimise import PolicyOptimiser, composite_reward  # noqa: E402
from wkgfc.web_retrieval import MockSearch, bm25_rank  # noqa: E402
from wkgfc.wikidata import _clean_literal, useless_relation  # noqa: E402


# --------------------------------------------------------------------------
# Fake Wikidata
# --------------------------------------------------------------------------
class FakeKG:
    ENT = {"Q42": "Douglas Adams", "Q691283": "St John's College", "Q350": "Cambridge",
           "Q36180": "writer", "Q145": "United Kingdom", "Q35794": "University of Cambridge"}
    EDGES = [  # (s, pid, label, o)
        ("Q42", "P69", "educated at", "Q691283"),
        ("Q42", "P19", "place of birth", "Q350"),
        ("Q42", "P106", "occupation", "Q36180"),
        ("Q42", "P27", "country of citizenship", "Q145"),
        ("Q42", "P213", "ISNI", None),
        ("Q691283", "P361", "part of", "Q35794"),
        ("Q691283", "P131", "located in the administrative territorial entity", "Q350"),
        ("Q35794", "P131", "located in the administrative territorial entity", "Q350"),
    ]
    LIT = {("Q42", "P569"): ("date of birth", "1952-03-11")}

    def __init__(self):
        self.queries = 0

    def search_entities(self, name, limit=5):
        return [{"id": q, "label": l, "description": ""} for q, l in self.ENT.items()
                if name.lower() in l.lower()][:limit]

    def relations(self, qid):
        self.queries += 2
        rels = []
        for s, p, lab, o in self.EDGES:
            if s == qid:
                rels.append({"pid": p, "label": lab, "direction": "out"})
            if o == qid:
                rels.append({"pid": p, "label": lab, "direction": "in"})
        for (s, p), (lab, _) in self.LIT.items():
            if s == qid:
                rels.append({"pid": p, "label": lab, "direction": "out"})
        return [r for r in rels if not useless_relation(r["label"])]

    def values(self, qid, pid, direction, limit=10):
        self.queries += 1
        if (qid, pid) in self.LIT and direction == "out":
            return [{"id": "", "label": self.LIT[(qid, pid)][1]}]
        out = []
        for s, p, lab, o in self.EDGES:
            if p != pid or o is None:
                continue
            if direction == "out" and s == qid:
                out.append({"id": o, "label": self.ENT[o]})
            if direction == "in" and o == qid:
                out.append({"id": s, "label": self.ENT[s]})
        return out[:limit]

    def label(self, qid):
        return self.ENT.get(qid, qid)


# --------------------------------------------------------------------------
# Scripted LLM: reads the prompt and answers like a sensible model would
# --------------------------------------------------------------------------
def make_responder(agent_mode="smart"):
    def responder(prompt, purpose):
        claim = re.search(r"Claim: (.*)", prompt).group(1) if "Claim:" in prompt else ""
        cl = claim.lower()
        if purpose == "relation_prune":
            lines = re.findall(r"^(\d+)\. (.*)$", prompt.split("Relations:")[1], re.M)
            scored = []
            for i, lab in lines:
                words = [w for w in re.findall(r"[a-z]+", lab.lower()) if len(w) > 3]
                hit = sum(w in cl for w in words)
                scored.append((hit, int(i)))
            scored.sort(reverse=True)
            return json.dumps({"relations": [{"id": i, "score": 0.25} for _, i in scored[:4]]})
        if purpose == "entity_score":
            n = len(re.findall(r"^\d+\. ", prompt.split("Candidates:")[1], re.M))
            return json.dumps({"scores": [1 / n] * n})
        if purpose == "agent_step":
            ev = prompt.split("Evidence so far:")[1]
            if agent_mode == "always_search":
                return json.dumps({"action": "webSearch" if "webSearch" in prompt.split("Previous actions:")[1].split("\n")[0] else "expandKG",
                                   "query": "douglas adams", "verdict": ""})
            if agent_mode == "overconfident":
                return json.dumps({"action": "verdict", "verdict": "SUPPORTED", "confidence": 0.9})
            if "educated at, St John's College" in ev and "st john" in cl:
                return json.dumps({"thought": "found", "sufficient": True, "action": "verdict",
                                   "verdict": "SUPPORTED", "confidence": 0.95, "justification": "educated at"})
            if "educated at" in ev and "oxford" in cl:
                return json.dumps({"action": "verdict", "verdict": "REFUTED", "confidence": 0.9})
            if "towel day" in cl and "Towel Day" not in ev:
                return json.dumps({"action": "webSearch", "query": "Towel Day Douglas Adams date", "verdict": ""})
            if "towel day" in cl:
                return json.dumps({"action": "verdict", "verdict": "SUPPORTED", "confidence": 0.8})
            return json.dumps({"action": "expandKG", "verdict": ""})
        if purpose in ("forced_verdict", "verdict", "direct"):
            return json.dumps({"verdict": "REFUTED" if "oxford" in cl else "SUPPORTED", "confidence": 0.6})
        if purpose == "web_query":
            return json.dumps({"query": claim[:60]})
        if purpose == "web_filter":
            n = len(re.findall(r"^\[\d+\]", prompt, re.M))
            return json.dumps({"passages": [{"id": i, "relevant": True, "confidence": 0.8,
                                             "key_sentences": "Towel Day is celebrated on 25 May in honour of Douglas Adams."}
                                            for i in range(1, n + 1)]})
        if purpose == "web_to_kg":
            return json.dumps({"triples": [{"subject": "Towel Day", "relation": "commemorates",
                                            "object": "Douglas Adams", "source": 1},
                                           {"subject": "Towel Day", "relation": "point in time",
                                            "object": "25 May", "source": 1}],
                               "annotations": [{"entity": "Douglas Adams", "text": "Fans carry towels on 25 May.",
                                                "source": 1}]})
        if purpose == "self_critique":
            return json.dumps({"error_types": ["premature_termination"], "critique": "Check all parts first."})
        if purpose == "textgrad_gradient":
            return "The prompt should insist on checking every part of the claim before a verdict."
        if purpose == "textgrad_update":
            pol = prompt.split("<<<")[1].split(">>>")[0].strip()
            return json.dumps({"policy": pol + "\n- Always verify each part of the claim before deciding."})
        return "{}"
    return responder


PAGES = {"towel": [
    {"title": "Towel Day - Wikipedia", "link": "https://en.wikipedia.org/wiki/Towel_Day",
     "snippet": "Towel Day is celebrated every year on 25 May as a tribute to the author Douglas Adams."},
    {"title": "Towel Day fact check", "link": "https://www.snopes.com/towel", "snippet": "blocked site"},
    {"title": "Recipes", "link": "https://food.example.com", "snippet": "How to bake bread at home."},
]}


def make_checker(variant="wkgfc", agent_mode="smart", **kw):
    cfg = Config(llm_backend="mock", search_backend="mock", **{**VARIANTS[variant], **kw})
    llm = MockLLM(cfg, make_responder(agent_mode))
    return cfg, llm, FactChecker(cfg, llm, FakeKG(), MockSearch(PAGES))


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_parse_json_and_labels():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"verdict": "REFUTED"} hope this helps') == {"verdict": "REFUTED"}
    assert parse_json("no json") is None
    assert normalise_label("Supports") == "SUPPORTED" and normalise_label("NOT_SUPPORTED") == "REFUTED"
    assert normalise_label("maybe") == ""


def test_literals_and_relation_filter():
    assert _clean_literal("1952-03-11T00:00:00Z") == "1952-03-11"
    assert _clean_literal("1952-01-01T00:00:00Z") == "1952"
    assert useless_relation("ISNI") is False and useless_relation("VIAF ID") and useless_relation("Commons category")


def test_evidence_graph_dedup():
    g = EvidenceGraph()
    assert g.add(Triple("Douglas Adams", "educated at", "St John's College"))
    assert not g.add(Triple("douglas  adams", "Educated_at", "st john's college", source="web"))
    g.annotate("Douglas Adams", "note")
    assert "Knowledge-graph triples" in g.render() and g.stats()["kg_triples"] == 1


def test_bm25_prefers_relevant():
    s = bm25_rank("Towel Day Douglas Adams", ["Towel Day honours Douglas Adams", "bread recipes"])
    assert s[0] > s[1]


def test_agent_supported_from_kg():
    cfg, llm, fc = make_checker()
    r = fc.check("Douglas Adams was educated at St John's College.")
    assert r.verdict == "SUPPORTED" and not r.forced and not r.error, r.error
    assert r.trajectory[0]["action"] == "initKGRetrieval"
    assert any(t["relation"] == "educated at" for t in r.evidence["triples"])
    assert r.cost["llm_calls"] > 0 and r.cost["sparql_queries"] > 0


def test_agent_refuted():
    cfg, llm, fc = make_checker()
    r = fc.check("Douglas Adams was educated at Oxford.")
    assert r.verdict == "REFUTED", r


def test_agent_web_search_path():
    cfg, llm, fc = make_checker()
    r = fc.check("Towel Day honours Douglas Adams.")
    actions = [t["action"] for t in r.trajectory]
    assert "webSearch" in actions and r.verdict == "SUPPORTED"
    web = [t for t in r.evidence["triples"] if t["source"] == "web"]
    assert web and all("snopes" not in t["url"] for t in web)  # fact-check sites blocked
    assert r.flags["requested_web"]


def test_forced_verdict_when_steps_exhausted():
    cfg, llm, fc = make_checker(agent_mode="always_search", max_steps=3)
    r = fc.check("Douglas Adams wrote about towels.")
    assert r.forced and r.trajectory[-1]["action"] == "forcedVerdict" and r.verdict in ("SUPPORTED", "REFUTED")


def test_overconfidence_flag():
    cfg, llm, fc = make_checker(agent_mode="overconfident")
    r = fc.check("Douglas Adams was educated at Oxford.")
    assert r.flags["over_confidence_pattern"] and r.verdict == "SUPPORTED"


def test_pipeline_variants():
    for v in ("kg_only", "web_only", "kg_web_no_agent", "kg_agent"):
        cfg, llm, fc = make_checker(variant=v)
        r = fc.check("Towel Day honours Douglas Adams.")
        assert r.verdict in ("SUPPORTED", "REFUTED") and not r.error, (v, r.error)
        acts = {t["action"] for t in r.trajectory}
        if v == "kg_only":
            assert "webSearch" not in acts and r.evidence["stats"]["hops"] >= 1
        if v == "web_only":
            assert "initKGRetrieval" not in acts and "webSearch" in acts


def test_hop_budget_respected():
    cfg, llm, fc = make_checker(variant="kg_only", max_hops_N=2, init_hops=1)
    r = fc.check("Douglas Adams was born in Cambridge.")
    assert r.evidence["stats"]["hops"] <= 2


def test_metrics():
    gold = ["SUPPORTED"] * 4 + ["REFUTED"] * 4
    pred = ["SUPPORTED"] * 3 + ["REFUTED"] * 5
    assert abs(balanced_accuracy(gold, pred) - 0.875) < 1e-9
    assert 0 < macro_f1(gold, pred) <= 1
    recs = [{"gold": "SUPPORTED", "pred": "REFUTED", "forced": True, "flags": {"insufficient_kg": True}},
            {"gold": "REFUTED", "pred": "SUPPORTED", "forced": False, "flags": {"over_confidence_pattern": True}},
            {"gold": "REFUTED", "pred": "REFUTED", "forced": False, "flags": {}}]
    e = error_analysis(recs)
    assert e["wrong"] == 2 and e["IK_%"] == 50.0 and e["EM_%"] == 50.0 and e["OC_%"] == 50.0
    assert summarise(recs)["n"] == 3
    assert mcnemar(gold, pred, gold)["c"] == 1


def test_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        c = Cache(d, "t")
        k = Cache.make_key("x", 1)
        assert c.get_or_compute(k, lambda: {"v": 1}) == {"v": 1}
        assert c.get_or_compute(k, lambda: {"v": 2}) == {"v": 1} and c.hits == 1


def test_optimiser_loop():
    cfg, llm, fc = make_checker()
    data = [{"claim": "Douglas Adams was educated at St John's College.", "label": "SUPPORTED"},
            {"claim": "Douglas Adams was educated at Oxford.", "label": "REFUTED"},
            {"claim": "Towel Day honours Douglas Adams.", "label": "SUPPORTED"}]
    r = fc.check(data[0]["claim"])
    assert 0.7 <= composite_reward(r, "SUPPORTED", cfg.max_steps) <= 1.0
    with tempfile.TemporaryDirectory() as d:
        out = PolicyOptimiser(cfg, fc, llm, d).run(data, data, epochs=2, batch_size=2)
        best = json.load(open(os.path.join(d, "best_policy.json")))
        assert "expandKG" in best["policy"] and len(out["history"]) == 3
        assert os.path.exists(os.path.join(d, "experience_buffer.jsonl"))


def test_bedrock_backend_with_fake_client():
    from wkgfc.llm import make_llm

    class FakeBedrock:
        def __init__(self):
            self.calls = []

        def converse(self, **kw):
            self.calls.append(kw)
            return {"output": {"message": {"content": [{"text": 'Here: {"verdict": "SUPPORTED"}'}]}},
                    "usage": {"inputTokens": 12, "outputTokens": 5}}

    cfg = Config(llm_backend="bedrock", llm_model="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    llm = make_llm(cfg)
    fake = FakeBedrock()
    llm._client = fake
    out = llm.complete_json("Is it true?", purpose="ping")
    assert out == {"verdict": "SUPPORTED"}
    kw = fake.calls[0]
    assert kw["modelId"].startswith("us.anthropic") and kw["inferenceConfig"]["temperature"] == 0.0
    assert kw["system"][0]["text"] and "JSON" in kw["messages"][0]["content"][0]["text"]
    assert llm.usage.prompt_tokens == 12 and llm.usage.completion_tokens == 5


def test_gemini_backend_routing_and_missing_key():
    from wkgfc.llm import GEMINI_BASE_URL, GeminiLLM, make_llm
    cfg = Config(llm_backend="gemini", llm_model="gemini-2.5-flash")
    llm = make_llm(cfg)
    assert isinstance(llm, GeminiLLM) and "generativelanguage.googleapis.com" in GEMINI_BASE_URL
    saved = os.environ.pop("GEMINI_API_KEY", None)
    try:
        try:
            llm._get_client()
            assert False, "expected missing-key error"
        except RuntimeError as e:
            assert "GEMINI_API_KEY" in str(e)
    finally:
        if saved is not None:
            os.environ["GEMINI_API_KEY"] = saved


def test_openai_compatible_auth_error_not_retried():
    from wkgfc.llm import GeminiLLM

    class AuthErr(Exception):
        status_code = 401

    class Fake:
        def __init__(self):
            self.n = 0
            self.chat = self
            self.completions = self

        def create(self, **kw):
            self.n += 1
            raise AuthErr("invalid key")

    llm = GeminiLLM(Config(llm_backend="gemini", llm_model="m"))
    llm._client = Fake()
    try:
        llm.complete("hi")
        assert False
    except RuntimeError as e:
        assert "not retrying" in str(e)
    assert llm._client.n == 1


def test_fallback_to_next_model_on_busy():
    from wkgfc.llm import GeminiLLM

    class Busy(Exception):
        status_code = 503

    class Fake:
        def __init__(self):
            self.models = []
            self.chat = self
            self.completions = self

        def create(self, **kw):
            self.models.append(kw["model"])
            if kw["model"] == "main":
                raise Busy("overloaded")
            class U: prompt_tokens = 3; completion_tokens = 2
            class M: content = "OK"
            class C: message = M()
            class R: choices = [C()]; usage = U()
            return R()

    import wkgfc.llm as L
    sleep, L.time.sleep = L.time.sleep, lambda s: None
    try:
        llm = GeminiLLM(Config(llm_backend="gemini", llm_model="main", llm_fallback_models=["backup"]))
        llm._client = Fake()
        assert llm.complete("hi") == "OK"
    finally:
        L.time.sleep = sleep
    assert llm.last_model == "backup" and llm.usage.by_model == {"backup": 1}
    assert llm._client.models == ["main", "main", "backup"]


def test_parallel_kg_retrieval_matches_sequential():
    def run(workers, claim):
        cfg, llm, checker = make_checker(kg_parallel_requests=workers)
        res = checker.check(claim)
        return res.verdict, [(t["subject"], t["relation"], t["obj"]) for t in res.evidence.get("triples", [])]
    for claim in ("Douglas Adams was educated at St John's College, Cambridge.",
                  "Douglas Adams was born in London."):
        assert run(1, claim) == run(4, claim)


def test_optimiser_parallel_matches_sequential():
    data = [{"claim": "Douglas Adams was educated at St John's College.", "label": "SUPPORTED"},
            {"claim": "Douglas Adams was educated at Oxford.", "label": "REFUTED"},
            {"claim": "Towel Day honours Douglas Adams.", "label": "SUPPORTED"}]
    hist = []
    for workers in (1, 3):
        cfg, llm, fc = make_checker()
        with tempfile.TemporaryDirectory() as d:
            opt = PolicyOptimiser(cfg, fc, llm, d, make_checker=lambda: make_checker()[2], workers=workers)
            hist.append(opt.run(data, data, epochs=2, batch_size=3)["history"])
    assert hist[0] == hist[1]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(fns)} tests passed")


# --- per-class metrics and the error taxonomy -------------------------------------------------

def test_per_class_exposes_refutation_asymmetry():
    from wkgfc.metrics import per_class, summarise
    # 10 claims: 5 refuted (all caught), 5 supported (3 wrongly refuted) -> REFUTED cries wolf.
    gold = ["REFUTED"] * 5 + ["SUPPORTED"] * 5
    pred = ["REFUTED"] * 5 + ["REFUTED"] * 3 + ["SUPPORTED"] * 2
    pc = per_class(gold, pred)
    assert pc["REFUTED"]["recall"] == 100.0
    assert pc["REFUTED"]["precision"] == 62.5          # 5 of 8 predicted refutations are right
    assert pc["SUPPORTED"]["recall"] == 40.0
    assert pc["REFUTED"]["support"] == 5
    recs = [{"gold": g, "pred": p, "flags": {}, "cost": {}} for g, p in zip(gold, pred)]
    assert summarise(recs)["per_class"]["REFUTED"]["precision"] == 62.5


def _err_rec(rid, gold, pred, **kw):
    rec = {"id": rid, "claim": f"claim {rid}", "gold": gold, "pred": pred, "confidence": 0.9,
           "forced": False, "flags": {"steps": 2, "retrieval_actions": 1, "requested_web": True,
                                      "insufficient_kg": False, "over_confidence_pattern": False},
           "cost": {"llm_calls": 10}, "justification": "because", "evidence": {"triples": [], "annotations": {}}}
    rec.update(kw)
    return rec


def test_taxonomy_counts_direction_without_an_llm():
    from wkgfc.error_taxonomy import classify_errors, taxonomy_report
    recs = [_err_rec("a", "SUPPORTED", "REFUTED"), _err_rec("b", "SUPPORTED", "REFUTED"),
            _err_rec("c", "REFUTED", "SUPPORTED"), _err_rec("d", "REFUTED", "REFUTED")]
    rows = classify_errors(recs, llm=None)
    assert len(rows) == 3                                     # the correct one is skipped
    rep = taxonomy_report(recs, rows)
    assert rep["direction"] == {"false_refutation": 2, "missed_refutation": 1}
    assert rep["by_category"] == {"unclassified": 3}           # no llm -> no cause assigned


def test_taxonomy_discounts_label_noise_from_accuracy():
    from wkgfc.error_taxonomy import classify_errors, taxonomy_report
    recs = [_err_rec(str(i), "SUPPORTED", "SUPPORTED") for i in range(8)]
    recs += [_err_rec("x", "SUPPORTED", "REFUTED"), _err_rec("y", "SUPPORTED", "REFUTED")]

    class FakeLLM:
        def __init__(self): self.n = 0
        def complete_json(self, prompt, **kw):
            assert "strongest case FOR the gold label" in prompt   # de-biasing step is in the prompt
            assert "confidence 0.9" not in prompt                  # system confidence must be hidden
            self.n += 1
            return {"category": "label_noise" if self.n == 1 else "strict_semantic",
                    "system_actually_right": self.n == 1, "reason": "r", "case_for_gold": "c"}

    rows = classify_errors(recs, FakeLLM())
    rep = taxonomy_report(recs, rows)
    assert rep["wrong"] == 2 and rep["dataset_fault"] == 1
    assert rep["strict_match_refutations"] == 1
    assert rep["accuracy_excluding_label_noise"] == 90.0   # 8 right + 1 bad gold, of 10


def test_label_noise_is_judged_last():
    from wkgfc.error_taxonomy import CATEGORIES
    assert list(CATEGORIES)[-1] == "label_noise"            # ordering is load-bearing for the prompt


# --- refutation audit (extension) -------------------------------------------------------------

def _checker(**flags):
    from wkgfc.agent import FactChecker
    from wkgfc.config import Config
    from wkgfc.llm import MockLLM
    cfg = Config(use_kg=False, use_web=False, use_agent=False, **flags)
    seen = []

    def responder(prompt, system):
        seen.append(prompt)
        if "check\nwhether the mismatch" in prompt or "REFUTATION" in prompt or "classify the mismatch" in prompt:
            return '{"case_for_claim": "c", "mismatch": "specificity", "verdict": "SUPPORTED",' \
                   ' "confidence": 0.8, "justification": "a gluten-free diet is a healthy diet"}'
        return '{"verdict": "REFUTED", "confidence": 0.95, "justification": "evidence says gluten-free"}'

    return FactChecker(cfg, MockLLM(cfg, responder)), seen


def test_audit_overturns_only_mismatch_types_the_rule_allows():
    # specificity is diagnosed but NOT allowed to overturn: on FEVER-200 it fixed 1 and broke 7.
    checker, _ = _checker(audit_refutations=True)
    r = checker.check("Coeliac disease is treated by maintaining a healthy diet.")
    assert r.verdict == "REFUTED"
    assert r.flags["audit"]["mismatch"] == "specificity"
    assert r.flags["audit"]["model_verdict"] == "SUPPORTED"      # the model wanted to flip
    assert r.flags["audit"]["allowed_by_rule"] is False          # the rule stopped it
    assert r.flags["audit"]["overturned"] is False
    assert any(t["action"] == "refutationAudit" for t in r.trajectory)

    # widen the rule and the same audit now overturns
    checker, _ = _checker(audit_refutations=True, audit_overturn_on=("specificity",))
    r = checker.check("Coeliac disease is treated by maintaining a healthy diet.")
    assert r.verdict == "SUPPORTED" and r.flags["audit"]["overturned"] is True


def test_audit_never_overturns_a_negated_claim():
    checker, _ = _checker(audit_refutations=True, audit_overturn_on=("specificity",))
    r = checker.check("Shadowhunters did not begin its second season in January 2017.")
    assert r.verdict == "REFUTED"                                # negation guard holds
    assert r.flags["audit"]["negated_claim"] is True
    assert r.flags["audit"]["overturned"] is False


def test_audit_is_off_by_default_so_the_paper_system_is_unchanged():
    checker, seen = _checker()
    r = checker.check("Coeliac disease is treated by maintaining a healthy diet.")
    assert r.verdict == "REFUTED"                     # no second pass, no overturn
    assert r.flags.get("audit") is None
    assert not any("classify the mismatch" in p for p in seen)


def test_audit_never_fires_on_a_supported_verdict():
    from wkgfc.agent import FactChecker
    from wkgfc.config import Config
    from wkgfc.llm import MockLLM
    cfg = Config(use_kg=False, use_web=False, use_agent=False, audit_refutations=True)
    calls = []

    def responder(prompt, system):
        calls.append(prompt)
        return '{"verdict": "SUPPORTED", "confidence": 0.9, "justification": "ok"}'

    r = FactChecker(cfg, MockLLM(cfg, responder)).check("The sky is blue.")
    assert r.verdict == "SUPPORTED" and r.flags.get("audit") is None
    assert len(calls) == 1                            # exactly one verdict call, no audit


def test_unparseable_audit_keeps_the_original_verdict():
    from wkgfc.agent import FactChecker
    from wkgfc.config import Config
    from wkgfc.llm import MockLLM
    cfg = Config(use_kg=False, use_web=False, use_agent=False, audit_refutations=True)

    def responder(prompt, system):
        if "classify the mismatch" in prompt:
            return "not json at all"
        return '{"verdict": "REFUTED", "confidence": 0.95, "justification": "j"}'

    r = FactChecker(cfg, MockLLM(cfg, responder)).check("A claim.")
    assert r.verdict == "REFUTED"                     # fail closed, never silently flip
    assert r.flags["audit"]["overturned"] is False
    assert r.flags["audit"]["model_verdict"] == ""


def test_tolerance_note_reaches_the_verdict_prompt_only_when_enabled():
    for enabled in (False, True):
        checker, seen = _checker(verdict_tolerance=enabled)
        checker.check("A claim.")
        has_note = any("absence of evidence is not contradiction" in p for p in seen)
        assert has_note is enabled


def test_extension_variants_are_registered():
    from wkgfc.config import VARIANTS
    assert VARIANTS["wkgfc"].get("audit_refutations") is None          # paper system untouched
    assert VARIANTS["wkgfc_tol_audit"]["verdict_tolerance"] is True
    assert VARIANTS["wkgfc_tol_audit"]["audit_refutations"] is True
