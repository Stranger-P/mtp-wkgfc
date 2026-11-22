"""Knowledge Graph Retrieval Module (paper Sec. 4.2): expand-and-prune beam search over Wikidata.

Per hop, for each frontier entity:
  1. SPARQL templates fetch its outgoing + incoming relations;
  2. the LLM keeps the top-k relations (relation pruning);
  3. SPARQL fetches the values of each kept relation (all become evidence triples);
  4. the LLM scores candidate entities; the global top-k (relation score x entity score) form the
     next frontier (entity pruning).
Cost per hop is about (#frontier) relation-prune calls + (#kept relations) entity-score calls,
matching the paper's N + k*N + 1 estimate."""
from __future__ import annotations

from . import prompts
from .entities import extract_mentions, link_entities
from .kg import EvidenceGraph, Triple
from .parallel import pmap


class KGRetriever:
    def __init__(self, cfg, kg_client, llm):
        self.cfg = cfg
        self.kg = kg_client
        self.llm = llm
        self.k = cfg.beam_size_k

    # ------------------------------------------------------------------
    def init_retrieval(self, claim: str, graph: EvidenceGraph) -> EvidenceGraph:
        """Action a0 = initKGRetrieval(claim): NER -> entity linking -> initial hops."""
        mentions = extract_mentions(claim, self.llm)
        linked = link_entities(claim, mentions, self.kg, self.llm,
                               disambiguate=self.cfg.extra.get("llm_entity_disambiguation", False))
        graph.topic_entities = dict(linked)
        graph.frontier = dict(linked)
        graph.mentions = mentions
        for _ in range(self.cfg.init_hops):
            if not graph.frontier or graph.hops_done >= self.cfg.max_hops_N:
                break
            self.expand_hop(claim, graph)
        return graph

    def hops_left(self, graph: EvidenceGraph) -> int:
        return max(0, self.cfg.max_hops_N - graph.hops_done)

    # ------------------------------------------------------------------
    def expand_hop(self, claim: str, graph: EvidenceGraph) -> int:
        """Action expandKG(): one more hop from the frontier, then prune. Returns #new triples."""
        if not graph.frontier:
            return 0
        before = len(graph.triples)
        candidates: list[tuple[float, str, str]] = []  # (score, qid, label)
        visited = graph.visited
        visited.update(graph.topic_entities)

        workers = getattr(self.cfg, "kg_parallel_requests", 1)
        frontier = list(graph.frontier.items())[: self.k]

        # Stage A (independent per entity): candidate relations + LLM relation pruning, in parallel.
        def prep(item):
            qid, name = item
            rels = self._candidate_relations(qid, graph)
            return self._prune_relations(claim, name, rels) if rels else None

        pruned_all = pmap(prep, frontier, workers)

        # Stage B: entities in the original order, so `visited` filtering is exactly as before.
        for (qid, name), pruned in zip(frontier, pruned_all):
            visited.add(qid)
            if pruned is None:
                continue
            for rel, _ in pruned:
                graph.explored_relations.add((qid, rel["pid"], rel["direction"]))
            vals_all = pmap(lambda rp: self.kg.values(qid, rp[0]["pid"], rp[0]["direction"],
                                                      self.cfg.max_objects_per_relation), pruned, workers)
            jobs = []
            for (rel, rel_score), vals in zip(pruned, vals_all):
                ent_vals = []
                for v in vals:
                    if rel["direction"] == "out":
                        t = Triple(name, rel["label"], v["label"], "kg", qid, rel["pid"], v["id"])
                    else:
                        t = Triple(v["label"], rel["label"], name, "kg", v["id"], rel["pid"], qid)
                    graph.add(t)
                    if v["id"] and v["id"] not in visited:
                        ent_vals.append(v)
                jobs.append((rel, rel_score, ent_vals))
            scored_all = pmap(lambda j: self._score_entities(claim, name, j[0], j[2], j[1]), jobs, workers)
            for scored in scored_all:
                for v, sc in scored:
                    candidates.append((sc, v["id"], v["label"]))

        best: dict[str, tuple[float, str]] = {}
        for s, qid, label in candidates:
            if qid not in best or s > best[qid][0]:
                best[qid] = (s, label)
        ranked = sorted(best.items(), key=lambda x: -x[1][0])[: self.k]
        graph.frontier = {qid: label for qid, (s, label) in ranked}
        graph.hops_done += 1
        return len(graph.triples) - before

    # ------------------------------------------------------------------
    def _candidate_relations(self, qid: str, graph: EvidenceGraph) -> list[dict]:
        seen, out = set(), []
        for r in self.kg.relations(qid):
            key = (r["pid"], r["direction"])
            if key in seen or (qid, r["pid"], r["direction"]) in graph.explored_relations:
                continue
            seen.add(key)
            out.append(r)
        out.sort(key=lambda r: (r["direction"] != "out", r["label"]))  # stable prompt order
        return out[: self.cfg.max_relations_per_entity]

    def _prune_relations(self, claim: str, entity: str, rels: list[dict]) -> list[tuple[dict, float]]:
        if len(rels) <= self.k:
            return [(r, 1.0 / len(rels)) for r in rels]
        listing = "\n".join(f"{i}. {r['label']}{' (incoming)' if r['direction'] == 'in' else ''}"
                            for i, r in enumerate(rels, 1))
        out = self.llm.complete_json(
            prompts.RELATION_PRUNE.format(k=self.k, claim=claim, entity=entity, relations=listing),
            purpose="relation_prune", default={})
        picked = []
        for item in out.get("relations", []) if isinstance(out, dict) else []:
            try:
                i, s = int(item["id"]), float(item.get("score", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= i <= len(rels) and all(p[0] is not rels[i - 1] for p in picked):
                picked.append((rels[i - 1], s))
        if not picked:  # format error -> fall back to the first k (ToG returns nothing; we keep going)
            picked = [(r, 1.0 / self.k) for r in rels[: self.k]]
        picked.sort(key=lambda x: -x[1])
        return picked[: self.k]

    def _score_entities(self, claim, head, rel, ents, rel_score) -> list[tuple[dict, float]]:
        if not ents:
            return []
        if len(ents) == 1:
            return [(ents[0], rel_score)]
        ents = sorted(ents, key=lambda v: v["label"])[:20]
        listing = "\n".join(f"{i}. {v['label']}" for i, v in enumerate(ents, 1))
        out = self.llm.complete_json(
            prompts.ENTITY_SCORE.format(claim=claim, head=head, relation=rel["label"], candidates=listing),
            purpose="entity_score", default={})
        scores = out.get("scores", []) if isinstance(out, dict) else []
        try:
            scores = [float(x) for x in scores][: len(ents)]
        except (TypeError, ValueError):
            scores = []
        if len(scores) != len(ents) or sum(scores) <= 0:
            scores = [1.0 / len(ents)] * len(ents)
        return [(v, s * rel_score) for v, s in zip(ents, scores)]
