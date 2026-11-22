"""Web Retrieval Module (paper Sec. 4.3).

Coarse-to-fine filtering: Serper (Google) results -> BM25 ranking against the claim + query ->
LLM consistency filter with a confidence per passage. Web-to-KG integration: surviving sentences
become triples aligned with the KG schema (duplicates skipped, extra details kept as annotations).
Web evidence is treated as hypothesis-expanding, not authoritative."""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from urllib.parse import urlparse

from . import prompts
from .cache import Cache, NullCache
from .http import HTTPError, get_text, post_json
from .kg import EvidenceGraph, Triple

# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------


class SerperSearch:
    URL = "https://google.serper.dev/search"

    def __init__(self, cfg, cache: Cache | None = None):
        self.cfg = cfg
        self.cache = cache or NullCache()
        self.calls = 0

    def search(self, query: str) -> list[dict]:
        key = Cache.make_key("serper", query, self.cfg.web_results_per_query)

        def run():
            api_key = os.environ.get("SERPER_API_KEY")
            if not api_key:
                raise RuntimeError("SERPER_API_KEY is not set (add it to .env).")
            self.calls += 1
            try:
                data = post_json(self.URL, {"q": query, "num": self.cfg.web_results_per_query},
                                 {"X-API-KEY": api_key}, timeout=self.cfg.request_timeout)
            except HTTPError:
                return []
            results = []
            ab = data.get("answerBox")
            if ab:
                results.append({"title": ab.get("title", "answer box"), "link": ab.get("link", ""),
                                "snippet": ab.get("answer") or ab.get("snippet", ""), "date": ""})
            for r in data.get("organic", []):
                results.append({"title": r.get("title", ""), "link": r.get("link", ""),
                                "snippet": r.get("snippet", ""), "date": r.get("date", "")})
            return results

        return self.cache.get_or_compute(key, run)


class MockSearch:
    """Offline search for tests: `pages` maps a keyword to a list of result dicts."""

    def __init__(self, pages: dict[str, list[dict]]):
        self.pages = pages
        self.calls = 0

    def search(self, query: str) -> list[dict]:
        self.calls += 1
        q = query.lower()
        out = []
        for kw, res in self.pages.items():
            if kw.lower() in q:
                out.extend(res)
        return out


# ---------------------------------------------------------------------------
# BM25 (small self-contained implementation; same formula as rank_bm25.BM25Okapi)
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of in on at to for and or is was were be by with as from that this it its are".split())


def _tok(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def bm25_rank(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    toks = [_tok(d) for d in docs]
    n = len(toks)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in toks) / n or 1.0
    df = Counter(w for t in toks for w in set(t))
    q = _tok(query)
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log((n - df[w] + 0.5) / (df[w] + 0.5) + 1)
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avgdl))
        scores.append(s)
    return scores


# ---------------------------------------------------------------------------
# Web retriever
# ---------------------------------------------------------------------------


class WebRetriever:
    def __init__(self, cfg, search_backend, llm):
        self.cfg = cfg
        self.search = search_backend
        self.llm = llm

    def make_query(self, claim: str) -> str:
        """Query for the non-agent variants (paper Variant 4: queries come from the claim, not the KG)."""
        out = self.llm.complete_json(prompts.WEB_QUERY_FROM_CLAIM.format(claim=claim),
                                     purpose="web_query", default={})
        q = out.get("query") if isinstance(out, dict) else None
        return q or claim

    # ------------------------------------------------------------------
    def retrieve(self, claim: str, query: str, graph: EvidenceGraph) -> dict:
        """Action webSearch(query, currentKG). Mutates `graph`; returns a short report."""
        graph.web_queries.append(query)
        results = [r for r in self.search.search(query) if not self._blocked(r.get("link", ""))]
        passages = self._passages(results)
        if not passages:
            return {"query": query, "results": 0, "kept": 0, "new_triples": 0}

        # Coarse: BM25 against claim + query
        scores = bm25_rank(f"{claim} {query}", [p["text"] for p in passages])
        ranked = [p for _, p in sorted(zip(scores, passages), key=lambda x: -x[0])]
        ranked = ranked[: self.cfg.bm25_top_passages]

        # Fine: LLM relevance + consistency confidence
        kept = self._llm_filter(claim, ranked)
        new = self._integrate(claim, kept, graph) if kept else 0
        return {"query": query, "results": len(results), "kept": len(kept), "new_triples": new}

    # ------------------------------------------------------------------
    def _blocked(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return any(host == d or host.endswith("." + d) for d in self.cfg.blocked_domains)

    def _passages(self, results: list[dict]) -> list[dict]:
        out = []
        for r in results:
            text = " ".join(x for x in (r.get("title", ""), r.get("snippet", "")) if x).strip()
            if text:
                out.append({"text": text, "url": r.get("link", ""), "date": r.get("date", "")})
            if self.cfg.fetch_full_pages and r.get("link"):
                for chunk in _page_chunks(r["link"]):
                    out.append({"text": chunk, "url": r["link"], "date": r.get("date", "")})
        return out

    def _llm_filter(self, claim: str, passages: list[dict]) -> list[dict]:
        listing = "\n".join(f"[{i}] ({p['url']}) {p['text'][:700]}" for i, p in enumerate(passages, 1))
        out = self.llm.complete_json(prompts.WEB_FILTER.format(claim=claim, passages=listing),
                                     purpose="web_filter", default={})
        kept = []
        for item in out.get("passages", []) if isinstance(out, dict) else []:
            try:
                i = int(item["id"])
                conf = float(item.get("confidence", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= i <= len(passages) and item.get("relevant") and conf >= self.cfg.min_web_confidence:
                p = dict(passages[i - 1])
                p["confidence"] = conf
                p["key_sentences"] = item.get("key_sentences") or p["text"]
                kept.append(p)
        return kept

    def _integrate(self, claim: str, kept: list[dict], graph: EvidenceGraph) -> int:
        entities = sorted({t.subject for t in graph.triples} | {t.obj for t in graph.triples}
                          | set(graph.topic_entities.values()))[:80]
        relations = sorted({t.relation for t in graph.kg_triples()})[:60]
        sentences = "\n".join(f"[{i}] ({p['url']}) {p['key_sentences']}" for i, p in enumerate(kept, 1))
        out = self.llm.complete_json(
            prompts.WEB_TO_TRIPLES.format(claim=claim, entities="; ".join(entities) or "(none)",
                                          relations="; ".join(relations) or "(none)", sentences=sentences),
            purpose="web_to_kg", default={})
        if not isinstance(out, dict):
            return 0
        added = 0
        for t in out.get("triples", []):
            try:
                s, r, o = str(t["subject"]), str(t["relation"]), str(t["object"])
            except (KeyError, TypeError):
                continue
            src = kept[_src_index(t, len(kept))]
            if graph.add(Triple(s, r, o, "web", url=src["url"], confidence=src["confidence"])):
                added += 1
        for a in out.get("annotations", []):
            if isinstance(a, dict) and a.get("entity") and a.get("text"):
                graph.annotate(str(a["entity"]), str(a["text"]))
        return added


def _src_index(item: dict, n: int) -> int:
    try:
        i = int(item.get("source", 1)) - 1
    except (TypeError, ValueError):
        i = 0
    return min(max(i, 0), n - 1)


def _page_chunks(url: str, max_chunks: int = 3, size: int = 600) -> list[str]:
    try:
        html = get_text(url)
    except Exception:
        return []
    try:
        import trafilatura

        text = trafilatura.extract(html) or ""
    except ImportError:
        text = re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", html, flags=re.S)
    text = " ".join(text.split())
    return [text[i:i + size] for i in range(0, min(len(text), size * max_chunks), size)]


def make_search(cfg, cache=None, mock_pages=None):
    if cfg.search_backend == "mock":
        return MockSearch(mock_pages or {})
    return SerperSearch(cfg, cache)
