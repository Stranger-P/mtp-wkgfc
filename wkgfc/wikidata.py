"""Wikidata access: entity linking (wbsearchentities) and the two SPARQL templates
(outgoing / incoming relations) used by expand-and-prune (paper Sec. 4.2)."""
from __future__ import annotations

import threading

from typing import Protocol

from .parallel import pmap
from .cache import Cache, NullCache
from .http import HTTPError, get_json

SPARQL_URL = "https://query.wikidata.org/sparql"
API_URL = "https://www.wikidata.org/w/api.php"

# Relations that carry no evidential value (adapted from Think-on-Graph's abandon_rels for Wikidata).
_USELESS = {
    "category's main topic", "topic's main category", "stack exchange site", "commons category",
    "commons gallery", "described by source", "on focus list of wikimedia project", "page banner",
    "different from", "said to be the same as", "maintained by wikiproject", "topic has template",
    "image", "logo image", "signature", "pronunciation audio", "locator map image",
}
_USELESS_SUFFIX = (" ID", " id", " code", " number", "URL", "website", " username", " handle", " image",
                   " category", " template")


def useless_relation(label: str) -> bool:
    low = label.lower()
    return (low in _USELESS or label.endswith(_USELESS_SUFFIX) or "wikimedia" in low
            or "wikidata" in low or "commons" in low)


class KGClient(Protocol):
    def search_entities(self, name: str, limit: int = 5) -> list[dict]: ...
    def relations(self, qid: str) -> list[dict]: ...
    def values(self, qid: str, pid: str, direction: str, limit: int) -> list[dict]: ...
    def label(self, qid: str) -> str: ...


class WikidataClient:
    """Live client for the public Wikidata endpoints, with caching.

    Returned shapes:
      search_entities -> [{"id": "Q42", "label": "...", "description": "..."}]
      relations       -> [{"pid": "P69", "label": "educated at", "direction": "out"|"in"}]
      values          -> [{"id": "Q691283" or "", "label": "St John's College"}]
    """

    def __init__(self, cfg, cache: Cache | None = None):
        self.cfg = cfg
        self.cache = cache or NullCache()
        self.headers = {"User-Agent": cfg.wikidata_user_agent, "Accept": "application/sparql-results+json"}
        self.lang = cfg.language
        self.queries = 0  # live SPARQL queries (cost reporting)
        self._count_lock = threading.Lock()
        # Wikidata Query Service allows only a few parallel queries per client: one cap for the whole
        # process, shared by every claim worker.
        self._sem = _shared_semaphore(max(1, getattr(cfg, "kg_parallel_requests", 1)))

    # ------------------------------------------------------------------
    def _sparql(self, query: str) -> list[dict]:
        key = Cache.make_key("sparql", query)

        def run():
            with self._count_lock:
                self.queries += 1
            try:
                with self._sem:
                    data = get_json(SPARQL_URL, {"query": query, "format": "json"}, self.headers,
                                    timeout=self.cfg.request_timeout)
            except HTTPError:
                return []  # timeouts on huge entities -> treat as no result
            return data.get("results", {}).get("bindings", [])

        return self.cache.get_or_compute(key, run)

    @staticmethod
    def _qid(uri: str) -> str:
        return uri.rsplit("/", 1)[-1] if uri.startswith("http://www.wikidata.org/entity/") else ""

    # ------------------------------------------------------------------
    def search_entities(self, name: str, limit: int = 5) -> list[dict]:
        key = Cache.make_key("wbsearch", name, self.lang, limit)

        def run():
            try:
                data = get_json(API_URL, {
                    "action": "wbsearchentities", "search": name, "language": self.lang,
                    "uselang": self.lang, "type": "item", "limit": limit, "format": "json",
                }, {"User-Agent": self.cfg.wikidata_user_agent}, timeout=self.cfg.request_timeout)
            except HTTPError:
                return []
            return [{"id": r["id"], "label": r.get("label", name), "description": r.get("description", "")}
                    for r in data.get("search", [])]

        return self.cache.get_or_compute(key, run)

    def relations(self, qid: str) -> list[dict]:
        """SPARQL template 1+2: distinct outgoing and incoming direct-claim properties of an entity."""
        lang = self.lang
        out_q = f"""
SELECT DISTINCT ?prop ?propLabel WHERE {{
  wd:{qid} ?wdt ?o .
  ?prop wikibase:directClaim ?wdt .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
}}"""
        in_q = f"""
SELECT ?prop ?propLabel WHERE {{
  {{ SELECT DISTINCT ?wdt WHERE {{ ?s ?wdt wd:{qid} . }} LIMIT 300 }}
  ?prop wikibase:directClaim ?wdt .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
}}"""
        rels = []
        pairs = ((out_q, "out"), (in_q, "in"))
        results = pmap(lambda qd: self._sparql(qd[0]), pairs, getattr(self.cfg, "kg_parallel_requests", 1))
        for (q, direction), rows in zip(pairs, results):
            for b in rows:
                pid = b["prop"]["value"].rsplit("/", 1)[-1]
                label = b.get("propLabel", {}).get("value", pid)
                if not useless_relation(label):
                    rels.append({"pid": pid, "label": label, "direction": direction})
        return rels

    def values(self, qid: str, pid: str, direction: str, limit: int = 10) -> list[dict]:
        lang = self.lang
        if direction == "out":
            q = f"""
SELECT ?v ?vLabel WHERE {{
  wd:{qid} wdt:{pid} ?v .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
}} LIMIT {limit}"""
        else:
            q = f"""
SELECT ?v ?vLabel WHERE {{
  ?v wdt:{pid} wd:{qid} .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
}} LIMIT {limit}"""
        out = []
        for b in self._sparql(q):
            raw = b["v"]["value"]
            vid = self._qid(raw)
            label = b.get("vLabel", {}).get("value", raw) if vid else _clean_literal(raw)
            if vid and label == vid:  # entity without an English label: skip (ToG drops "UnName" nodes)
                continue
            out.append({"id": vid, "label": label})
        return out

    def label(self, qid: str) -> str:
        rows = self._sparql(f'SELECT ?l WHERE {{ wd:{qid} rdfs:label ?l . FILTER(LANG(?l)="{self.lang}") }} LIMIT 1')
        return rows[0]["l"]["value"] if rows else qid


_SEM = None
_SEM_LOCK = threading.Lock()


def _shared_semaphore(n: int):
    global _SEM
    with _SEM_LOCK:
        if _SEM is None:
            _SEM = threading.BoundedSemaphore(n)
        return _SEM


def _clean_literal(v: str) -> str:
    # "1952-03-11T00:00:00Z" -> "1952-03-11"
    if len(v) >= 20 and v[4] == "-" and v.endswith("Z") and "T" in v:
        d = v.split("T")[0]
        return d[:4] if d.endswith("-01-01") else d
    return v
