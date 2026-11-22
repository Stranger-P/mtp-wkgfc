"""Topic-entity extraction (spaCy NER) and linking to Wikidata (paper Sec. 4.2, 'KG Entity Mapping')."""
from __future__ import annotations

import re

from . import prompts

_SKIP_LABELS = {"CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY", "TIME"}
_nlp = None


def _load_spacy():
    global _nlp
    if _nlp is None:
        try:
            import spacy

            for name in ("en_core_web_trf", "en_core_web_lg", "en_core_web_sm"):
                try:
                    _nlp = spacy.load(name)
                    break
                except OSError:
                    continue
        except ImportError:
            _nlp = False
        if _nlp is None:
            _nlp = False
    return _nlp or None


def extract_mentions(claim: str, llm=None) -> list[str]:
    """Named entities from spaCy; noun-chunk / capitalised-span / LLM fallbacks when NER finds nothing."""
    nlp = _load_spacy()
    mentions: list[str] = []
    if nlp is not None:
        doc = nlp(claim)
        mentions = [e.text for e in doc.ents if e.label_ not in _SKIP_LABELS]
        if not mentions:
            mentions = [nc.text for nc in doc.noun_chunks if nc.root.pos_ == "PROPN"]
    if not mentions:
        mentions = _capitalised_spans(claim)
    if not mentions and llm is not None:
        out = llm.complete_json(prompts.ENTITY_EXTRACTION_FALLBACK.format(claim=claim),
                                purpose="entity_extraction", default={})
        mentions = [m for m in out.get("entities", []) if isinstance(m, str)]
    seen, uniq = set(), []
    for m in mentions:
        m = m.strip(" .,'\"").removeprefix("the ").removeprefix("The ")
        if m and m.lower() not in seen:
            seen.add(m.lower())
            uniq.append(m)
    return uniq


def _capitalised_spans(text: str) -> list[str]:
    spans = re.findall(r"(?:[A-Z][\w'\-.]*(?:\s+(?:of|the|de|and|von|van)\s+|\s+)?)+", text)
    out = []
    for s in spans:
        s = s.strip()
        if s and not (len(s.split()) == 1 and text.startswith(s)):  # skip sentence-initial single word
            out.append(s)
    return out


def link_entities(claim: str, mentions: list[str], kg, llm=None, disambiguate: bool = False,
                  candidates_per_mention: int = 5) -> dict[str, str]:
    """Map mentions to Wikidata QIDs. Default = top wbsearchentities hit (paper);
    disambiguate=True lets the LLM pick among candidates using their descriptions (improvement B)."""
    linked: dict[str, str] = {}
    for m in mentions:
        cands = kg.search_entities(m, limit=candidates_per_mention)
        if not cands:
            continue
        choice = cands[0]
        if disambiguate and llm is not None and len(cands) > 1:
            listing = "\n".join(f"{i}. {c['label']} ({c['id']}): {c.get('description', '')}"
                                for i, c in enumerate(cands, 1))
            out = llm.complete_json(prompts.ENTITY_DISAMBIGUATION.format(claim=claim, mention=m, candidates=listing),
                                    purpose="entity_linking", default={})
            idx = out.get("choice", 1)
            if isinstance(idx, int) and 1 <= idx <= len(cands):
                choice = cands[idx - 1]
            elif idx == 0:
                continue
        linked[choice["id"]] = choice["label"]
    return linked
