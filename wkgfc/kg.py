"""The evidence graph the agent observes: KG triples (high-precision anchors) plus
web-derived triples and textual annotations (hypothesis-expanding evidence), paper Sec. 4.3."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    obj: str
    source: str = "kg"          # "kg" | "web"
    subject_id: str = ""        # Wikidata QID when known
    relation_id: str = ""       # Wikidata PID when known
    obj_id: str = ""
    url: str = ""               # web provenance
    confidence: float = 1.0     # web consistency confidence (fine filter)

    def key(self) -> tuple:
        return (_norm(self.subject), _norm(self.relation), _norm(self.obj))

    def render(self) -> str:
        tag = "" if self.source == "kg" else f" [web, conf={self.confidence:.2f}]"
        return f"({self.subject}, {self.relation}, {self.obj}){tag}"


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").split())


@dataclass
class EvidenceGraph:
    triples: list[Triple] = field(default_factory=list)
    annotations: dict = field(default_factory=dict)      # entity name -> list[str] (web text)
    topic_entities: dict = field(default_factory=dict)   # QID -> label (seeds)
    frontier: dict = field(default_factory=dict)         # QID -> label (entities to expand next hop)
    explored_relations: set = field(default_factory=set)  # (QID, PID, direction)
    hops_done: int = 0
    web_queries: list = field(default_factory=list)
    visited: set = field(default_factory=set)            # QIDs already expanded
    mentions: list = field(default_factory=list)         # NER mentions from the claim

    def __post_init__(self):
        self._keys = {t.key() for t in self.triples}

    # --------------------------------------------------------------
    def add(self, t: Triple) -> bool:
        """Add a triple; returns False if an equivalent one already exists (skip rule, Sec. 4.3)."""
        k = t.key()
        if k in self._keys:
            return False
        self._keys.add(k)
        self.triples.append(t)
        return True

    def annotate(self, entity: str, text: str) -> None:
        notes = self.annotations.setdefault(entity, [])
        if text not in notes:
            notes.append(text)

    def entity_names(self) -> set[str]:
        names = set()
        for t in self.triples:
            names.add(_norm(t.subject))
            names.add(_norm(t.obj))
        return names

    def has_entity(self, name: str) -> bool:
        return _norm(name) in self.entity_names()

    def kg_triples(self) -> list[Triple]:
        return [t for t in self.triples if t.source == "kg"]

    def web_triples(self) -> list[Triple]:
        return [t for t in self.triples if t.source == "web"]

    # --------------------------------------------------------------
    def render(self, max_triples: int = 200) -> str:
        """Text form shown to the LLM. KG triples first (anchors), then web triples, then notes."""
        lines = []
        kg = self.kg_triples()
        web = self.web_triples()
        if kg:
            lines.append("Knowledge-graph triples (Wikidata):")
            lines += [f"- {t.render()}" for t in kg[:max_triples]]
        if web:
            lines.append("Web-derived triples:")
            lines += [f"- {t.render()}" for t in web[:max_triples]]
        if self.annotations:
            lines.append("Web annotations:")
            for ent, notes in self.annotations.items():
                for n in notes[:3]:
                    lines.append(f"- {ent}: {n}")
        if not lines:
            return "(no evidence retrieved yet)"
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "kg_triples": len(self.kg_triples()),
            "web_triples": len(self.web_triples()),
            "annotations": sum(len(v) for v in self.annotations.values()),
            "hops": self.hops_done,
            "web_queries": len(self.web_queries),
        }

    def to_dict(self) -> dict:
        return {
            "triples": [asdict(t) for t in self.triples],
            "annotations": self.annotations,
            "topic_entities": self.topic_entities,
            "stats": self.stats(),
        }
