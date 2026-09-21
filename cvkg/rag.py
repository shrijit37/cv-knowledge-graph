"""Hybrid GraphRAG retrieval over the knowledge graph.

GraphRAG means the LLM answers from *structured graph context* rather than flat
text chunks. This module implements that in three steps:

1. **Graph grounding** (SPARQL): locate the relevant candidate and pull their
   skills, roles, projects and education as graph-derived facts.
2. **Embedding ranking**: embed the query and each fact with
   ``sentence-transformers`` (local, no API) and keep the top-k.
3. **Lexical rerank**: rerank by token overlap to prefer facts that explicitly
   mention the query terms.

The returned context carries provenance (the graph-derived facts), so an LLM
prompted with it can answer grounded questions about candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .query import candidate_facts, candidates


@dataclass
class GraphRAG:
    """Lightweight GraphRAG over an :class:`rdflib.Graph`.

    ``embedder`` is any callable ``texts: list[str] -> np.ndarray`` of shape
    ``(len(texts), dim)``. The default uses the ``sentence_transformers``
    package; if it is not installed we fall back to a pure-scoring mode that
    skips embedding ranking.
    """

    graph: Any
    embedder: Any = None
    top_k: int = 8

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = _make_default_embedder()

    def retrieve(self, question: str, candidate: str | None = None) -> list[dict[str, Any]]:
        """Return ranked graph facts relevant to ``question``.

        Each item: ``{"text": ..., "score": float, "candidate": str}``.
        """
        # 1. graph grounding via SPARQL
        if candidate:
            names = [candidate]
        else:
            names = [r["name"] for r in candidates(self.graph) if r.get("name")]
        facts: list[tuple[str, str]] = []  # (candidate, fact text)
        for name in names:
            cf = candidate_facts(self.graph, name)
            for label in cf.get("skills", []):
                facts.append((name, f"Skill: {label}"))
            for label in cf.get("roles", []):
                facts.append((name, f"Role: {label}"))
            for label in cf.get("projects", []):
                facts.append((name, f"Project: {label}"))
            for label in cf.get("education", []):
                facts.append((name, f"Education: {label}"))

        if not facts:
            return []

        # 2. embedding ranking (best effort)
        texts = [f for _, f in facts]
        if self.embedder is not None:
            try:
                q_vec = self.embedder([question])[0]
                f_vecs = self.embedder(texts)
                scores = (f_vecs @ q_vec).tolist()
            except Exception:
                scores = [0.0] * len(texts)
        else:
            scores = [0.0] * len(texts)

        # 3. lexical rerank
        q_terms = {w for w in question.lower().split() if len(w) > 2}
        lex = [len(q_terms & set(f.lower().split())) / max(1, len(q_terms)) for f in texts]
        blended = [0.6 * s + 0.4 * l for s, l in zip(scores, lex)]

        ranked = sorted(
            zip(blended, texts, [n for n, _ in facts]),
            key=lambda t: t[0],
            reverse=True,
        )
        return [
            {"text": f, "score": round(float(sc), 4), "candidate": n}
            for sc, f, n in ranked[: self.top_k]
        ]

    def format_context(self, items: list[dict[str, Any]]) -> str:
        """Render retrieved facts as prompt-ready grounded context."""
        lines = []
        for it in items:
            lines.append(f"- [{it['candidate']}] {it['text']}  (score {it['score']})")
        return "\n".join(lines)


def _make_default_embedder() -> Any:
    """Try to load sentence-transformers; return None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None
