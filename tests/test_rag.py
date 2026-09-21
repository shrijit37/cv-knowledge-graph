"""Tests for the GraphRAG retrieval layer."""

from __future__ import annotations

import numpy as np
import pytest

from cvkg.query import candidate_facts
from cvkg.rag import GraphRAG


class FakeEmbedder:
    """Deterministic embedder: 1-hot-ish vectors keyed by lowercase words."""

    def __call__(self, texts):
        vocab = {
            "skill:", "python", "rdf", "graphrag", "role:", "project:",
            "education:", "typescript", "ai integration engineer",
        }
        rows = []
        for t in texts:
            row = np.zeros(len(vocab), dtype=float)
            for i, w in enumerate(vocab):
                if w.strip(":").lower() in t.lower():
                    row[i] = 1.0
            rows.append(row)
        return np.array(rows)


def test_graphrag_retrieves_grounded_facts(graph):
    rag = GraphRAG(graph, embedder=FakeEmbedder(), top_k=8)
    items = rag.retrieve("which candidate knows GraphRAG?", candidate="Shrijit Srivastav")
    assert items, "expected at least one fact"
    assert any("graphrag" in i["text"].lower() for i in items)


def test_graphrag_lists_candidates_from_all_graphs(graph):
    rag = GraphRAG(graph, embedder=FakeEmbedder())
    items = rag.retrieve("who has RDF experience?")
    assert any("shrijit" in i["candidate"].lower() for i in items)


def test_candidate_facts_shape(graph):
    facts = candidate_facts(graph, "Shrijit Srivastav")
    assert isinstance(facts["skills"], list)
    assert isinstance(facts["roles"], list)
    assert isinstance(facts["projects"], list)
    assert isinstance(facts["education"], list)
    assert any("python" in s.lower() for s in facts["skills"])


def test_graphrag_no_candidate_no_crash(graph):
    rag = GraphRAG(graph, embedder=FakeEmbedder())
    items = rag.retrieve("unrelated question about nothing")
    assert isinstance(items, list)
