"""Tests for the extraction layer (offline heuristic path and JSON cleaning)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cvkg.extract import (
    EXTRACTION_PROMPT,
    HeuristicExtractor,
    OpenAIExtractor,
    _clean_json,
)

SAMPLE_RESUME = """\
Shrijit Srivastav
shrijit.work7@gmail.com

AI Integration Engineer - RapidData.io | Jul 2025 - Dec 2025
Built LLM resume-processing pipelines with Python and TypeScript.
Worked with RDF, SPARQL, GraphDB, and GraphRAG concepts.

Project - cv-knowledge-graph
Technologies: Python, RDF, GraphRAG

Education
B.Tech Electronics & Communication Engineering
"""


def test_heuristic_extracts_skills():
    data = HeuristicExtractor().extract(SAMPLE_RESUME)
    labels = {s["label"] for s in data["skills"]}
    assert {"Python", "TypeScript", "RDF", "GraphRAG"} <= labels


def test_heuristic_candidate():
    data = HeuristicExtractor().extract(SAMPLE_RESUME)
    assert data["candidate"]["email"] == "shrijit.work7@gmail.com"


def test_heuristic_roles():
    data = HeuristicExtractor().extract(SAMPLE_RESUME)
    titles = {r["title"] for r in data["roles"]}
    assert "AI Integration Engineer" in titles


def test_clean_json_strips_fences():
    assert _clean_json("```json\n{\"a\": 1}\n```") == '{"a": 1}'


def test_clean_json_finds_object():
    assert _clean_json("prefix {\"a\": 1} suffix") == '{"a": 1}'


def test_extractors_are_callable():
    """Both extractors implement the Extractor protocol."""
    h = HeuristicExtractor()
    assert callable(h.extract)


def test_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIExtractor()


def test_prompt_mentions_rules():
    assert "Never invent" in EXTRACTION_PROMPT
    assert "JSON" in EXTRACTION_PROMPT
