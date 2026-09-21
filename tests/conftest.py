"""Shared test fixtures: a known JSON-LD document and its expected RDF shape."""

from __future__ import annotations

import pytest

SAMPLE_JSONLD = {
    "candidate": {"fullName": "Shrijit Srivastav", "email": "shrijit.work7@gmail.com"},
    "skills": [
        {"label": "Python", "proficiency": "advanced"},
        {"label": "TypeScript", "proficiency": "advanced"},
        {"label": "RDF", "proficiency": "intermediate"},
        {"label": "GraphRAG", "proficiency": "intermediate"},
    ],
    "roles": [
        {
            "title": "AI Integration Engineer",
            "employer": "RapidData.io",
            "startDate": "2025-07",
            "endDate": "2025-12",
            "summary": "Built LLM resume-processing pipelines.",
        }
    ],
    "projects": [
        {
            "name": "cv-knowledge-graph",
            "role": "Author",
            "technologies": ["RDF", "Python"],
            "description": "RDF knowledge graph pipeline for CVs.",
            "url": "https://github.com/shrijit37/cv-knowledge-graph",
        }
    ],
    "education": [
        {
            "degree": "B.Tech",
            "field": "Electronics & Communication Engineering",
            "institution": "Jaypee Institute of Information Technology",
            "graduatedAt": "2024",
        }
    ],
}


@pytest.fixture
def sample_jsonld() -> dict:
    return SAMPLE_JSONLD


@pytest.fixture
def graph(sample_jsonld):
    from cvkg.build_graph import build_graph

    return build_graph(
        sample_jsonld,
        source="sample.pdf",
        extractor_model="pytest",
    )
