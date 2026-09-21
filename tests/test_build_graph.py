"""Ontology and graph-construction tests."""

from __future__ import annotations

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from cvkg.build_graph import KG, ONT, build_graph
from cvkg.query import (
    CANDIDATE_STRENGTH_QUERY,
    CANDIDATES_QUERY,
    SKILL_BY_PROJECT_QUERY,
    SKILLS_QUERY,
    skills_of,
)


def test_ontology_loads(graph):
    """The built-in ontology TTL must parse as RDF."""
    graph.parse("ontology.ttl", format="turtle")
    assert len(graph) > 0


def test_candidate_class_present(graph):
    assert (URIRef(str(KG["shrijit-srivastav"])), RDF.type, ONT.Candidate) in graph


def test_skills_linked(graph, sample_jsonld):
    labels = set(s["label"] for s in sample_jsonld["skills"])
    actual = set()
    for s in graph.objects(None, ONT.hasLabel):
        if isinstance(s, Literal):
            actual.add(str(s))
    assert labels <= actual


def test_roles_and_provenance(graph):
    roles = list(graph.subjects(RDF.type, ONT.Role))
    assert roles, "expected at least one Role"
    with_prov = list(graph.triples((None, ONT.roleEvidence, None)))
    assert with_prov, "expected role evidence (provenance) nodes"
    _, _, ev = with_prov[0]
    assert any(graph.triples((ev, ONT.sourceFile, Literal("sample.pdf"))))


def test_project_uses_skill(graph):
    rows = list(graph.query(SKILL_BY_PROJECT_QUERY))
    tech = {str(r[2]) for r in rows}
    assert {"RDF", "Python"} <= tech


def test_query_candidates(graph):
    rows = list(graph.query(CANDIDATES_QUERY))
    assert any("Shrijit Srivastav" in str(r[1]) for r in rows)


def test_query_skills(graph, sample_jsonld):
    rows = list(graph.query(SKILLS_QUERY))
    labels = {str(r[1]) for r in rows}
    assert {"Python", "GraphRAG"} <= labels


def test_query_strength(graph):
    rows = list(graph.query(CANDIDATE_STRENGTH_QUERY))
    assert len(rows) == 1
    assert int(rows[0][1]) >= 4


def test_skills_of(graph):
    got = skills_of(graph, "Shrijit Srivastav")
    assert "Python" in got and "RDF" in got
    assert skills_of(graph, "Nobody") == []


def test_empty_skills_skipped():
    data = {
        "candidate": {"fullName": "A", "email": None},
        "skills": [{"label": "", "proficiency": None}],
        "roles": [],
        "projects": [],
        "education": [],
    }
    g = build_graph(data, source="x", extractor_model="t")
    assert len(list(g.subjects(RDF.type, ONT.Skill))) == 0
