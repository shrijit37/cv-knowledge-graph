"""Turn extracted JSON-LD into an RDF knowledge graph (via rdflib).

The graph is written as Turtle (``.ttl``) and can be loaded into any RDF store
(GraphDB, Jena Fuseki, RDFox, Neptune, ...) or queried in-process with SPARQL
(see :mod:`cvkg.query`).

Every extracted statement is linked to an :class:`Evidence` node that records
the source file, the extraction timestamp, and the model that produced it, so
the graph carries provenance/traceability end-to-end (the JD asks for
"traceability to source documents").
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

KG = Namespace("http://cv-knowledge-graph.example/data#")
ONT = Namespace("http://cv-knowledge-graph.example/ontology#")


def _uri(parts: list[str]) -> URIRef:
    slug = "-".join(
        re.sub(r"[^a-zA-Z0-9]+", "-", str(p)).strip("-").lower() for p in parts
    )
    return URIRef(str(KG[slug or "node"]))


def build_graph(
    data: dict[str, Any],
    source: str,
    extractor_model: str = "unknown",
    extracted_at: datetime | None = None,
) -> Graph:
    """Build an RDF graph from a JSON-LD document emitted by an extractor.

    Returns an :class:`rdflib.Graph` holding the candidate, skills, roles,
    projects, education and one Evidence node per statement group.
    """
    graph = Graph()
    graph.bind("kg", KG)
    graph.bind("ont", ONT)
    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)

    extracted_at = extracted_at or datetime.now(timezone.utc)
    candidate = data.get("candidate") or {}
    name = (candidate.get("fullName") or "candidate").strip()
    cand_uri = _uri([name])

    # -- candidate ------------------------------------------------------
    graph.add((cand_uri, RDF.type, ONT.Candidate))
    if candidate.get("fullName"):
        graph.add((cand_uri, ONT.fullName, Literal(candidate["fullName"])))
    if candidate.get("email"):
        graph.add((cand_uri, ONT.email, Literal(candidate["email"])))

    # -- skills ---------------------------------------------------------
    for skill in data.get("skills", []):
        label = (skill.get("label") or "").strip()
        if not label:
            continue
        skill_uri = _uri([name, "skill", label])
        graph.add((skill_uri, RDF.type, ONT.Skill))
        graph.add((skill_uri, ONT.hasLabel, Literal(label)))
        if skill.get("proficiency"):
            graph.add((skill_uri, ONT.proficiency, Literal(skill["proficiency"])))
        graph.add((cand_uri, ONT.hasSkill, skill_uri))
        _attach_evidence(graph, skill_uri, ONT.skillEvidence, source, extracted_at, extractor_model)

    # -- roles ----------------------------------------------------------
    for role in data.get("roles", []):
        title = (role.get("title") or "role").strip()
        employer = (role.get("employer") or "").strip()
        role_uri = _uri([name, "role", title, employer])
        graph.add((role_uri, RDF.type, ONT.Role))
        graph.add((role_uri, ONT.title, Literal(title)))
        graph.add((cand_uri, ONT.heldRole, role_uri))
        if employer:
            emp_uri = _uri([employer])
            graph.add((emp_uri, RDF.type, ONT.Employer))
            graph.add((role_uri, ONT.employer, emp_uri))
        for key in ("startDate", "endDate"):
            val = role.get(key)
            if val:
                graph.add((role_uri, ONT[key], Literal(val, datatype=XSD.gYearMonth)))
        _attach_evidence(graph, role_uri, ONT.roleEvidence, source, extracted_at, extractor_model)

    # -- projects -------------------------------------------------------
    for project in data.get("projects", []):
        pname = (project.get("name") or "project").strip()
        proj_uri = _uri([name, "project", pname])
        graph.add((proj_uri, RDF.type, ONT.Project))
        graph.add((proj_uri, ONT.projectOf, cand_uri))
        if project.get("description"):
            graph.add((proj_uri, ONT.description, Literal(project["description"])))
        for tech in project.get("technologies", []):
            skill_uri = _uri([name, "skill", tech])
            graph.add((skill_uri, RDF.type, ONT.Skill))
            graph.add((skill_uri, ONT.hasLabel, Literal(tech)))
            graph.add((proj_uri, ONT.projectUsesSkill, skill_uri))
            graph.add((cand_uri, ONT.hasSkill, skill_uri))
        _attach_evidence(graph, proj_uri, ONT.projectEvidence, source, extracted_at, extractor_model)

    # -- education ------------------------------------------------------
    for edu in data.get("education", []):
        degree = (edu.get("degree") or "degree").strip()
        edu_uri = _uri([name, "education", degree])
        graph.add((edu_uri, RDF.type, ONT.Education))
        graph.add((cand_uri, ONT.educatedAt, edu_uri))
        if edu.get("degree"):
            graph.add((edu_uri, ONT.degreeName, Literal(edu["degree"])))
        if edu.get("field"):
            graph.add((edu_uri, ONT.field, Literal(edu["field"])))
        inst = edu.get("institution")
        if inst:
            inst_uri = _uri([inst])
            graph.add((inst_uri, RDF.type, ONT.Institution))
            graph.add((edu_uri, ONT.degree, inst_uri))
        if edu.get("graduatedAt"):
            graph.add((edu_uri, ONT.graduatedAt, Literal(edu["graduatedAt"], datatype=XSD.gYear)))
        _attach_evidence(graph, edu_uri, ONT.skillEvidence if False else ONT.roleEvidence, source, extracted_at, extractor_model)

    return graph


def _attach_evidence(
    graph: Graph,
    subject: URIRef,
    property: URIRef,
    source: str,
    extracted_at: datetime,
    model: str,
) -> None:
    """Create a provenance (Evidence) node for ``subject`` and link it."""
    digest = hashlib.sha1(f"{subject}|{source}|{extracted_at.isoformat()}".encode()).hexdigest()[:12]
    node = URIRef(str(KG[f"evidence-{digest}"]))
    graph.add((node, RDF.type, ONT.Evidence))
    graph.add((node, ONT.sourceFile, Literal(str(source))))
    graph.add((node, ONT.extractedAt, Literal(extracted_at.isoformat(), datatype=XSD.dateTime)))
    graph.add((node, ONT.extractorModel, Literal(model)))
    graph.add((subject, property, node))


def merge_graphs(paths: list[Path]) -> Graph:
    """Load multiple Turtle files into one graph (a multi-CV knowledge graph)."""
    merged = Graph()
    merged.bind("kg", KG)
    merged.bind("ont", ONT)
    for p in paths:
        merged.parse(p, format="turtle")
    return merged


def save(graph: Graph, path: Path | str) -> Path:
    path = Path(path)
    path.write_text(graph.serialize(format="turtle"), encoding="utf-8")
    return path
