"""SPARQL queries and a hybrid GraphRAG retrieval layer over the knowledge graph.

Two retrieval modes:

* :mod:`cvkg.query` - classic graph-native retrieval: SPARQL SELECT / ASK
  queries over the RDF graph. This is the "structured graph context" path.
* :mod:`cvkg.rag` - GraphRAG: candidate statements are located with SPARQL,
  ranked with embeddings plus a lexical rerank, and returned as grounded
  context (with provenance) for an LLM to answer from.

Together they demonstrate the "graph-native retrieval pipelines, combining
SPARQL queries, reasoning, and embeddings" the JD calls for.
"""

from __future__ import annotations

from typing import Iterable

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from .build_graph import KG, ONT

# ---------------------------------------------------------------------------
# SPARQL queries (graph-native retrieval)
# ---------------------------------------------------------------------------

CANDIDATES_QUERY = """\
PREFIX ont: <http://cv-knowledge-graph.example/ontology#>
SELECT DISTINCT ?candidate ?name WHERE {
  ?candidate a ont:Candidate .
  OPTIONAL { ?candidate ont:fullName ?name }
}
ORDER BY ?name
"""

SKILLS_QUERY = """\
PREFIX ont: <http://cv-knowledge-graph.example/ontology#>
SELECT ?candidate ?skill WHERE {
  ?candidate ont:hasSkill ?s .
  ?s ont:hasLabel ?skill .
}
ORDER BY ?candidate ?skill
"""

SKILL_BY_PROJECT_QUERY = """\
PREFIX ont: <http://cv-knowledge-graph.example/ontology#>
SELECT ?candidate ?project ?skill WHERE {
  ?project a ont:Project ;
           ont:projectOf ?candidate ;
           ont:projectUsesSkill ?s .
  ?s ont:hasLabel ?skill .
}
ORDER BY ?candidate ?project ?skill
"""

CANDIDATE_STRENGTH_QUERY = """\
PREFIX ont: <http://cv-knowledge-graph.example/ontology#>
SELECT ?candidate (COUNT(DISTINCT ?skill) AS ?skillCount) WHERE {
  ?candidate a ont:Candidate ;
             ont:hasSkill ?s .
  ?s ont:hasLabel ?skill .
}
GROUP BY ?candidate
ORDER BY DESC(?skillCount)
"""


def query(graph: Graph, sparql: str) -> list[dict[str, str]]:
    """Run a SPARQL query and return rows as dicts of string values."""
    rows = []
    for row in graph.query(sparql):
        rows.append({str(k): str(v) for k, v in row.asdict().items()})
    return rows


def candidates(graph: Graph) -> list[dict[str, str]]:
    return query(graph, CANDIDATES_QUERY)


def candidate_uri(graph: Graph, name: str) -> str | None:
    """Resolve a candidate by full name (case-insensitive) to a URI."""
    for r in query(graph, CANDIDATES_QUERY):
        if r.get("name", "").strip().lower() == name.strip().lower():
            return r["candidate"]
    return None


def skills_of(graph: Graph, name: str) -> list[str]:
    cand = candidate_uri(graph, name)
    if not cand:
        return []
    sparql = (
        "PREFIX ont: <http://cv-knowledge-graph.example/ontology#>\n"
        "SELECT ?skill WHERE { <%s> ont:hasSkill ?s . ?s ont:hasLabel ?skill . }"
        % cand
    )
    return [r["skill"] for r in query(graph, sparql)]


def candidate_facts(graph: Graph, name: str) -> dict[str, list[str]]:
    """Human-readable facts about a candidate: skills, roles, projects, education."""
    cand = candidate_uri(graph, name)
    out = {"skills": [], "roles": [], "projects": [], "education": []}
    if not cand:
        return out

    def label_of(node) -> str:
        """Read the ont:hasLabel literal of a skill node (fallback: URI fragment)."""
        for s, p, o in graph.triples((node, ONT.hasLabel, None)):
            if isinstance(o, Literal):
                return str(o)
        return str(node).rsplit("#", 1)[-1].rsplit("/", 1)[-1]

    for _, p, o in graph.triples((URIRef(cand), None, None)):
        if p == ONT.hasSkill:
            out["skills"].append(label_of(o))
        elif p == ONT.heldRole:
            for subj, prop, obj in graph.triples((o, ONT.title, None)):
                out["roles"].append(str(obj))
        elif p == ONT.educatedAt:
            for subj, prop, obj in graph.triples((o, ONT.degreeName, None)):
                out["education"].append(str(obj))
    for subj, p, obj in graph.triples((None, ONT.projectOf, URIRef(cand))):
        for _, _, name_obj in graph.triples((subj, ONT.description, None)):
            out["projects"].append(label_of(subj))
    return out
