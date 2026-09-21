"""cv-knowledge-graph: build and query an RDF knowledge graph from CVs/resumes.

Pipeline stages:

* :mod:`cvkg.document` - load a PDF/text document (with OCR fallback).
* :mod:`cvkg.extract` - LLM (or offline heuristic) extraction into JSON-LD.
* :mod:`cvkg.build_graph` - JSON-LD -> RDF triples (rdflib) with provenance.
* :mod:`cvkg.query` - SPARQL retrieval.
* :mod:`cvkg.rag` - hybrid GraphRAG over the graph.

Usage::

    from cvkg.document import load_pdf
    from cvkg.extract import OpenAIExtractor
    from cvkg.build_graph import build_graph

    text = load_pdf(Path("resume.pdf"))
    data = OpenAIExtractor().extract(text)      # or HeuristicExtractor()
    graph = build_graph(data, source="resume.pdf", extractor_model="gpt-4o-mini")
"""

from .build_graph import (
    KG,
    ONT,
    build_graph,
    merge_graphs,
    save,
)
from .document import load_pdf
from .extract import (
    EXTRACTION_PROMPT,
    Extractor,
    HeuristicExtractor,
    OpenAIExtractor,
)
from .query import (
    CANDIDATE_STRENGTH_QUERY,
    CANDIDATES_QUERY,
    SKILLS_QUERY,
    candidate_uri,
    candidates,
    query,
    skills_of,
)
from .rag import GraphRAG

__all__ = [
    "EXTRACTION_PROMPT",
    "Extractor",
    "HeuristicExtractor",
    "OpenAIExtractor",
    "KG",
    "ONT",
    "build_graph",
    "merge_graphs",
    "save",
    "load_pdf",
    "CANDIDATES_QUERY",
    "SKILLS_QUERY",
    "CANDIDATE_STRENGTH_QUERY",
    "candidates",
    "candidate_uri",
    "skills_of",
    "query",
    "GraphRAG",
]

__version__ = "0.1.0"
