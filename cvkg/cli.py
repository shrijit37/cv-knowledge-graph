"""Command-line entry point for the cv-knowledge-graph pipeline.

Examples
--------
Build a graph from a PDF with the LLM extractor::

    python -m cvkg.cli build resume.pdf -o graph.ttl

Build with the offline heuristic extractor (no API)::

    python -m cvkg.cli build resume.pdf --heuristic -o graph.ttl

Ask a SPARQL-style question with GraphRAG::

    python -m cvkg.cli ask graph.ttl "who has built a resume parser?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .build_graph import ONT, build_graph, merge_graphs, save
from .document import load_pdf
from .extract import HeuristicExtractor, OpenAIExtractor
from .query import candidates, skills_of
from .rag import GraphRAG


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cvkg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="extract from a document and build an RDF graph")
    b.add_argument("document", type=Path, help="PDF or plain-text resume")
    b.add_argument("-o", "--output", default="graph.ttl", help="output .ttl path")
    b.add_argument("--heuristic", action="store_true", help="use the offline heuristic extractor")
    b.add_argument("--model", default=None, help="override the LLM model name")

    a = sub.add_parser("ask", help="GraphRAG question over a graph")
    a.add_argument("graph", type=Path, help="Turtle file(s) to load (comma-separated)")
    a.add_argument("question", help="natural-language question")
    a.add_argument("--candidate", default=None, help="restrict to a candidate name")
    a.add_argument("--top-k", type=int, default=8)

    s = sub.add_parser("skills", help="list skills of a candidate")
    s.add_argument("graph", type=Path)
    s.add_argument("candidate", help="candidate full name")

    args = parser.parse_args(argv)

    if args.command == "build":
        return _build(args)
    if args.command == "ask":
        return _ask(args)
    if args.command == "skills":
        return _skills(args)
    return 2


def _build(args: argparse.Namespace) -> int:
    text = load_pdf(args.document)
    if not text.strip():
        print(f"error: no text could be extracted from {args.document}", file=sys.stderr)
        return 1
    extractor = HeuristicExtractor() if args.heuristic else OpenAIExtractor(model=args.model or "")
    data = extractor.extract(text)
    graph = build_graph(
        data,
        source=str(args.document),
        extractor_model=type(extractor).__name__,
    )
    out = save(graph, args.output)
    print(f"wrote {out} ({len(graph)} triples)")
    return 0


def _ask(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in str(args.graph).split(",") if p]
    graph = merge_graphs(paths)
    rag = GraphRAG(graph, top_k=args.top_k)
    items = rag.retrieve(args.question, candidate=args.candidate)
    print(rag.format_context(items))
    return 0


def _skills(args: argparse.Namespace) -> int:
    graph = merge_graphs([args.graph])
    names = [r["name"] for r in candidates(graph) if r.get("name")]
    print(json.dumps({n: skills_of(graph, n) for n in names}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
