# cv-knowledge-graph

Build an **RDF knowledge graph from CVs/resumes** using LLM-based extraction,
then retrieve from it with **SPARQL** and a **hybrid GraphRAG** layer.

This is a compact, end-to-end implementation of a knowledge-engineering
pipeline: unstructured documents (PDF or text) -> structured entity/relation
extraction -> an OWL-ontologized RDF graph with provenance -> graph-native
retrieval (SPARQL) and grounded graph-aware retrieval (GraphRAG). It is the
kind of pipeline used to turn technical documents into queryable knowledge,
but scoped to resumes so it stays small and fully runnable.

```
┌────────────┐     ┌────────────┐     ┌─────────────┐     ┌───────────────┐
│ PDF / text │ ──► │ extract.py │ ──► │ build_graph │ ──► │ graph.ttl     │
│ (documents)│     │  (LLM)     │     │  (rdflib)   │     │  (Turtle/RDF) │
└────────────┘     └────────────┘     └─────────────┘     └───────┬───────┘
                                                                  │
                                      ┌───────────────────────────┼───────────┐
                                      ▼                           ▼           ▼
                               ┌────────────┐              ┌──────────┐ ┌──────────┐
                               │ query.py   │  SPARQL     │ rag.py   │ │ Fuseki /  │
                               │ (SELECT/   │ ──────────► │ GraphRAG │ │ GraphDB / │
                               │  ASK)      │  + reasoning│ embeddings│ │ Neptune   │
                               └────────────┘              └──────────┘ └──────────┘
```

## Features

- **OWL ontology** (`ontology.ttl`): `Candidate`, `Skill`, `Role`, `Project`,
  `Education`, `Employer`, `Evidence`, with object/data properties and timing
  constraints.
- **LLM extraction** (`cvkg/extract.py`): pluggable — OpenAI-compatible API
  (OpenAI, Azure OpenAI, or any local server) or a dependency-free offline
  `HeuristicExtractor` for demos/tests.
- **Document handling** (`cvkg/document.py`): PDF text extraction with an
  optional OCR fallback (tesseract) for scanned files.
- **Graph construction** (`cvkg/build_graph.py`): JSON-LD -> RDF triples via
  `rdflib`, with **provenance** (source file, extraction timestamp, model)
  attached to every statement.
- **SPARQL retrieval** (`cvkg/query.py`): graph-native queries
  (candidates, skills, skill-by-project, candidate strength).
- **GraphRAG** (`cvkg/rag.py`): hybrid retrieval — SPARQL grounding +
  embedding ranking + lexical rerank, returning graph-grounded context for an
  LLM to answer from.
- **Graph repository** (`compose.yaml`): optional Jena Fuseki / GraphDB /
  Neptune deployment with a SPARQL endpoint.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai,rag]"     # rag pulls sentence-transformers
```

## Quick start

Build a graph from a sample resume using the offline extractor:

```bash
python -m cvkg.cli build samples/sample_resume.txt --heuristic -o graph.ttl
# wrote graph.ttl (63 triples)
```

Build with an LLM (uses `OPENAI_API_KEY`, or `OPENAI_BASE_URL`/`OPENAI_MODEL`
for Azure/local endpoints):

```bash
python -m cvkg.cli build resume.pdf -o graph.ttl
```

Ask a grounded GraphRAG question:

```bash
python -m cvkg.cli ask graph.ttl "who has RDF experience?"
# - [Shrijit Srivastav] Skill: RDF  (score ...)
```

List skills:

```bash
python -m cvkg.cli skills graph.ttl "Shrijit Srivastav"
```

Run SPARQL in Python:

```python
from rdflib import Graph
from cvkg.query import CANDIDATES_QUERY, query

g = Graph().parse("graph.ttl", format="turtle")
print(query(g, CANDIDATES_QUERY))
```

## Load into a graph repository

```bash
docker compose up -d          # Jena Fuseki on :3030
docker compose run --rm load  # uploads graph.ttl into the cvkg dataset
```

Then run any query at `http://localhost:3030/#/dataset/cvkg/query`. The same
`.ttl` loads into GraphDB, RDFox, Neptune, or any RDF store.

## Tests

```bash
pytest -q
```

## Provenance / evidence linking

Every extracted statement links to an `Evidence` node recording the source
file, extraction timestamp, and model (`cvkg/build_graph.py`), so any fact in
the graph is traceable back to a source document — meeting the "traceability
to source documents" requirement for knowledge-graph ingestion.

## License

MIT — see [LICENSE](LICENSE).
