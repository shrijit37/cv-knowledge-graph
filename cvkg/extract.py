"""LLM-based knowledge extraction from resume/CV text into JSON-LD.

The extraction layer is behind a small interface so the underlying model can be
swapped without touching the rest of the pipeline:

* :class:`OpenAIExtractor` - calls an OpenAI-compatible chat-completions endpoint
  (works with the OpenAI API, Azure OpenAI, or any compatible local server) using
  whichever client is installed (``openai``, or ``requests`` for a raw HTTP call
  when the SDK is unavailable).
* :class:`HeuristicExtractor` - a dependency-free fallback that extracts skills
  from a small dictionary and records roles/projects as opaque blocks. It exists
  so the pipeline can be exercised without any network or API key, and so tests
  run offline.

Both return the same JSON-LD document shape consumed by
:mod:`cvkg.build_graph`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

EXTRACTION_PROMPT = """\
You are a knowledge-extraction engine. Given the text of a resume or CV, extract
the information below and return ONLY a JSON object (no markdown, no prose) with
exactly this shape:

{
  "candidate": {"fullName": string, "email": string},
  "skills": [{"label": string, "proficiency": string}],
  "roles": [{"title": string, "employer": string, "startDate": "YYYY-MM"|null,
             "endDate": "YYYY-MM"|null, "summary": string}],
  "projects": [{"name": string, "role": string, "technologies": [string],
                "description": string, "url": string|null}],
  "education": [{"degree": string, "field": string, "institution": string,
                 "graduatedAt": "YYYY"|null}]
}

Rules:
- Only extract facts present in the text. Never invent names, dates, or skills.
- If a field is unknown use null (or an empty array for lists).
- Keep dates in YYYY-MM or YYYY form.
- Respond with the JSON object only.
"""


class Extractor(Protocol):
    """Interface implemented by every extraction backend."""

    def extract(self, text: str) -> dict[str, Any]:
        """Return the JSON-LD document for ``text`` (see prompt shape)."""
        ...


def _clean_json(raw: str) -> str:
    """Strip markdown fences or surrounding prose and return the JSON payload."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw


@dataclass
class OpenAIExtractor:
    """Extract via an OpenAI-compatible chat-completions endpoint.

    Configure with ``OPENAI_API_KEY`` (and optionally ``OPENAI_BASE_URL`` /
    ``OPENAI_MODEL``) or pass ``api_key``/``url``/``model`` explicitly. Uses the
    ``openai`` SDK when installed, otherwise falls back to ``requests``.
    """

    url: str = ""
    model: str = ""
    api_key: str = ""
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAIExtractor requires an API key: set OPENAI_API_KEY or pass api_key."
            )
        default_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.url = self.url or f"{default_url.rstrip('/')}/chat/completions"
        self.model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def extract(self, text: str) -> dict[str, Any]:
        try:
            return self._extract_with_openai_sdk(text)
        except ImportError:
            return self._extract_with_requests(text)

    # -- openai SDK path ------------------------------------------------
    def _extract_with_openai_sdk(self, text: str) -> dict[str, Any]:
        # imported lazily so the rest of the project works without the SDK
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=self.api_key, base_url=self.url.rsplit("/chat/completions", 1)[0])
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": text[:30000]},
            ],
        )
        return self._parse(resp.choices[0].message.content or "")

    # -- raw HTTP path (no openai SDK) ----------------------------------
    def _extract_with_requests(self, text: str) -> dict[str, Any]:
        import requests  # type: ignore

        resp = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": text[:30000]},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return self._parse(resp.json()["choices"][0]["message"]["content"])

    def _parse(self, raw: str) -> dict[str, Any]:
        data = json.loads(_clean_json(raw))
        if not isinstance(data, dict):
            raise ValueError("Extractor returned a non-object payload.")
        return data


@dataclass
class HeuristicExtractor:
    """Offline, dependency-free extraction used for tests and quick demos.

    Identifies skills from a built-in dictionary and records each top-level
    resume section as an opaque block, so the pipeline (extract -> build graph
    -> query) can be demonstrated with zero network access.
    """

    skill_bank: tuple[str, ...] = (
        "Python", "TypeScript", "JavaScript", "React", "Node.js", "Express",
        "Next.js", "RDF", "OWL", "SPARQL", "GraphDB", "Jena", "Neptune",
        "rdflib", "GraphRAG", "PostgreSQL", "MongoDB", "Redis", "Docker",
        "AWS", "Git", "OpenAI", "LLM", "FastAPI", "Tailwind",
    )

    def extract(self, text: str) -> dict[str, Any]:
        found = [s for s in self.skill_bank if re.search(rf"\b{re.escape(s)}\b", text, re.I)]
        return {
            "candidate": self._candidate(text),
            "skills": [{"label": s, "proficiency": "mentioned"} for s in found],
            "roles": self._roles(text),
            "projects": [],
            "education": self._education(text),
        }

    def _candidate(self, text: str) -> dict[str, Any]:
        name = None
        m = re.search(r"(?im)^([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)+)\s*$", text)
        if m:
            name = m.group(1).strip()
        email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
        return {"fullName": name, "email": email.group(0) if email else None}

    def _roles(self, text: str) -> list[dict[str, Any]]:
        roles = []
        for m in re.finditer(r"(?im)^(.+?)\s*[-|]\s*(.+)$", text):
            roles.append({"title": m.group(1).strip(), "employer": m.group(2).strip(),
                          "startDate": None, "endDate": None, "summary": ""})
        return roles

    def _education(self, text: str) -> list[dict[str, Any]]:
        edu = []
        m = re.search(r"(?i)((B\.?Tech|M\.?Tech|B\.?Sc|M\.?Sc|MBA|PhD)[^,\n]*)", text)
        if m:
            edu.append({"degree": m.group(1).strip(), "field": m.group(0).strip() or None,
                        "institution": None, "graduatedAt": None})
        return edu
