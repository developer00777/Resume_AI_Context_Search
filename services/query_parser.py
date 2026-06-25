"""
Query parser — extracts structured filters from a natural-language resume search query.

Runs a single lightweight LLM call against the short query string (never the corpus).
Returns structured filters that drive the Cypher pre-filter step in ResumeService.search().

The parser is deliberately forgiving:
- Fields left unrecognised stay None (no filter applied for that dimension).
- A failed LLM call falls back to an empty ParsedQuery so search degrades gracefully
  to pure semantic mode rather than erroring out.
"""
import json
import logging
import re
from typing import Optional

from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a resume-search query parser. Extract structured filters from a recruiter's
natural-language query.

Return ONLY a JSON object with these fields (omit any field you cannot determine):

{
  "location": "<city or region, e.g. Bangalore, Mumbai, Remote>",
  "min_yoe": <integer minimum years of experience, e.g. 3>,
  "max_yoe": <integer maximum years of experience, optional>,
  "must_skills": ["<canonical lowercase skill>", ...],
  "nice_skills": ["<canonical lowercase skill>", ...],
  "role": "<normalised job title, e.g. backend engineer, data scientist>",
  "company_type": "<product | service | startup | enterprise | agency>",
  "industries": ["<industry name, e.g. FinTech, SaaS, Healthcare>"],
  "semantic_query": "<rewritten search phrase stripped of filter terms, for embedding>"
}

Skill normalisation rules:
  JS / JavaScript / ES6 → javascript
  Node / NodeJS         → nodejs
  SF / SFDC / Salesforce CRM → salesforce
  React / ReactJS       → react
  k8s                   → kubernetes
  Postgres / PostgreSQL  → postgresql
  ML / Machine Learning → machine learning

If the full query is purely semantic with no extractable filters, return:
  {"semantic_query": "<original query>"}

Return valid JSON only. No explanation, no markdown fences.
"""


class ParsedQuery:
    """Structured output from the query parser."""

    __slots__ = (
        "location",
        "min_yoe",
        "max_yoe",
        "must_skills",
        "nice_skills",
        "role",
        "company_type",
        "industries",
        "semantic_query",
    )

    def __init__(
        self,
        location: Optional[str] = None,
        min_yoe: Optional[int] = None,
        max_yoe: Optional[int] = None,
        must_skills: Optional[list[str]] = None,
        nice_skills: Optional[list[str]] = None,
        role: Optional[str] = None,
        company_type: Optional[str] = None,
        industries: Optional[list[str]] = None,
        semantic_query: Optional[str] = None,
    ):
        self.location = location
        self.min_yoe = min_yoe
        self.max_yoe = max_yoe
        self.must_skills = must_skills or []
        self.nice_skills = nice_skills or []
        self.role = role
        self.company_type = company_type
        self.industries = industries or []
        self.semantic_query = semantic_query

    def has_filters(self) -> bool:
        """True if at least one structured filter was extracted."""
        return bool(
            self.location
            or self.min_yoe is not None
            or self.max_yoe is not None
            or self.must_skills
            or self.role
            or self.company_type
            or self.industries
        )

    def __repr__(self) -> str:
        parts = []
        if self.location:
            parts.append(f"location={self.location!r}")
        if self.min_yoe is not None:
            parts.append(f"min_yoe={self.min_yoe}")
        if self.must_skills:
            parts.append(f"must_skills={self.must_skills}")
        if self.role:
            parts.append(f"role={self.role!r}")
        if self.semantic_query:
            parts.append(f"semantic_query={self.semantic_query!r}")
        return f"ParsedQuery({', '.join(parts)})"


class QueryParser:
    """
    Wraps an LLM client to parse recruiter queries into structured filters.

    Accepts the same LLM config as ResumeService so no extra credentials are needed.
    Uses the small/fast model (same as Graphiti's extraction tier).
    """

    def __init__(self, llm_client: OpenAIClient):
        self._llm = llm_client

    async def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural-language query into structured filters.

        Falls back to ParsedQuery(semantic_query=query) on any error so the
        calling search pipeline degrades gracefully to pure semantic mode.
        """
        if not query or not query.strip():
            return ParsedQuery()

        try:
            response = await self._llm.client.chat.completions.create(
                model=self._llm.config.model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": query.strip()},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            return _parse_response(raw, query)

        except Exception as exc:
            logger.warning("QueryParser LLM call failed (%s) — falling back to semantic-only", exc)
            return ParsedQuery(semantic_query=query)


def _parse_response(raw: str, original_query: str) -> ParsedQuery:
    """Parse the LLM JSON response into a ParsedQuery, stripping markdown fences."""
    # Strip markdown code fences if the model added them
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("QueryParser: could not parse LLM response as JSON: %r", raw[:200])
        return ParsedQuery(semantic_query=original_query)

    if not isinstance(data, dict):
        return ParsedQuery(semantic_query=original_query)

    def _int_or_none(val) -> Optional[int]:
        try:
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    def _str_list(val) -> list[str]:
        if isinstance(val, list):
            return [str(v).strip().lower() for v in val if v]
        return []

    return ParsedQuery(
        location=data.get("location") or None,
        min_yoe=_int_or_none(data.get("min_yoe")),
        max_yoe=_int_or_none(data.get("max_yoe")),
        must_skills=_str_list(data.get("must_skills")),
        nice_skills=_str_list(data.get("nice_skills")),
        role=data.get("role") or None,
        company_type=data.get("company_type") or None,
        industries=_str_list(data.get("industries")),
        semantic_query=data.get("semantic_query") or original_query,
    )
