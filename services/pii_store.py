"""
PII mapping store — persists token→original mappings in Neo4j.

Each mapping is stored as a :PiiMapping node with properties:
  token     — the masked token, e.g. "[EMAIL:abc123def456]"
  original  — the original PII value
  pii_type  — "email" | "phone" | "candidate"
  source_id — the resume source_id this mapping came from

Lookup is O(1) via a unique index on `token`.
The store is append-only; existing tokens are never overwritten (deterministic UUIDs
guarantee the same token always maps to the same original value).
"""
import logging
from typing import Optional

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

_CREATE_INDEX = """
CREATE INDEX pii_mapping_token IF NOT EXISTS
FOR (p:PiiMapping) ON (p.token)
"""

_UPSERT = """
MERGE (p:PiiMapping {token: $token})
ON CREATE SET p.original  = $original,
              p.pii_type  = $pii_type,
              p.source_id = $source_id
"""

_LOOKUP_MANY = """
UNWIND $tokens AS tok
MATCH (p:PiiMapping {token: tok})
RETURN p.token AS token, p.original AS original
"""


class PiiStore:
    """Thin Neo4j-backed store for PII token↔original mappings."""

    def __init__(self, driver: AsyncDriver):
        self._driver = driver

    async def ensure_index(self) -> None:
        await self._driver.execute_query(_CREATE_INDEX)
        logger.info("PiiMapping index ensured")

    async def save_mappings(
        self,
        mappings: dict[str, str],
        source_id: str,
    ) -> None:
        """Persist token→original entries for one resume."""
        if not mappings:
            return

        records = []
        for token, original in mappings.items():
            if token.startswith("[EMAIL:"):
                pii_type = "email"
            elif token.startswith("[PHONE:"):
                pii_type = "phone"
            elif token.startswith("[CANDIDATE:"):
                pii_type = "candidate"
            else:
                pii_type = "unknown"

            records.append({
                "token": token,
                "original": original,
                "pii_type": pii_type,
                "source_id": source_id,
            })

        for rec in records:
            await self._driver.execute_query(
                _UPSERT,
                parameters_=rec,
            )

        logger.debug("Saved %d PII mappings for source_id=%s", len(records), source_id)

    async def resolve_tokens(self, tokens: list[str]) -> dict[str, str]:
        """
        Resolve a list of tokens to their original PII values.

        Returns only the tokens that were found — missing tokens are omitted.
        """
        if not tokens:
            return {}

        result = await self._driver.execute_query(
            _LOOKUP_MANY,
            parameters_={"tokens": tokens},
        )

        return {
            record["token"]: record["original"]
            for record in result.records
        }

    async def resolve_text(self, text: str) -> str:
        """
        Find all [TYPE:uuid] tokens in text and replace them with original values.

        Used when returning search results to the caller.
        """
        import re
        token_re = re.compile(r"\[(?:CANDIDATE|EMAIL|PHONE):[0-9a-f]{12}\]")
        found_tokens = list(set(token_re.findall(text)))
        if not found_tokens:
            return text

        mapping = await self.resolve_tokens(found_tokens)
        for token, original in mapping.items():
            text = text.replace(token, original)
        return text
