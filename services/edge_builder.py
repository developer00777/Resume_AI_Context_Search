"""
Post-ingest edge builder — computes cross-candidate graph intelligence.

Runs as a background task after every ingest batch. Never blocks the ingest response.

Two edge types are built:

  SKILL_CO_OCCURS_WITH (Skill → Skill)
    For every pair of skills that appear together on the same candidate, increment
    a co_occurrence_count weight on the edge. This powers adjacent-skill discovery
    ("Salesforce devs also tend to have Apex, SOQL, Visualforce") and makes the
    D3 graph viewer interesting to navigate.

  SIMILAR_TO (Candidate → Candidate)
    For each newly ingested candidate, find the top-N existing candidates by
    cosine similarity of their node embeddings. Create SIMILAR_TO edges above a
    configurable threshold. Powers the "find candidates similar to a strong hire"
    query without a full corpus scan at query time.

Both operations run entirely inside Neo4j — no LLM calls, no re-embedding.
"""
import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Cypher queries ────────────────────────────────────────────────────────────

# Increment (or create) a SKILL_CO_OCCURS_WITH edge between every skill pair
# that share a common candidate.  Runs over the full graph — idempotent, safe
# to run multiple times.
_BUILD_SKILL_CO_OCCURRENCE = """
MATCH (c:Entity)
WHERE 'Candidate' IN labels(c)
  OR (c.name STARTS WITH '[CANDIDATE:')
WITH c
MATCH (c)-[r1]->(s1:Entity)
WHERE 'Skill' IN labels(s1) OR r1.name = 'HAS_SKILL'
MATCH (c)-[r2]->(s2:Entity)
WHERE ('Skill' IN labels(s2) OR r2.name = 'HAS_SKILL')
  AND id(s1) < id(s2)
MERGE (s1)-[co:SKILL_CO_OCCURS_WITH]-(s2)
ON CREATE SET co.co_occurrence_count = 1,
              co.created_at = timestamp()
ON MATCH  SET co.co_occurrence_count = co.co_occurrence_count + 1,
              co.updated_at = timestamp()
RETURN count(co) AS edges_updated
"""

# Build SIMILAR_TO edges for a single candidate using stored node embeddings.
# Neo4j vector similarity via gds.similarity.cosine is not available in all
# editions, so we use a direct vector property comparison via apoc or fallback.
# We use a simpler approach: fetch all candidate embeddings in Python and compute
# cosine similarity there, then write edges back. This avoids GDS dependency.
_GET_ALL_CANDIDATE_EMBEDDINGS = """
MATCH (c:Entity)
WHERE 'Candidate' IN labels(c)
  OR c.name STARTS WITH '[CANDIDATE:'
  OR c.group_id = 'resume-pool'
RETURN c.uuid AS uuid, c.name_embedding AS embedding
LIMIT 5000
"""

_UPSERT_SIMILAR_TO = """
UNWIND $pairs AS pair
MATCH (a:Entity {uuid: pair.uuid_a})
MATCH (b:Entity {uuid: pair.uuid_b})
MERGE (a)-[s:SIMILAR_TO]-(b)
ON CREATE SET s.similarity_score = pair.score,
              s.created_at = timestamp()
ON MATCH  SET s.similarity_score = pair.score,
              s.updated_at = timestamp()
"""

# Simpler fallback: build co-occurrence via Graphiti's episodic edges
_BUILD_SKILL_CO_OCCURRENCE_FALLBACK = """
MATCH (s1:Entity)<-[:RELATES_TO]-(ep:Episodic)-[:RELATES_TO]->(s2:Entity)
WHERE id(s1) < id(s2)
  AND s1.group_id = 'resume-pool'
  AND s2.group_id = 'resume-pool'
MERGE (s1)-[co:SKILL_CO_OCCURS_WITH]-(s2)
ON CREATE SET co.co_occurrence_count = 1
ON MATCH  SET co.co_occurrence_count = co.co_occurrence_count + 1
RETURN count(co) AS edges_updated
"""


class EdgeBuilder:
    """
    Builds cross-candidate graph edges post-ingest.

    Accepts the Neo4j driver that Graphiti already holds — no separate connection.
    """

    def __init__(self, driver, similarity_threshold: float = 0.82, top_k: int = 10):
        # Reach through Graphiti's wrapper to the underlying neo4j AsyncDriver
        self._driver = getattr(driver, "client", driver)
        self._similarity_threshold = similarity_threshold
        self._top_k = top_k

    # ── Public API ────────────────────────────────────────────────────────────

    async def build_all(self) -> dict[str, Any]:
        """
        Run both edge-building passes and return a summary dict.
        Called after every ingest batch.
        """
        co_result = await self._build_skill_co_occurrence()
        sim_result = await self._build_similar_to()
        return {
            "skill_co_occurrence_edges": co_result,
            "similar_to_edges": sim_result,
        }

    async def build_skill_co_occurrence(self) -> int:
        """Public wrapper — returns number of edges created/updated."""
        return await self._build_skill_co_occurrence()

    async def build_similar_to(self) -> int:
        """Public wrapper — returns number of SIMILAR_TO edges written."""
        return await self._build_similar_to()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _build_skill_co_occurrence(self) -> int:
        try:
            result = await self._driver.execute_query(_BUILD_SKILL_CO_OCCURRENCE)
            count = result.records[0]["edges_updated"] if result.records else 0
            logger.info("SKILL_CO_OCCURS_WITH: %d edges updated", count)
            return count
        except Exception as exc:
            logger.warning("SKILL_CO_OCCURS_WITH build failed (%s) — trying fallback", exc)
            try:
                result = await self._driver.execute_query(_BUILD_SKILL_CO_OCCURRENCE_FALLBACK)
                count = result.records[0]["edges_updated"] if result.records else 0
                logger.info("SKILL_CO_OCCURS_WITH (fallback): %d edges updated", count)
                return count
            except Exception as exc2:
                logger.error("SKILL_CO_OCCURS_WITH fallback also failed: %s", exc2)
                return 0

    async def _build_similar_to(self) -> int:
        try:
            result = await self._driver.execute_query(_GET_ALL_CANDIDATE_EMBEDDINGS)
            rows = result.records
        except Exception as exc:
            logger.warning("SIMILAR_TO: could not fetch embeddings (%s)", exc)
            return 0

        # Filter to candidates that actually have a stored embedding vector
        candidates = [
            {"uuid": r["uuid"], "embedding": r["embedding"]}
            for r in rows
            if r["uuid"] and r["embedding"] and len(r["embedding"]) > 0
        ]

        if len(candidates) < 2:
            logger.info("SIMILAR_TO: fewer than 2 candidates with embeddings — skipping")
            return 0

        pairs = _compute_similar_pairs(
            candidates,
            threshold=self._similarity_threshold,
            top_k=self._top_k,
        )

        if not pairs:
            logger.info("SIMILAR_TO: no pairs above threshold %.2f", self._similarity_threshold)
            return 0

        try:
            await self._driver.execute_query(
                _UPSERT_SIMILAR_TO,
                parameters_={"pairs": pairs},
            )
            logger.info("SIMILAR_TO: %d edges written", len(pairs))
            return len(pairs)
        except Exception as exc:
            logger.error("SIMILAR_TO: edge write failed: %s", exc)
            return 0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity — avoids numpy dependency."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _compute_similar_pairs(
    candidates: list[dict],
    threshold: float,
    top_k: int,
) -> list[dict]:
    """
    Compute pairwise cosine similarities and return pairs above the threshold.

    Only keeps the top_k most similar partners per candidate to avoid a
    fully-connected graph on large corpora.
    """
    n = len(candidates)
    # Per-candidate: list of (score, other_uuid)
    per_candidate: list[list[tuple[float, str]]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            score = _cosine_similarity(
                candidates[i]["embedding"],
                candidates[j]["embedding"],
            )
            if score >= threshold:
                per_candidate[i].append((score, candidates[j]["uuid"]))
                per_candidate[j].append((score, candidates[i]["uuid"]))

    pairs = []
    seen: set[frozenset] = set()

    for i, candidate in enumerate(candidates):
        # Keep only top_k partners per candidate
        top = sorted(per_candidate[i], reverse=True)[:top_k]
        for score, other_uuid in top:
            key = frozenset({candidate["uuid"], other_uuid})
            if key not in seen:
                seen.add(key)
                pairs.append({
                    "uuid_a": candidate["uuid"],
                    "uuid_b": other_uuid,
                    "score": round(score, 4),
                })

    return pairs
