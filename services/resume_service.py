"""
Resume Intelligence service — Graphiti wrapper for resume ingestion and search.

Two-tier LLM architecture:
  - LLM (extraction/query)  — configured via LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
  - Embedder                — configured via EMBEDDING_BASE_URL / EMBEDDING_MODEL / EMBEDDING_API_KEY

Both tiers are independently switchable between OpenRouter and local Ollama
purely via Railway environment variables. No code change required.

PII safety:
  Before any text reaches the LLM, PiiMasker strips name / email / phone and
  replaces them with deterministic tokens. The token→original mapping is
  persisted in Neo4j (PiiStore) and re-hydrated in search results before
  they are returned to the caller.

Search pipeline (Phase 1 upgrade):
  query → QueryParser (LLM) → ParsedQuery
        → Cypher pre-filter (location / YoE / must-skills)
        → Graphiti semantic search scoped to filtered candidate IDs
        → cross-encoder rerank
        → PII re-hydration

Post-ingest graph intelligence (Phase 2 upgrade):
  after each ingest batch → EdgeBuilder.build_all() (background task)
        → SKILL_CO_OCCURS_WITH edges (weighted co-occurrence)
        → SIMILAR_TO edges (cosine similarity between candidate embeddings)
"""
import asyncio
import logging
from datetime import date, datetime, timezone
from typing import Any, Optional

from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from config.entity_types import ENTITY_TYPES
from config.edge_types import EDGE_TYPES, EDGE_TYPE_MAP
from models.resume import Resume
from services.pii_masker import mask_pii
from services.pii_store import PiiStore
from services.query_parser import QueryParser
from services.edge_builder import EdgeBuilder

logger = logging.getLogger(__name__)

# All resumes share one group_id so cross-candidate graph queries work
RESUME_GROUP_ID = "resume-pool"


def current_years_experience(
    earliest_role_start: date | None,
    captured_yoe: float | None = None,
    parsed_on: date | None = None,
    today: date | None = None,
) -> float | None:
    """Always-current YoE anchored on earliest role start or rolled forward from parse date."""
    today = today or date.today()
    if earliest_role_start:
        return round((today - earliest_role_start).days / 365.25, 1)
    if captured_yoe is not None and parsed_on:
        drift = (today - parsed_on).days / 365.25
        return round(captured_yoe + drift, 1)
    return None


class ResumeService:
    """
    Graphiti wrapper for resume knowledge graph operations.

    Ingestion pipeline (per resume):
      raw text → PiiMasker → masked episode → Graphiti (LLM extraction + embed) → Neo4j

    Search pipeline:
      query → Graphiti hybrid search → raw results → PiiStore re-hydration → caller
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        # LLM tier (extraction + query rewriting)
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str,
        # Embedding tier (independently configurable)
        embedding_model: str,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
    ):
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._llm_api_key = llm_api_key
        self._llm_base_url = llm_base_url
        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._embedding_api_key = embedding_api_key or llm_api_key
        self._embedding_base_url = embedding_base_url or llm_base_url
        self.client: Optional[Graphiti] = None
        self._pii_store: Optional[PiiStore] = None
        self._query_parser: Optional[QueryParser] = None
        self._edge_builder: Optional[EdgeBuilder] = None

    # ── Connection ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        if not self._llm_api_key:
            raise RuntimeError("LLM_API_KEY is required.")

        llm_client = OpenAIClient(
            config=LLMConfig(
                api_key=self._llm_api_key,
                base_url=self._llm_base_url,
                model=self._llm_model,
                small_model=self._llm_model,
            )
        )

        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=self._embedding_api_key,
                base_url=self._embedding_base_url,
                embedding_model=self._embedding_model,
            )
        )

        # Cross-encoder must receive the same LLM config explicitly — otherwise
        # Graphiti 0.29+ falls back to reading OPENAI_API_KEY from the environment.
        cross_encoder = OpenAIRerankerClient(
            config=LLMConfig(
                api_key=self._llm_api_key,
                base_url=self._llm_base_url,
                model=self._llm_model,
            )
        )

        self.client = Graphiti(
            uri=self._neo4j_uri,
            user=self._neo4j_user,
            password=self._neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

        await self.client.build_indices_and_constraints()

        # PiiStore uses the same Neo4j driver that Graphiti already opened
        self._pii_store = PiiStore(self.client.driver)
        await self._pii_store.ensure_index()

        # QueryParser reuses the same LLM client — no extra credentials needed
        self._query_parser = QueryParser(llm_client)

        # EdgeBuilder reuses the same Neo4j driver
        self._edge_builder = EdgeBuilder(self.client.driver)

        logger.info(
            "ResumeService connected | LLM: %s @ %s | Embedder: %s @ %s",
            self._llm_model, self._llm_base_url,
            self._embedding_model, self._embedding_base_url,
        )

    async def disconnect(self) -> None:
        if self.client:
            await self.client.close()
            self.client = None
            self._pii_store = None
            self._query_parser = None
            self._edge_builder = None

    # ── Ingestion ──────────────────────────────────────────────────────────────

    async def ingest_resume(self, resume: Resume) -> str:
        """
        Ingest a single resume.

        1. Masks PII (name / email / phone) → deterministic tokens.
        2. Persists token→original mappings in Neo4j PiiStore.
        3. Sends masked episode to Graphiti (LLM extraction + embedding).

        Returns the candidate_uuid assigned during masking.
        """
        if not self.client or not self._pii_store:
            raise RuntimeError("Not connected. Call connect() first.")

        mask = mask_pii(resume.full_text, resume.source_id)

        await self._pii_store.save_mappings(mask.mappings, resume.source_id)

        masked_resume = Resume(
            source_id=resume.source_id,
            full_text=mask.masked_text,
            candidate_name=f"[CANDIDATE:{mask.candidate_uuid}]" if mask.candidate_uuid else None,
            parsed_on=resume.parsed_on,
        )

        await self.client.add_episode(
            name=f"Resume: [CANDIDATE:{mask.candidate_uuid}] (src={resume.source_id})",
            episode_body=masked_resume.to_episode_content(),
            source=EpisodeType.message,
            source_description=f"resume (source_id={resume.source_id})",
            reference_time=datetime.now(timezone.utc),
            group_id=RESUME_GROUP_ID,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )

        # Build cross-candidate edges in the background — does not block response
        if self._edge_builder:
            asyncio.create_task(self._run_edge_builder())

        return mask.candidate_uuid or resume.source_id

    async def ingest_resumes_bulk(self, resumes: list[Resume]) -> list[str]:
        """
        Bulk ingest multiple resumes with PII masking applied to each.

        Returns a list of candidate_uuids in the same order as the input.
        """
        if not self.client or not self._pii_store:
            raise RuntimeError("Not connected. Call connect() first.")
        if not resumes:
            return []

        now = datetime.now(timezone.utc)
        raw_episodes: list[RawEpisode] = []
        candidate_uuids: list[str] = []

        for resume in resumes:
            mask = mask_pii(resume.full_text, resume.source_id)
            await self._pii_store.save_mappings(mask.mappings, resume.source_id)

            uid = mask.candidate_uuid or resume.source_id
            candidate_uuids.append(uid)

            masked_resume = Resume(
                source_id=resume.source_id,
                full_text=mask.masked_text,
                candidate_name=f"[CANDIDATE:{uid}]",
                parsed_on=resume.parsed_on,
            )

            raw_episodes.append(RawEpisode(
                name=f"Resume: [CANDIDATE:{uid}] (src={resume.source_id})",
                content=masked_resume.to_episode_content(),
                source=EpisodeType.message,
                source_description=f"resume (source_id={resume.source_id})",
                reference_time=now,
            ))

        await self.client.add_episode_bulk(
            raw_episodes,
            group_id=RESUME_GROUP_ID,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )

        logger.info("Bulk ingested %d resumes (PII masked)", len(resumes))

        # Build cross-candidate edges in the background — does not block response
        if self._edge_builder:
            asyncio.create_task(self._run_edge_builder())

        return candidate_uuids

    # ── Search ─────────────────────────────────────────────────────────────────

    async def search(self, query: str, num_results: int = 50) -> dict[str, Any]:
        """
        Hybrid semantic + keyword search with Cypher pre-filter.

        Pipeline:
          1. QueryParser extracts structured filters (location, YoE, must-skills, role).
          2. Cypher pre-filter narrows the candidate pool to those matching hard constraints.
          3. Graphiti semantic search runs within the filtered set.
          4. PII tokens are re-hydrated before returning results.

        Falls back to pure semantic search if query parsing fails or yields no filters.
        """
        if not self.client or not self._pii_store:
            raise RuntimeError("Not connected. Call connect() first.")

        # Step 1: parse query into structured filters
        parsed = None
        if self._query_parser:
            parsed = await self._query_parser.parse(query)
            logger.info("QueryParser result: %s", parsed)

        # Step 2: Cypher pre-filter — get candidate UUIDs matching hard constraints
        filtered_uuids: list[str] | None = None
        if parsed and parsed.has_filters():
            filtered_uuids = await self._cypher_prefilter(parsed)
            logger.info(
                "Cypher pre-filter returned %d candidates for query %r",
                len(filtered_uuids) if filtered_uuids is not None else -1,
                query,
            )
            # If the filter returned zero results, drop it and fall back to full semantic
            if filtered_uuids is not None and len(filtered_uuids) == 0:
                logger.info("Pre-filter yielded 0 results — falling back to full semantic search")
                filtered_uuids = None

        # Step 3: semantic search — use the rewritten query text for better embedding
        semantic_query = (parsed.semantic_query if parsed and parsed.semantic_query else query)
        config = COMBINED_HYBRID_SEARCH_RRF.model_copy(update={"limit": num_results})
        results = await self.client._search(
            query=semantic_query,
            group_ids=[RESUME_GROUP_ID],
            config=config,
        )

        nodes = [self._node_to_dict(n) for n in results.nodes]
        edges = [self._edge_to_dict(e) for e in results.edges]

        # Step 4: if we have a Cypher-filtered candidate set, narrow the node results
        if filtered_uuids is not None:
            filtered_set = set(filtered_uuids)
            nodes = [n for n in nodes if n["uuid"] in filtered_set or not _is_candidate_node(n)]

        # Step 5: re-hydrate PII tokens
        nodes = await self._rehydrate_nodes(nodes)
        edges = await self._rehydrate_edges(edges)

        return {
            "nodes": nodes,
            "edges": edges,
            "parsed_filters": _parsed_to_dict(parsed) if parsed else None,
        }

    async def _cypher_prefilter(self, parsed) -> list[str] | None:
        """
        Run structured Cypher constraints against the graph and return matching candidate UUIDs.

        Each constraint is optional — only clauses for populated filter fields are added.
        Returns None if the query fails (caller treats this as no filter).
        """
        if not self.client:
            return None

        driver = getattr(self.client.driver, "client", self.client.driver)
        conditions = []
        params: dict[str, Any] = {}

        # Location filter — match candidates linked to a Location node
        if parsed.location:
            conditions.append(
                "EXISTS { MATCH (c)-[:LOCATED_IN]->(loc:Entity) "
                "WHERE toLower(loc.name) CONTAINS toLower($location) }"
            )
            params["location"] = parsed.location

        # YoE filter — uses earliest_role_start stored as node property by Graphiti
        if parsed.min_yoe is not None:
            conditions.append("c.earliest_role_start IS NOT NULL")
            conditions.append(
                "duration.between(date(c.earliest_role_start), date()).years >= $min_yoe"
            )
            params["min_yoe"] = parsed.min_yoe

        if parsed.max_yoe is not None:
            conditions.append("c.earliest_role_start IS NOT NULL")
            conditions.append(
                "duration.between(date(c.earliest_role_start), date()).years <= $max_yoe"
            )
            params["max_yoe"] = parsed.max_yoe

        # Must-skills filter — candidate must have ALL listed skills
        for i, skill in enumerate(parsed.must_skills):
            key = f"skill_{i}"
            conditions.append(
                f"EXISTS {{ MATCH (c)-[:HAS_SKILL]->(s:Entity) "
                f"WHERE toLower(s.name) CONTAINS toLower(${key}) }}"
            )
            params[key] = skill

        # Role filter
        if parsed.role:
            conditions.append(
                "EXISTS { MATCH (c)-[:WORKED_AS]->(r:Entity) "
                "WHERE toLower(r.name) CONTAINS toLower($role) }"
            )
            params["role"] = parsed.role

        # Company type filter
        if parsed.company_type:
            conditions.append(
                "EXISTS { MATCH (c)-[:WORKED_AT]->(co:Entity) "
                "WHERE toLower(co.company_type) = toLower($company_type) }"
            )
            params["company_type"] = parsed.company_type

        # Industries filter (match any listed industry)
        if parsed.industries:
            conditions.append(
                "EXISTS { MATCH (c)-[:WORKED_AT]->(co:Entity) "
                "WHERE any(ind IN $industries WHERE toLower(co.industry) CONTAINS toLower(ind)) }"
            )
            params["industries"] = parsed.industries

        if not conditions:
            return None

        where_clause = " AND ".join(conditions)
        cypher = f"""
            MATCH (c:Entity)
            WHERE c.group_id = 'resume-pool'
              AND ({where_clause})
            RETURN c.uuid AS uuid
            LIMIT 2000
        """

        try:
            result = await driver.execute_query(cypher, parameters_=params)
            return [r["uuid"] for r in result.records if r["uuid"]]
        except Exception as exc:
            logger.warning("Cypher pre-filter failed (%s) — falling back to full semantic", exc)
            return None

    async def get_candidate(self, candidate_name: str) -> dict[str, Any]:
        return await self.search(
            query=f"Everything about candidate {candidate_name}: skills, roles, companies, education",
            num_results=100,
        )

    async def find_similar_candidates(self, candidate_name: str, num_results: int = 20) -> dict[str, Any]:
        """
        Find candidates similar to a given candidate.

        Tries the SIMILAR_TO graph edge first (set by EdgeBuilder post-ingest).
        Falls back to semantic search if no edges exist yet.
        """
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        driver = getattr(self.client.driver, "client", self.client.driver)

        cypher = """
            MATCH (anchor:Entity {group_id: 'resume-pool'})
            WHERE toLower(anchor.name) CONTAINS toLower($name)
            WITH anchor LIMIT 1
            MATCH (anchor)-[s:SIMILAR_TO]-(other:Entity)
            RETURN other.uuid AS uuid, other.name AS name, s.similarity_score AS score
            ORDER BY score DESC
            LIMIT $limit
        """
        try:
            result = await driver.execute_query(
                cypher,
                parameters_={"name": candidate_name, "limit": num_results},
            )
            if result.records:
                similar = [
                    {"uuid": r["uuid"], "name": r["name"], "similarity_score": r["score"]}
                    for r in result.records
                ]
                return {"similar_candidates": similar, "source": "graph_edges"}
        except Exception as exc:
            logger.warning("SIMILAR_TO graph query failed (%s) — falling back to semantic", exc)

        # Fallback: semantic search
        results = await self.search(
            query=f"Candidates with similar background and skills to {candidate_name}",
            num_results=num_results,
        )
        results["source"] = "semantic_fallback"
        return results

    async def get_skill_pool(self, skill: str) -> dict[str, Any]:
        """
        Get all candidates with a skill and their adjacent (co-occurring) skills.

        Uses SKILL_CO_OCCURS_WITH edges built by EdgeBuilder for adjacent skills.
        Falls back to semantic search for the candidate list if Cypher finds nothing.
        """
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        driver = getattr(self.client.driver, "client", self.client.driver)

        # Adjacent skills via co-occurrence graph
        adjacent_cypher = """
            MATCH (s:Entity)-[co:SKILL_CO_OCCURS_WITH]-(adj:Entity)
            WHERE toLower(s.name) CONTAINS toLower($skill)
            RETURN adj.name AS adjacent_skill, co.co_occurrence_count AS count
            ORDER BY count DESC
            LIMIT 20
        """
        adjacent_skills = []
        try:
            result = await driver.execute_query(
                adjacent_cypher, parameters_={"skill": skill}
            )
            adjacent_skills = [
                {"skill": r["adjacent_skill"], "co_occurrence_count": r["count"]}
                for r in result.records
            ]
        except Exception as exc:
            logger.warning("Adjacent skill query failed: %s", exc)

        # Candidates with this skill
        results = await self.search(
            query=f"Candidates who have {skill} skill, their experience and related skills",
            num_results=100,
        )
        results["adjacent_skills"] = adjacent_skills
        return results

    async def get_pool_intelligence(self) -> dict[str, Any]:
        """
        Aggregate analytics over the full candidate pool via direct Cypher.

        Returns deterministic counts and distributions rather than semantic search guesses.
        Falls back to semantic search if Cypher queries fail.
        """
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        driver = getattr(self.client.driver, "client", self.client.driver)
        intelligence: dict[str, Any] = {}

        # Top skills by candidate count
        try:
            result = await driver.execute_query("""
                MATCH (c:Entity {group_id: 'resume-pool'})-[:HAS_SKILL]->(s:Entity)
                RETURN s.name AS skill, count(DISTINCT c) AS candidate_count
                ORDER BY candidate_count DESC
                LIMIT 30
            """)
            intelligence["top_skills"] = [
                {"skill": r["skill"], "candidate_count": r["candidate_count"]}
                for r in result.records
            ]
        except Exception as exc:
            logger.warning("Top skills query failed: %s", exc)

        # Thin skills (supply gaps — fewest candidates)
        try:
            result = await driver.execute_query("""
                MATCH (c:Entity {group_id: 'resume-pool'})-[:HAS_SKILL]->(s:Entity)
                RETURN s.name AS skill, count(DISTINCT c) AS candidate_count
                ORDER BY candidate_count ASC
                LIMIT 15
            """)
            intelligence["thin_skills"] = [
                {"skill": r["skill"], "candidate_count": r["candidate_count"]}
                for r in result.records
            ]
        except Exception as exc:
            logger.warning("Thin skills query failed: %s", exc)

        # Top locations
        try:
            result = await driver.execute_query("""
                MATCH (c:Entity {group_id: 'resume-pool'})-[:LOCATED_IN]->(loc:Entity)
                RETURN loc.name AS location, count(DISTINCT c) AS candidate_count
                ORDER BY candidate_count DESC
                LIMIT 20
            """)
            intelligence["top_locations"] = [
                {"location": r["location"], "candidate_count": r["candidate_count"]}
                for r in result.records
            ]
        except Exception as exc:
            logger.warning("Top locations query failed: %s", exc)

        # Top companies
        try:
            result = await driver.execute_query("""
                MATCH (c:Entity {group_id: 'resume-pool'})-[:WORKED_AT]->(co:Entity)
                RETURN co.name AS company, count(DISTINCT c) AS candidate_count
                ORDER BY candidate_count DESC
                LIMIT 20
            """)
            intelligence["top_companies"] = [
                {"company": r["company"], "candidate_count": r["candidate_count"]}
                for r in result.records
            ]
        except Exception as exc:
            logger.warning("Top companies query failed: %s", exc)

        # Total candidate count
        try:
            result = await driver.execute_query("""
                MATCH (c:Entity {group_id: 'resume-pool'})
                WHERE 'Candidate' IN labels(c) OR c.name STARTS WITH '[CANDIDATE:'
                RETURN count(c) AS total
            """)
            intelligence["total_candidates"] = result.records[0]["total"] if result.records else 0
        except Exception as exc:
            logger.warning("Total candidate count query failed: %s", exc)

        # If all Cypher queries failed, fall back to semantic
        if not intelligence:
            logger.warning("All pool intelligence Cypher queries failed — falling back to semantic")
            return await self._search_raw(
                query="Overview of all candidate skills, locations, companies, roles, and experience levels in the pool",
                num_results=200,
            )

        return intelligence

    async def _search_raw(self, query: str, num_results: int) -> dict[str, Any]:
        """Direct semantic search without query parsing — used as analytics fallback."""
        config = COMBINED_HYBRID_SEARCH_RRF.model_copy(update={"limit": num_results})
        results = await self.client._search(
            query=query,
            group_ids=[RESUME_GROUP_ID],
            config=config,
        )
        nodes = [self._node_to_dict(n) for n in results.nodes]
        edges = [self._edge_to_dict(e) for e in results.edges]
        nodes = await self._rehydrate_nodes(nodes)
        edges = await self._rehydrate_edges(edges)
        return {"nodes": nodes, "edges": edges}

    async def get_graph_for_visualization(self) -> dict[str, Any]:
        results = await self.get_pool_intelligence()
        return self._format_for_visualization(results)

    # ── PII re-hydration helpers ────────────────────────────────────────────────

    async def _rehydrate_nodes(self, nodes: list[dict]) -> list[dict]:
        for node in nodes:
            if node.get("name"):
                node["name"] = await self._pii_store.resolve_text(node["name"])
            if node.get("summary"):
                node["summary"] = await self._pii_store.resolve_text(node["summary"])
        return nodes

    async def _rehydrate_edges(self, edges: list[dict]) -> list[dict]:
        for edge in edges:
            if edge.get("fact"):
                edge["fact"] = await self._pii_store.resolve_text(edge["fact"])
        return edges

    # ── Serialisation helpers ───────────────────────────────────────────────────

    def _node_to_dict(self, node) -> dict[str, Any]:
        return {
            "uuid": node.uuid,
            "name": node.name,
            "labels": node.labels if hasattr(node, "labels") else [],
            "summary": node.summary if hasattr(node, "summary") else None,
            "attributes": node.attributes if hasattr(node, "attributes") else {},
            "created_at": node.created_at.isoformat() if hasattr(node, "created_at") and node.created_at else None,
        }

    def _edge_to_dict(self, edge) -> dict[str, Any]:
        return {
            "uuid": edge.uuid,
            "name": edge.name if hasattr(edge, "name") else None,
            "fact": edge.fact if hasattr(edge, "fact") else None,
            "source_node_uuid": edge.source_node_uuid if hasattr(edge, "source_node_uuid") else None,
            "target_node_uuid": edge.target_node_uuid if hasattr(edge, "target_node_uuid") else None,
            "valid_at": edge.valid_at.isoformat() if hasattr(edge, "valid_at") and edge.valid_at else None,
            "invalid_at": edge.invalid_at.isoformat() if hasattr(edge, "invalid_at") and edge.invalid_at else None,
        }

    async def _run_edge_builder(self) -> None:
        """Background task: build cross-candidate edges after ingest. Never raises."""
        try:
            summary = await self._edge_builder.build_all()
            logger.info("EdgeBuilder complete: %s", summary)
        except Exception as exc:
            logger.error("EdgeBuilder background task failed: %s", exc)

    def _format_for_visualization(self, results: dict) -> dict[str, Any]:
        colors = {
            "Candidate": "#4ECDC4",
            "Skill": "#45B7D1",
            "Company": "#FF6B6B",
            "Role": "#96CEB4",
            "Location": "#FFEAA7",
            "Education": "#DDA0DD",
            "Certification": "#FFB347",
            "Entity": "#999999",
        }
        nodes = [
            {
                "id": n["uuid"],
                "label": n["name"],
                "type": n["labels"][0] if n.get("labels") else "Entity",
                "color": colors.get(n["labels"][0] if n.get("labels") else "Entity", "#999999"),
                "size": 10,
                "properties": n.get("attributes", {}),
            }
            for n in results.get("nodes", [])
        ]
        edges = [
            {
                "id": e["uuid"],
                "source": e["source_node_uuid"],
                "target": e["target_node_uuid"],
                "label": e.get("name", ""),
                "fact": e.get("fact", ""),
                "valid_at": e.get("valid_at"),
            }
            for e in results.get("edges", [])
        ]
        return {"nodes": nodes, "edges": edges}


# ── Module-level helpers ────────────────────────────────────────────────────────

def _is_candidate_node(node: dict) -> bool:
    """True if this node represents a Candidate entity."""
    labels = node.get("labels", [])
    name = node.get("name", "")
    return "Candidate" in labels or (isinstance(name, str) and name.startswith("[CANDIDATE:"))


def _parsed_to_dict(parsed) -> dict:
    """Serialise a ParsedQuery to a plain dict for inclusion in API responses."""
    return {
        "location": parsed.location,
        "min_yoe": parsed.min_yoe,
        "max_yoe": parsed.max_yoe,
        "must_skills": parsed.must_skills,
        "nice_skills": parsed.nice_skills,
        "role": parsed.role,
        "company_type": parsed.company_type,
        "industries": parsed.industries,
        "semantic_query": parsed.semantic_query,
    }
