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
"""
import logging
from datetime import date, datetime, timezone
from typing import Any, Optional

from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from config.entity_types import ENTITY_TYPES
from config.edge_types import EDGE_TYPES, EDGE_TYPE_MAP
from models.resume import Resume
from services.pii_masker import mask_pii
from services.pii_store import PiiStore

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

        self.client = Graphiti(
            uri=self._neo4j_uri,
            user=self._neo4j_user,
            password=self._neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
        )

        await self.client.build_indices_and_constraints()

        # PiiStore uses the same Neo4j driver that Graphiti already opened
        self._pii_store = PiiStore(self.client.driver)
        await self._pii_store.ensure_index()

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
        return candidate_uuids

    # ── Search ─────────────────────────────────────────────────────────────────

    async def search(self, query: str, num_results: int = 50) -> dict[str, Any]:
        """
        Hybrid semantic + keyword search.

        Tokens in returned facts/summaries are re-hydrated from PiiStore
        so the caller receives real names, emails, and phones.
        """
        if not self.client or not self._pii_store:
            raise RuntimeError("Not connected. Call connect() first.")

        results = await self.client._search(
            query=query,
            group_ids=[RESUME_GROUP_ID],
            config=COMBINED_HYBRID_SEARCH_RRF,
            num_results=num_results,
        )

        nodes = [self._node_to_dict(n) for n in results.nodes]
        edges = [self._edge_to_dict(e) for e in results.edges]

        # Re-hydrate PII tokens in node names/summaries and edge facts
        nodes = await self._rehydrate_nodes(nodes)
        edges = await self._rehydrate_edges(edges)

        return {"nodes": nodes, "edges": edges}

    async def get_candidate(self, candidate_name: str) -> dict[str, Any]:
        return await self.search(
            query=f"Everything about candidate {candidate_name}: skills, roles, companies, education",
            num_results=100,
        )

    async def find_similar_candidates(self, candidate_name: str, num_results: int = 20) -> dict[str, Any]:
        return await self.search(
            query=f"Candidates with similar background and skills to {candidate_name}",
            num_results=num_results,
        )

    async def get_skill_pool(self, skill: str) -> dict[str, Any]:
        return await self.search(
            query=f"Candidates who have {skill} skill, their experience and related skills",
            num_results=100,
        )

    async def get_pool_intelligence(self) -> dict[str, Any]:
        return await self.search(
            query="Overview of all candidate skills, locations, companies, roles, and experience levels in the pool",
            num_results=200,
        )

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
