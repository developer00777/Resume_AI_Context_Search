"""
Resume Intelligence service — Graphiti wrapper for resume ingestion and search.
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
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_RRF,
)

from config.entity_types import ENTITY_TYPES
from config.edge_types import EDGE_TYPES, EDGE_TYPE_MAP
from models.resume import Resume

logger = logging.getLogger(__name__)

# All resumes live under one group so cross-candidate graph queries work
RESUME_GROUP_ID = "resume-pool"


def current_years_experience(
    earliest_role_start: date | None,
    captured_yoe: float | None = None,
    parsed_on: date | None = None,
    today: date | None = None,
) -> float | None:
    """Always-current YoE. Anchors on earliest role start when available,
    otherwise rolls the captured value forward from its parse date."""
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

    Handles connection, ingestion (single + bulk), semantic search,
    and pre-built graph queries.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        openai_api_key: str,
        openai_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_model: str = "openai/text-embedding-3-small",
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key or openai_api_key
        self.embedding_base_url = embedding_base_url or openai_base_url
        self.client: Optional[Graphiti] = None

    async def connect(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required.")

        llm_client = OpenAIClient(
            config=LLMConfig(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                model=self.model_name,
                small_model=self.model_name,
            )
        )
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=self.embedding_api_key,
                base_url=self.embedding_base_url,
                embedding_model=self.embedding_model,
            )
        )
        self.client = Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
        )
        await self.client.build_indices_and_constraints()
        logger.info("ResumeService connected to Neo4j and indices built")

    async def disconnect(self) -> None:
        if self.client:
            await self.client.close()
            self.client = None

    # ── Ingestion ──────────────────────────────────────────────

    async def ingest_resume(self, resume: Resume) -> None:
        """Ingest a single resume as a Graphiti episode."""
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        await self.client.add_episode(
            name=f"Resume: {resume.candidate_name or resume.source_id}",
            episode_body=resume.to_episode_content(),
            source=EpisodeType.message,
            source_description=f"resume (source_id={resume.source_id})",
            reference_time=datetime.now(timezone.utc),
            group_id=RESUME_GROUP_ID,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )

    async def ingest_resumes_bulk(self, resumes: list[Resume]) -> None:
        """Bulk ingest multiple resumes. Prefer this over single ingestion."""
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")
        if not resumes:
            return

        now = datetime.now(timezone.utc)
        raw_episodes = [
            RawEpisode(
                name=f"Resume: {r.candidate_name or r.source_id}",
                content=r.to_episode_content(),
                source=EpisodeType.message,
                source_description=f"resume (source_id={r.source_id})",
                reference_time=now,
            )
            for r in resumes
        ]

        await self.client.add_episode_bulk(
            raw_episodes,
            group_id=RESUME_GROUP_ID,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )
        logger.info("Bulk ingested %d resumes", len(resumes))

    # ── Search ─────────────────────────────────────────────────

    async def search(self, query: str, num_results: int = 50) -> dict[str, Any]:
        """
        Hybrid semantic + keyword search over the resume pool.

        Returns nodes (candidates, skills, companies) and edges (relationships).
        The caller applies any structured filters (YoE, location) via the API layer.
        """
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        results = await self.client._search(
            query=query,
            group_ids=[RESUME_GROUP_ID],
            config=COMBINED_HYBRID_SEARCH_RRF,
            num_results=num_results,
        )

        return {
            "nodes": [self._node_to_dict(n) for n in results.nodes],
            "edges": [self._edge_to_dict(e) for e in results.edges],
        }

    async def get_candidate(self, candidate_name: str) -> dict[str, Any]:
        """Get a candidate's full graph context by name."""
        return await self.search(
            query=f"Everything about candidate {candidate_name}: skills, roles, companies, education",
            num_results=100,
        )

    # ── Pre-built graph queries ────────────────────────────────

    async def find_similar_candidates(self, candidate_name: str, num_results: int = 20) -> dict[str, Any]:
        """Find candidates semantically similar to a given candidate."""
        return await self.search(
            query=f"Candidates with similar background and skills to {candidate_name}",
            num_results=num_results,
        )

    async def get_skill_pool(self, skill: str) -> dict[str, Any]:
        """Get all candidates with a specific skill and their adjacent skills."""
        return await self.search(
            query=f"Candidates who have {skill} skill, their experience and related skills",
            num_results=100,
        )

    async def get_pool_intelligence(self) -> dict[str, Any]:
        """High-level pool stats — which skills, locations, and roles dominate."""
        return await self.search(
            query="Overview of all candidate skills, locations, companies, roles, and experience levels in the pool",
            num_results=200,
        )

    async def get_graph_for_visualization(self) -> dict[str, Any]:
        """Full graph data for D3 visualization."""
        results = await self.get_pool_intelligence()
        return self._format_for_visualization(results)

    # ── Helpers ────────────────────────────────────────────────

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
        nodes = []
        for node in results.get("nodes", []):
            node_type = node["labels"][0] if node.get("labels") else "Entity"
            nodes.append({
                "id": node["uuid"],
                "label": node["name"],
                "type": node_type,
                "color": colors.get(node_type, "#999999"),
                "size": 10,
                "properties": node.get("attributes", {}),
            })
        edges = []
        for edge in results.get("edges", []):
            edges.append({
                "id": edge["uuid"],
                "source": edge["source_node_uuid"],
                "target": edge["target_node_uuid"],
                "label": edge.get("name", ""),
                "fact": edge.get("fact", ""),
                "valid_at": edge.get("valid_at"),
            })
        return {"nodes": nodes, "edges": edges}
