"""
Resume Intelligence API Server

Provides REST endpoints for ingesting resumes and searching the
knowledge graph via Graphiti on Neo4j/FalkorDB.
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.auth import verify_api_key
from api.models import (
    CandidateResponse,
    GraphResponse,
    HealthResponse,
    IngestResponse,
    ResumeBulkIngestRequest,
    ResumeIngestRequest,
    SearchRequest,
    SearchResponse,
)
from config.settings import get_settings
from models.resume import Resume
from services.resume_service import ResumeService, current_years_experience

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

resume_service: Optional[ResumeService] = None
_start_time: float = 0.0


def _require_service() -> ResumeService:
    if not resume_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return resume_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global resume_service, _start_time
    _start_time = time.time()
    settings = get_settings()

    resume_service = ResumeService(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        llm_api_key=settings.llm_api_key,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        embedding_api_key=settings.embedding_api_key,
        embedding_base_url=settings.embedding_base_url,
    )

    async def _connect_with_retry() -> None:
        for attempt in range(1, 31):
            try:
                await resume_service.connect()
                logger.info("Resume Intelligence connected to Neo4j (attempt %d)", attempt)
                return
            except Exception as e:
                logger.warning("Neo4j connection attempt %d/30 failed: %s", attempt, e)
                await asyncio.sleep(10)
        logger.error("Could not connect to Neo4j after 30 attempts — running in degraded mode")

    asyncio.create_task(_connect_with_retry())

    if not settings.api_key:
        logger.warning("API key auth is DISABLED — set API_KEY env var to enable")

    yield

    await resume_service.disconnect()
    logger.info("Resume Intelligence service disconnected")


app = FastAPI(
    title="Resume Intelligence API",
    description="Semantic search + knowledge graph for resume data, powered by Graphiti on Neo4j",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins_raw = os.getenv("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    neo4j_ok = False
    if resume_service and resume_service.client:
        try:
            await resume_service.client.driver.execute_query("RETURN 1")
            neo4j_ok = True
        except Exception:
            pass

    settings = get_settings()
    status = "healthy" if neo4j_ok else ("degraded" if resume_service else "unhealthy")
    return HealthResponse(
        status=status,
        service="resume-intelligence",
        version="1.0.0",
        neo4j_connected=neo4j_ok,
        uptime_seconds=round(time.time() - _start_time, 1),
        llm_mode="local" if settings.llm_is_local else "openrouter",
        llm_model=settings.llm_model,
        embedding_mode="local" if settings.embedding_is_local else "openrouter",
        embedding_model=settings.embedding_model,
    )


# ── Ingest ─────────────────────────────────────────────────────────────────────


@app.post("/api/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
async def ingest_resume(request: ResumeIngestRequest):
    """Ingest a single resume into the knowledge graph."""
    service = _require_service()

    parsed_on = None
    if request.parsed_on:
        try:
            parsed_on = date.fromisoformat(request.parsed_on)
        except ValueError:
            pass

    resume = Resume(
        source_id=request.source_id,
        full_text=request.full_text,
        candidate_name=request.candidate_name,
        parsed_on=parsed_on,
    )
    candidate_uuid = await service.ingest_resume(resume)
    return IngestResponse(message="Resume ingested", episodes_ingested=1, candidate_uuid=candidate_uuid)


@app.post("/api/ingest/bulk", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
async def ingest_resumes_bulk(request: ResumeBulkIngestRequest):
    """Bulk ingest up to 500 resumes. Prefer this over single ingest for large loads."""
    service = _require_service()

    resumes = []
    for item in request.resumes:
        parsed_on = None
        if item.parsed_on:
            try:
                parsed_on = date.fromisoformat(item.parsed_on)
            except ValueError:
                pass
        resumes.append(Resume(
            source_id=item.source_id,
            full_text=item.full_text,
            candidate_name=item.candidate_name,
            parsed_on=parsed_on,
        ))

    candidate_uuids = await service.ingest_resumes_bulk(resumes)
    return IngestResponse(
        message=f"Ingested {len(resumes)} resumes",
        episodes_ingested=len(resumes),
        candidate_uuids=candidate_uuids,
    )


# ── Search ─────────────────────────────────────────────────────────────────────


@app.post("/api/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
async def search(request: SearchRequest):
    """
    Semantic + keyword hybrid search over the resume pool.

    Natural language queries work: "backend engineers with Salesforce experience in Bangalore".
    Returns ranked nodes and edges.
    """
    service = _require_service()
    try:
        results = await service.search(query=request.query, num_results=request.num_results)
        return SearchResponse(
            query=request.query,
            nodes=results["nodes"],
            edges=results["edges"],
            total_nodes=len(results["nodes"]),
            total_edges=len(results["edges"]),
            parsed_filters=results.get("parsed_filters"),
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Candidate ──────────────────────────────────────────────────────────────────


@app.get("/api/candidates/{candidate_name}", response_model=CandidateResponse, dependencies=[Depends(verify_api_key)])
async def get_candidate(candidate_name: str):
    """Get a candidate's full graph context — skills, roles, companies, education."""
    service = _require_service()
    results = await service.get_candidate(candidate_name)
    return CandidateResponse(
        candidate_name=candidate_name,
        nodes=results["nodes"],
        edges=results["edges"],
    )


@app.get("/api/candidates/{candidate_name}/similar", dependencies=[Depends(verify_api_key)])
async def find_similar(candidate_name: str, num_results: int = 20):
    """Find candidates semantically similar to a given candidate."""
    service = _require_service()
    results = await service.find_similar_candidates(candidate_name, num_results=num_results)
    return {"success": True, "candidate_name": candidate_name, **results}


# ── Skill pool ─────────────────────────────────────────────────────────────────


@app.get("/api/skills/{skill}", dependencies=[Depends(verify_api_key)])
async def get_skill_pool(skill: str):
    """Get all candidates with a specific skill and their adjacent skills."""
    service = _require_service()
    results = await service.get_skill_pool(skill)
    return {"success": True, "skill": skill, **results}


# ── Pool intelligence ──────────────────────────────────────────────────────────


@app.get("/api/pool/intelligence", dependencies=[Depends(verify_api_key)])
async def get_pool_intelligence():
    """High-level view of the candidate pool — skill distribution, top locations, roles."""
    service = _require_service()
    results = await service.get_pool_intelligence()
    return {"success": True, **results}


# ── YoE utility ────────────────────────────────────────────────────────────────


@app.get("/api/utils/yoe")
async def compute_yoe(
    earliest_role_start: Optional[str] = None,
    captured_yoe: Optional[float] = None,
    parsed_on: Optional[str] = None,
):
    """
    Compute always-current years of experience from a date anchor.

    - Provide earliest_role_start (YYYY-MM) for the preferred anchor.
    - Provide captured_yoe + parsed_on (YYYY-MM-DD) for the fallback roll-forward.
    """
    anchor = None
    if earliest_role_start:
        try:
            year, month = earliest_role_start.split("-")
            anchor = date(int(year), int(month), 1)
        except (ValueError, AttributeError):
            raise HTTPException(status_code=422, detail="earliest_role_start must be YYYY-MM")

    parsed = None
    if parsed_on:
        try:
            parsed = date.fromisoformat(parsed_on)
        except ValueError:
            raise HTTPException(status_code=422, detail="parsed_on must be YYYY-MM-DD")

    yoe = current_years_experience(
        earliest_role_start=anchor,
        captured_yoe=captured_yoe,
        parsed_on=parsed,
    )
    return {"years_of_experience": yoe}


# ── Graph edge rebuild ─────────────────────────────────────────────────────────


@app.post("/api/graph/rebuild-edges", dependencies=[Depends(verify_api_key)])
async def rebuild_edges():
    """
    Manually trigger SKILL_CO_OCCURS_WITH and SIMILAR_TO edge recomputation.

    This runs automatically after every ingest batch. Call this endpoint if you
    want to force a rebuild (e.g. after a large initial bulk load).
    Runs synchronously so the response confirms completion.
    """
    service = _require_service()
    if not service._edge_builder:
        raise HTTPException(status_code=503, detail="EdgeBuilder not initialised")
    summary = await service._edge_builder.build_all()
    return {"success": True, **summary}


# ── Graph visualization ────────────────────────────────────────────────────────


@app.get("/api/graph", response_model=GraphResponse, dependencies=[Depends(verify_api_key)])
async def get_graph():
    """Full graph data for D3 visualization."""
    service = _require_service()
    graph = await service.get_graph_for_visualization()
    return GraphResponse(nodes=graph["nodes"], edges=graph["edges"])


# ── Main ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api_server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
