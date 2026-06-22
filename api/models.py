"""
API request and response models for Resume Intelligence.
"""
from typing import Any, Optional
from pydantic import BaseModel, Field


# --- Health ---


class HealthResponse(BaseModel):
    status: str  # healthy | degraded | unhealthy
    service: str
    version: str
    neo4j_connected: bool
    uptime_seconds: float


# --- Ingest ---


class ResumeIngestRequest(BaseModel):
    """Ingest a single resume."""
    source_id: str = Field(..., description="Unique ID from the source system (Salesforce record ID)")
    full_text: str = Field(..., description="Full raw resume text")
    candidate_name: Optional[str] = Field(None, description="Candidate name if known")
    parsed_on: Optional[str] = Field(None, description="ISO date (YYYY-MM-DD) when the original Salesforce parse ran")


class ResumeBulkIngestRequest(BaseModel):
    """Bulk ingest up to 500 resumes."""
    resumes: list[ResumeIngestRequest] = Field(..., min_length=1, max_length=500)


class IngestResponse(BaseModel):
    success: bool = True
    message: str
    episodes_ingested: int


# --- Search ---


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    num_results: int = Field(default=50, ge=1, le=200)


class SearchResponse(BaseModel):
    success: bool = True
    query: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    total_nodes: int
    total_edges: int


# --- Candidate ---


class CandidateResponse(BaseModel):
    success: bool = True
    candidate_name: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


# --- Graph visualization ---


class GraphResponse(BaseModel):
    success: bool = True
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
