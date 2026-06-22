"""
Resume data model for the Resume Intelligence ingestion pipeline.
"""
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class Resume(BaseModel):
    """Normalized resume ready for Graphiti ingestion."""

    source_id: str = Field(..., description="ID from source system (Salesforce record ID)")
    full_text: str = Field(..., description="Full raw resume text extracted from PDF/Word")
    candidate_name: Optional[str] = Field(None, description="Candidate name if known at ingest time")
    parsed_on: Optional[date] = Field(None, description="Date the original Salesforce parse ran — used for YoE fallback")

    def to_episode_content(self) -> str:
        header = f"Resume Record\n{'='*40}\n"
        if self.candidate_name:
            header += f"Candidate: {self.candidate_name}\n"
        header += f"Source ID: {self.source_id}\n"
        if self.parsed_on:
            header += f"Originally parsed: {self.parsed_on.isoformat()}\n"
        header += f"\nResume Text:\n{'-'*40}\n"
        # Cap at 12000 chars to stay within LLM context limits
        return header + self.full_text[:12000]
