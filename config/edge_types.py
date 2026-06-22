"""
Edge (relationship) types for the Resume Intelligence knowledge graph.

These define the relationships between resume entities extracted by Graphiti.
Update EDGE_TYPE_MAP whenever adding new edge types.
"""
from typing import Optional
from pydantic import BaseModel, Field


class HasSkill(BaseModel):
    """Candidate possesses a skill"""
    proficiency: Optional[str] = Field(
        None,
        description="Proficiency level: 'expert', 'proficient', 'familiar'"
    )
    recency: Optional[str] = Field(
        None,
        description="How recently used: 'current', 'recent', 'dated'"
    )
    evidence: Optional[str] = Field(
        None,
        description="Short snippet from the resume supporting this skill claim (max 120 chars)"
    )
    years_used: Optional[float] = Field(None, description="Approximate years of experience with this skill")


class WorkedAt(BaseModel):
    """Candidate worked at a company"""
    start: Optional[str] = Field(None, description="Employment start date as YYYY-MM")
    end: Optional[str] = Field(None, description="Employment end date as YYYY-MM, or 'present'")
    title: Optional[str] = Field(None, description="Job title held at this company")
    tenure_months: Optional[int] = Field(None, description="Approximate tenure in months")


class WorkedAs(BaseModel):
    """Candidate held a specific role/title"""
    start: Optional[str] = Field(None, description="Start date as YYYY-MM")
    end: Optional[str] = Field(None, description="End date as YYYY-MM, or 'present'")
    company: Optional[str] = Field(None, description="Company where this role was held")


class LocatedIn(BaseModel):
    """Candidate is or was located in a city/region"""
    is_current: Optional[bool] = Field(None, description="True if this is the candidate's current location")


class StudiedAt(BaseModel):
    """Candidate attended an educational institution"""
    degree: Optional[str] = Field(None, description="Degree pursued at this institution")
    graduation_year: Optional[int] = Field(None, description="Year of graduation")


class HoldsCertification(BaseModel):
    """Candidate holds a professional certification"""
    obtained_year: Optional[int] = Field(None, description="Year the certification was obtained")
    is_active: Optional[bool] = Field(None, description="Whether the certification is currently valid")


class SkillCoOccursWith(BaseModel):
    """Two skills appear together on resumes — used for adjacent-skill discovery"""
    co_occurrence_count: Optional[int] = Field(
        None,
        description="Number of resumes where both skills appear together"
    )


class SimilarTo(BaseModel):
    """Candidate is semantically similar to another candidate (derived from embedding distance)"""
    similarity_score: Optional[float] = Field(
        None,
        description="Cosine similarity score between 0 and 1 (higher = more similar)"
    )


# Edge types dictionary for Graphiti
EDGE_TYPES = {
    'HAS_SKILL': HasSkill,
    'WORKED_AT': WorkedAt,
    'WORKED_AS': WorkedAs,
    'LOCATED_IN': LocatedIn,
    'STUDIED_AT': StudiedAt,
    'HOLDS_CERTIFICATION': HoldsCertification,
    'SKILL_CO_OCCURS_WITH': SkillCoOccursWith,
    'SIMILAR_TO': SimilarTo,
}

# Edge type map: constrains which edges can connect which node type pairs
EDGE_TYPE_MAP = {
    ('Candidate', 'Skill'): ['HAS_SKILL'],
    ('Candidate', 'Company'): ['WORKED_AT'],
    ('Candidate', 'Role'): ['WORKED_AS'],
    ('Candidate', 'Location'): ['LOCATED_IN'],
    ('Candidate', 'Education'): ['STUDIED_AT'],
    ('Candidate', 'Certification'): ['HOLDS_CERTIFICATION'],
    ('Skill', 'Skill'): ['SKILL_CO_OCCURS_WITH'],
    ('Candidate', 'Candidate'): ['SIMILAR_TO'],
}
