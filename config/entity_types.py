"""
Entity types for the Resume Intelligence knowledge graph.

Guides Graphiti's LLM extraction to produce structured nodes from raw resume text.

Note: Fields cannot use graphiti-core's protected EntityNode attribute names:
  uuid, name, group_id, labels, created_at, name_embedding, summary, attributes
"""
from typing import Optional
from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """A job candidate extracted from a resume"""
    full_name: str = Field(..., description="Full name of the candidate")
    source_id: Optional[str] = Field(None, description="ID from the source system (e.g. Salesforce record ID)")
    earliest_role_start: Optional[str] = Field(
        None,
        description="ISO date YYYY-MM of the candidate's earliest employment start — used to compute live years of experience. Extract from the oldest job in the work history."
    )
    current_location: Optional[str] = Field(None, description="Current city or region where the candidate is based")
    email: Optional[str] = Field(None, description="Contact email if present in the resume")
    phone: Optional[str] = Field(None, description="Contact phone number if present")


class Skill(BaseModel):
    """A technical or professional skill mentioned in a resume"""
    normalized: str = Field(
        ...,
        description=(
            "Canonical lowercase skill name. Apply these aliases: "
            "JS/JavaScript/ES6→javascript, Node/NodeJS→nodejs, "
            "ML/Machine Learning→machine learning, "
            "SF/SFDC/Salesforce CRM→salesforce, "
            "React/ReactJS→react, k8s→kubernetes, "
            "Postgres/PostgreSQL→postgresql. "
            "Always lowercase, no punctuation."
        )
    )
    evidence: Optional[str] = Field(
        None,
        description="Short verbatim snippet from the resume that justifies this skill (max 120 chars)"
    )
    recency: Optional[str] = Field(
        None,
        description="How recently used: 'current' (within last 2 years), 'recent' (2-5 years), 'dated' (5+ years)"
    )


class Role(BaseModel):
    """A job title or position held by a candidate"""
    title: str = Field(..., description="Job title as written in the resume, e.g. 'Senior Backend Engineer'")
    normalized: Optional[str] = Field(
        None,
        description="Canonical lowercase title for grouping, e.g. 'backend engineer', 'data scientist'"
    )


class Company(BaseModel):
    """A company where a candidate has worked"""
    company_name: str = Field(..., description="Company or organisation name as written in the resume")
    industry: Optional[str] = Field(
        None,
        description="Industry sector, e.g. FinTech, SaaS, Healthcare, E-commerce, Consulting"
    )
    company_type: Optional[str] = Field(
        None,
        description="Type: 'product' (builds own product), 'service' (consulting/outsourcing), 'startup', 'enterprise', 'agency'"
    )


class Location(BaseModel):
    """A geographic location associated with a candidate or company"""
    city: str = Field(..., description="City or region name")
    country: Optional[str] = Field(None, description="Country")


class Education(BaseModel):
    """An educational qualification from a resume"""
    degree: str = Field(..., description="Degree name, e.g. 'B.Tech CSE', 'MBA', 'M.S. Computer Science'")
    institution: Optional[str] = Field(None, description="University or institution name")
    graduation_year: Optional[int] = Field(None, description="Year of graduation as integer")


class Certification(BaseModel):
    """A professional certification or credential"""
    cert_name: str = Field(..., description="Certification name, e.g. 'AWS Solutions Architect Associate'")
    issuer: Optional[str] = Field(None, description="Issuing body, e.g. 'Amazon Web Services', 'Google', 'Salesforce'")


# Entity types dictionary for Graphiti
ENTITY_TYPES = {
    'Candidate': Candidate,
    'Skill': Skill,
    'Role': Role,
    'Company': Company,
    'Location': Location,
    'Education': Education,
    'Certification': Certification,
}
