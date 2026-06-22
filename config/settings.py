"""
Environment configuration for Resume Intelligence service.

All four LLM/embedding modes are controlled purely via Railway environment variables:

  Mode 1 — Embed via OpenRouter:
    EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
    EMBEDDING_MODEL=openai/text-embedding-3-small
    EMBEDDING_API_KEY=sk-or-v1-...

  Mode 2 — Embed via local Ollama:
    EMBEDDING_BASE_URL=http://localhost:11434/v1
    EMBEDDING_MODEL=nomic-embed-text
    EMBEDDING_API_KEY=ollama

  Mode 3 — Query LLM via OpenRouter:
    LLM_BASE_URL=https://openrouter.ai/api/v1
    LLM_MODEL=anthropic/claude-sonnet-4
    LLM_API_KEY=sk-or-v1-...

  Mode 4 — Query LLM via local Ollama:
    LLM_BASE_URL=http://localhost:11434/v1
    LLM_MODEL=llama3.1:8b
    LLM_API_KEY=ollama

Mix freely: local embeddings + cloud LLM, or vice versa.
"""
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── LLM (extraction + query rewriting) ──────────────────────────────────
    # Used by Graphiti for entity extraction during ingest, and query parsing.
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="anthropic/claude-sonnet-4", alias="LLM_MODEL")

    # ── Embeddings ───────────────────────────────────────────────────────────
    # Independently configurable — can point to a different provider than the LLM.
    embedding_api_key: Optional[str] = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_base_url: Optional[str] = Field(default=None, alias="EMBEDDING_BASE_URL")
    embedding_model: str = Field(default="openai/text-embedding-3-small", alias="EMBEDDING_MODEL")

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(..., description="Neo4j password — required")

    # ── API auth ─────────────────────────────────────────────────────────────
    api_key: Optional[str] = Field(default=None)

    # ── Server ───────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)

    @model_validator(mode="after")
    def _resolve_embedding_key(self) -> "Settings":
        # If no dedicated embedding key, fall back to the LLM key.
        if not self.embedding_api_key:
            self.embedding_api_key = self.llm_api_key
        # If no dedicated embedding base URL, fall back to the LLM base URL.
        if not self.embedding_base_url:
            self.embedding_base_url = self.llm_base_url
        return self

    @property
    def llm_is_local(self) -> bool:
        return "localhost" in self.llm_base_url or "127.0.0.1" in self.llm_base_url

    @property
    def embedding_is_local(self) -> bool:
        return "localhost" in (self.embedding_base_url or "") or "127.0.0.1" in (self.embedding_base_url or "")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
