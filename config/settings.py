"""
Environment configuration for Resume Intelligence service.
"""
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # LLM — OpenRouter or any OpenAI-compatible endpoint
    openai_api_key: str = Field(default="")
    openai_base_url: Optional[str] = Field(default="https://openrouter.ai/api/v1")
    model_name: str = Field(default="anthropic/claude-sonnet-4")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(..., description="Neo4j password — must be set via NEO4J_PASSWORD env var")

    # Embeddings — OpenAI-compatible (set to Ollama base URL for on-server sovereignty)
    embedding_model: str = Field(default="openai/text-embedding-3-small")
    embedding_api_key: Optional[str] = Field(default=None)
    embedding_base_url: Optional[str] = Field(default=None)

    # API authentication (X-API-Key header; disabled when unset)
    api_key: Optional[str] = Field(default=None)

    # Server
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
