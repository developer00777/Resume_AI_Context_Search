"""
API key authentication for CHAMP Graph.
"""
import logging
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from config.settings import get_settings

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(_api_key_header),
) -> Optional[str]:
    """
    Verify API key from X-API-Key header.

    If CHAMP_GRAPH_API_KEY is not configured, auth is disabled (dev mode).
    If configured, the header must match exactly.
    """
    settings = get_settings()

    if not settings.api_key:
        logger.debug("API key auth is disabled — no CHAMP_GRAPH_API_KEY configured")
        return None

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={"code": "missing_api_key", "message": "X-API-Key header required"},
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=403,
            detail={"code": "invalid_api_key", "message": "Invalid API key"},
        )

    return api_key
