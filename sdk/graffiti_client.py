"""
GraffitiClient — async Python client for CHAMP Graph.

Designed for CHAMP Mail and other agent systems to import directly.
Wraps the REST API with retry logic, error handling, and typed responses.

Usage:
    from sdk import GraffitiClient

    async with GraffitiClient("https://your-champ-graph.railway.app") as client:
        await client.log_email(
            account_name="Acme Corp",
            from_address="rep@ourco.com",
            to_address="john@acme.com",
            subject="Follow-up",
            body="Hi John, ...",
            direction="outbound",
        )
        context = await client.get_email_context(
            account_name="Acme Corp",
            contact_email="john@acme.com",
        )
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class GraffitiClientError(Exception):
    """Raised when the CHAMP Graph API returns an error."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"CHAMP Graph API error {status_code}: {detail}")


class GraffitiClient:
    """
    Async client for the CHAMP Graph knowledge graph REST API.

    Parameters
    ----------
    base_url : str
        Base URL of the CHAMP Graph server (e.g., "https://your-app.railway.app")
    api_key : str, optional
        API key for X-API-Key authentication. If not set, auth is disabled.
    timeout : float
        Request timeout in seconds (default 30).
    max_retries : int
        Number of retries on transient failures (default 2).
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        max_connections: int = 20,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_connections = max_connections
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self) -> None:
        """Create the HTTP client."""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_connections // 2,
            ),
        )

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # ==========================================================
    # Health
    # ==========================================================

    async def health_check(self) -> Dict[str, Any]:
        """Check CHAMP Graph server health."""
        return await self._get("/health")

    # ==========================================================
    # Write operations — push data into the graph
    # ==========================================================

    async def log_email(
        self,
        account_name: str,
        from_address: str,
        to_address: str,
        subject: str,
        body: str,
        direction: str = "outbound",
    ) -> Dict[str, Any]:
        """
        Log a single email interaction into the knowledge graph.

        Parameters
        ----------
        account_name : str
            The company/account name (e.g., "Acme Corp")
        from_address : str
            Sender email address
        to_address : str
            Recipient email address
        subject : str
            Email subject line
        body : str
            Email body text
        direction : str
            "outbound" if we sent it, "inbound" if we received it
        """
        return await self._post("/api/hooks/email", json={
            "account_name": account_name,
            "from_address": from_address,
            "to_address": to_address,
            "subject": subject,
            "body": body,
            "direction": direction,
        })

    async def log_email_batch(
        self,
        account_name: str,
        emails: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Log multiple emails at once.

        Parameters
        ----------
        account_name : str
            The company/account name
        emails : list of dict
            Each dict has: from_address, to_address, subject, body, direction
        """
        return await self._post("/api/hooks/email/batch", json={
            "account_name": account_name,
            "emails": emails,
        })

    async def log_call(
        self,
        account_name: str,
        contact_name: str,
        summary: str,
        duration_minutes: int = 0,
        direction: str = "outbound",
        transcript: str = "",
    ) -> Dict[str, Any]:
        """Log a call interaction."""
        return await self._post("/api/hooks/call", json={
            "account_name": account_name,
            "contact_name": contact_name,
            "summary": summary,
            "duration_minutes": duration_minutes,
            "direction": direction,
            "transcript": transcript,
        })

    async def remember(
        self,
        account_name: str,
        content: str,
        source: str = "agent",
        name: str = "Agent note",
    ) -> Dict[str, Any]:
        """Store arbitrary information in the knowledge graph."""
        return await self._post("/api/ingest", json={
            "account_name": account_name,
            "mode": "raw",
            "content": content,
            "name": name,
            "source_description": f"Agent ingestion ({source})",
        })

    # ==========================================================
    # Read operations — pull context from the graph
    # ==========================================================

    async def get_email_context(
        self,
        account_name: str,
        contact_email: Optional[str] = None,
        contact_name: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get full context for composing an email follow-up.

        Returns interaction history, personal details, topics discussed,
        open opportunities, and suggested follow-up angles.

        Parameters
        ----------
        account_name : str
            The company/account name
        contact_email : str, optional
            Email address of the contact to focus on
        contact_name : str, optional
            Name of the contact to focus on
        subject : str, optional
            Subject/topic of the email for focused context
        """
        params: Dict[str, str] = {}
        if contact_email:
            params["contact_email"] = contact_email
        if contact_name:
            params["contact_name"] = contact_name
        if subject:
            params["subject"] = subject

        return await self._get(
            f"/api/accounts/{account_name}/email-context",
            params=params,
        )

    async def get_briefing(self, account_name: str) -> Dict[str, Any]:
        """Get a comprehensive pre-call/pre-email briefing."""
        return await self._get(f"/api/accounts/{account_name}/briefing")

    async def recall(
        self,
        account_name: str,
        query: str,
        num_results: int = 10,
    ) -> Dict[str, Any]:
        """Search the knowledge graph with a natural language query."""
        return await self._post("/api/query", json={
            "account": account_name,
            "query": query,
            "num_results": num_results,
        })

    async def get_timeline(
        self,
        account_name: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Get recent interaction timeline for an account."""
        return await self._get(
            f"/api/accounts/{account_name}/timeline",
            params={"limit": str(limit)},
        )

    async def get_contacts(self, account_name: str) -> Dict[str, Any]:
        """Get all contacts for an account."""
        return await self._get(f"/api/accounts/{account_name}/contacts")

    async def get_stakeholders(self, account_name: str) -> Dict[str, Any]:
        """Get stakeholder map (champions, blockers, decision-makers)."""
        return await self._get(
            f"/api/accounts/{account_name}/intelligence/stakeholder-map"
        )

    async def find_stale_contacts(
        self,
        account_name: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Find contacts not interacted with recently."""
        return await self._get(
            f"/api/accounts/{account_name}/intelligence/engagement-gaps",
            params={"days": str(days)},
        )

    # ==========================================================
    # Internal HTTP helpers
    # ==========================================================

    async def _get(
        self,
        path: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """GET request with retry."""
        return await self._request("GET", path, params=params)

    async def _post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """POST request with retry."""
        return await self._request("POST", path, json=json)

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute HTTP request with retries on transient failures."""
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() or use async with.")

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(
                    method, path, params=params, json=json
                )

                if response.status_code >= 500 and attempt < self.max_retries:
                    last_error = GraffitiClientError(
                        response.status_code, response.text
                    )
                    logger.warning(
                        "CHAMP Graph %s %s returned %d, retrying (%d/%d)",
                        method, path, response.status_code,
                        attempt + 1, self.max_retries,
                    )
                    continue

                if response.status_code >= 400:
                    raise GraffitiClientError(
                        response.status_code, response.text
                    )

                return response.json()

            except httpx.TimeoutException:
                last_error = GraffitiClientError(
                    408, f"Request timed out: {method} {path}"
                )
                if attempt < self.max_retries:
                    logger.warning(
                        "CHAMP Graph %s %s timed out, retrying (%d/%d)",
                        method, path, attempt + 1, self.max_retries,
                    )
                    continue
                raise last_error

            except httpx.ConnectError as e:
                last_error = GraffitiClientError(
                    503, f"Connection failed: {e}"
                )
                if attempt < self.max_retries:
                    logger.warning(
                        "CHAMP Graph connection failed, retrying (%d/%d)",
                        attempt + 1, self.max_retries,
                    )
                    continue
                raise last_error

        raise last_error  # type: ignore[misc]
