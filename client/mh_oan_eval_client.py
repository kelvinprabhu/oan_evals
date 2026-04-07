import json
import time
import threading
from dotenv import load_dotenv
import requests
load_dotenv()


import logging
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class Mh_OANEvalClient:
    """
    HTTP client for the OAN evaluation service.

    Supports context-manager usage::

        with OANEvalClient(base_url=..., token=...) as client:
            response = client.chat("What crops grow in Karnataka?")
    """

    def __init__(
        self,
        base_url: str = "",
        token: str | None = None,
        liveness_retry_count: int = 5,
        liveness_retry_wait: float = 3.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.liveness_retry_count = liveness_retry_count
        self.liveness_retry_wait = liveness_retry_wait

        self._lock = threading.Lock()
        self._session = self._build_session()
        self._wait_for_liveness()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        """Create a session with connection pooling and idempotent retries."""
        session = requests.Session()
        retry = Retry(
            total=3,
            status_forcelist={502, 503, 504},
            allowed_methods={"GET"},
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        if self.token:
            session.headers["Authorization"] = f"Bearer {self.token}"
        return session

    def _wait_for_liveness(self) -> None:
        url = f"{self.base_url}/api/health/live"
        for attempt in range(1, self.liveness_retry_count + 1):
            try:
                r = self._session.get(url, headers={"Accept": "application/json"}, timeout=5)
                # 200 = live; 403 = live but auth-gated (WAF / reverse proxy in CI)
                if r.status_code in (200, 403):
                    logger.info("Service is live — HTTP %s (attempt %d)", r.status_code, attempt)
                    return
                logger.warning(
                    "Liveness check failed: HTTP %s (attempt %d/%d)",
                    r.status_code, attempt, self.liveness_retry_count,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "Liveness check error: %s (attempt %d/%d)",
                    exc, attempt, self.liveness_retry_count,
                )

            if attempt < self.liveness_retry_count:
                time.sleep(self.liveness_retry_wait)

        raise RuntimeError(
            f"Service did not become live after {self.liveness_retry_count} attempts."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        session_id: str = "eval-session",
        user_id: str = "eval-user",
        source_lang: str = "en",
        target_lang: str = "en",
    ) -> str | None:
        """
        Send a chat query and return the full streamed response as a string.

        Returns ``None`` if the server returns a non-200 status or the
        response body is empty after stripping whitespace.
        """
        url = f"{self.base_url}/api/chat/"
        params = {
            "query": query,
            "session_id": session_id,
            "user_id": user_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

        try:
            with self._lock:
                response = self._session.get(url, params=params, stream=True, timeout=60)
        except requests.RequestException as exc:
            logger.error("Chat request raised an exception: %s", exc)
            return None

        if response.status_code != 200:
            logger.error("Chat request failed: HTTP %s", response.status_code)
            return None

        raw: bytearray = bytearray()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                raw.extend(chunk)

        result = raw.decode("utf-8", errors="replace").strip()
        return result or None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._session.close()
        logger.debug("OANEvalClient session closed.")

    # Alias kept for backward compatibility
    shutdown = close

    def __enter__(self) -> "Mh_OANEvalClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()