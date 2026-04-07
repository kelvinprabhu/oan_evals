import json
import time
import threading
from dotenv import load_dotenv
import requests
load_dotenv()


class OANEvalClient:
    def __init__(
        self,
        base_url: str = "",
        api_key: str | None = None,
        liveness_retry_count: int = 5,
        liveness_retry_wait: float = 3.0,
        token_refresh_buffer: float = 60.0,  # seconds before expiry to refresh
        token_params: dict = {
            "mobile": "9876543212",
            "name": "OAN Eval Client",
            "role": "Evaluvater",
            "metadata": "v1.0"
        },
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.liveness_retry_count = liveness_retry_count
        self.liveness_retry_wait = liveness_retry_wait
        self.token_refresh_buffer = token_refresh_buffer
        self.token_params = token_params or {
            "mobile": "0000000000",
            "name": "Deepeval Tester",
            "role": "Evaluator",
            "metadata": "Testing access token generation for DeepEval moderation evals",
        }

        self._token: str | None = None
        self._token_expiry: float = 0.0  # unix timestamp
        self._lock = threading.Lock()
        self._refresh_timer: threading.Timer | None = None

        self._wait_for_liveness()
        if self.api_key:
            self._token = self.api_key
            self._token_expiry = time.time() + (365 * 24 * 60 * 60)
        else:
            self._refresh_token()

    # ------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------

    def _wait_for_liveness(self) -> None:
        url = f"{self.base_url}/api/health/live"
        for attempt in range(1, self.liveness_retry_count + 1):
            try:
                response = requests.get(url, headers={"accept": "application/json"}, timeout=5)
                # 200 = live, 403 = live but auth-gated (e.g. WAF/reverse proxy in CI)
                if response.status_code in (200, 403):
                    print(f"[OANEvalClient] Service is live — status {response.status_code} (attempt {attempt})")
                    return
                print(f"[OANEvalClient] Liveness check failed: status {response.status_code} (attempt {attempt}/{self.liveness_retry_count})")
            except requests.RequestException as e:
                print(f"[OANEvalClient] Liveness check error: {e} (attempt {attempt}/{self.liveness_retry_count})")

            if attempt < self.liveness_retry_count:
                time.sleep(self.liveness_retry_wait)

        raise RuntimeError(
            f"[OANEvalClient] Service did not become live after {self.liveness_retry_count} attempts."
        )

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _fetch_token(self) -> tuple[str, float]:
        """Returns (token, expiry_unix_timestamp)."""
        url = f"{self.base_url}/api/token"
        response = requests.post(
            url,
            params=self.token_params,
            headers={"accept": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        token = data["token"]
        expires_in: int = data.get("expires_in", 900)
        expiry = time.time() + expires_in
        return token, expiry

    def _refresh_token(self) -> None:
        with self._lock:
            token, expiry = self._fetch_token()
            self._token = token
            self._token_expiry = expiry
            print(f"[OANEvalClient] Token refreshed — expires in {int(expiry - time.time())}s")

        # Cancel previous timer if any
        if self._refresh_timer is not None:
            self._refresh_timer.cancel()

        # Schedule next refresh before expiry
        refresh_in = max((self._token_expiry - time.time()) - self.token_refresh_buffer, 5)
        self._refresh_timer = threading.Timer(refresh_in, self._refresh_token)
        self._refresh_timer.daemon = True
        self._refresh_timer.start()
        print(f"[OANEvalClient] Next token refresh in {int(refresh_in)}s")

    @property
    def token(self) -> str:
        with self._lock:
            if self._token is None:
                raise RuntimeError("[OANEvalClient] Token not initialized")
            if time.time() >= self._token_expiry - self.token_refresh_buffer:
                # Synchronous fallback if timer didn't fire in time
                print("[OANEvalClient] Token near expiry — refreshing synchronously")
                self._refresh_token()
            return self._token

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        session_id: str = "eval-session",
        user_id: str = "eval-user",
        source_lang: str = "en",
        target_lang: str = "en",
    ) -> str | None:
        response = requests.get(
            f"{self.base_url}/api/chat/",
            params={
                "query": query,
                "session_id": session_id,
                "user_id": user_id,
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
            headers={"Authorization": f"Bearer {self.token}"},
            stream=True,
            timeout=60,
        )

        if response.status_code != 200:
            print(f"[OANEvalClient] Chat request failed: {response.status_code}")
            return None

        raw: bytearray = bytearray()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                raw.extend(chunk)

        output = raw.decode("utf-8", errors="replace").strip()
        return output or None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.cancel()
        print("[OANEvalClient] Shutdown complete.")
