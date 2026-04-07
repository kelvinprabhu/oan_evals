from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from settings.config import BASE_URL, MAX_WORKERS
from models.models import OANTestCase
from client.oan_eval_client import OANEvalClient
from client.mh_oan_eval_client import Mh_OANEvalClient

_oan_client: Optional[OANEvalClient] = None
_mh_client: Optional[Mh_OANEvalClient] = None


def get_client(base_url: str | None = None, api_key: str | None = None) -> OANEvalClient:
    global _oan_client
    desired_url = (base_url or BASE_URL).rstrip("/")
    if _oan_client is None:
        _oan_client = OANEvalClient(base_url=desired_url, api_key=api_key)
    else:
        if _oan_client.base_url != desired_url or (api_key and _oan_client.api_key != api_key):
            _oan_client = OANEvalClient(base_url=desired_url, api_key=api_key)
    return _oan_client


def get_mh_client(base_url: str, token: str) -> Mh_OANEvalClient:
    """Return a (possibly cached) Mh_OANEvalClient instance."""
    global _mh_client
    desired_url = base_url.rstrip("/")
    if _mh_client is None:
        _mh_client = Mh_OANEvalClient(base_url=desired_url, token=token)
    else:
        if _mh_client.base_url != desired_url or _mh_client.token != token:
            _mh_client.close()
            _mh_client = Mh_OANEvalClient(base_url=desired_url, token=token)
    return _mh_client


def _call_api(tc: OANTestCase, base_url: str | None, api_key: str | None) -> tuple[str, str | None]:
    output = get_client(base_url=base_url, api_key=api_key).chat(
        query=tc.input,
        session_id=tc.session_id,
        user_id="eval-user",
        source_lang=tc.language,
        target_lang=tc.language,
    )
    return tc.name, output


def _call_mh_api(tc: OANTestCase, base_url: str, token: str) -> tuple[str, str | None]:
    output = get_mh_client(base_url=base_url, token=token).chat(
        query=tc.input,
        session_id=tc.session_id,
        user_id="eval-user",
        source_lang=tc.language,
        target_lang=tc.language,
    )
    return tc.name, output


def fetch_all_outputs(
    cases: list[OANTestCase],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    max_workers: int | None = None,
) -> dict[str, str | None]:
    results: dict[str, str | None] = {}
    worker_count = max_workers or MAX_WORKERS
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(_call_api, tc, base_url, api_key): tc for tc in cases}
        for future in as_completed(futures):
            tc = futures[future]
            try:
                name, output = future.result()
                results[name] = output
                print(f"[API] OK {name!r} -> {len(output or '')} chars")
            except Exception as exc:
                results[tc.name] = None
                print(f"[API] FAIL {tc.name!r} -> {exc}")
    return results


def fetch_all_mh_outputs(
    cases: list[OANTestCase],
    *,
    base_url: str,
    token: str,
    max_workers: int | None = None,
) -> dict[str, str | None]:
    """Fetch outputs from the MH OAN endpoint for all test cases."""
    results: dict[str, str | None] = {}
    worker_count = max_workers or MAX_WORKERS
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(_call_mh_api, tc, base_url, token): tc for tc in cases}
        for future in as_completed(futures):
            tc = futures[future]
            try:
                name, output = future.result()
                results[name] = output
                print(f"[MH-API] OK {name!r} -> {len(output or '')} chars")
            except Exception as exc:
                results[tc.name] = None
                print(f"[MH-API] FAIL {tc.name!r} -> {exc}")
    return results
