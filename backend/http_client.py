import httpx

from .config import load_config

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        cfg = load_config()
        max_conn = max(1, int(getattr(cfg, "http_max_connections", 32)))
        max_keepalive = max(
            1, int(getattr(cfg, "http_max_keepalive_connections", 16))
        )
        limits = httpx.Limits(
            max_connections=max_conn,
            max_keepalive_connections=min(max_keepalive, max_conn),
        )
        _client = httpx.AsyncClient(timeout=120.0, limits=limits)
    return _client


async def close_client() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
