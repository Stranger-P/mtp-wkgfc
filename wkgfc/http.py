"""Dependency-free HTTP helpers with retries and polite rate-limit handling."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request


class HTTPError(RuntimeError):
    pass


def get_json(url: str, params: dict | None = None, headers: dict | None = None,
             timeout: int = 60, retries: int = 4) -> dict:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return _request(urllib.request.Request(url, headers=headers or {}), timeout, retries)


def post_json(url: str, payload: dict, headers: dict | None = None,
              timeout: int = 60, retries: int = 4) -> dict:
    body = json.dumps(payload).encode()
    h = {"Content-Type": "application/json", **(headers or {})}
    return _request(urllib.request.Request(url, data=body, headers=h, method="POST"), timeout, retries)


def get_text(url: str, headers: dict | None = None, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def _request(req, timeout, retries):
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:  # rate limited: respect Retry-After
                wait = int(e.headers.get("Retry-After", "5") or 5)
                time.sleep(min(wait, 60))
                continue
            if e.code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise HTTPError(f"HTTP {e.code} for {req.full_url[:200]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise HTTPError(f"request failed after {retries} tries: {last}")
