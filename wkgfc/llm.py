"""LLM access. One interface (`LLM.complete`) for OpenAI-compatible APIs and an offline mock.

All calls are cached and counted, so every experiment can report LLM calls and tokens per claim
(the paper's cost analysis, Sec. 5.6)."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .cache import Cache, NullCache


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_purpose: dict = field(default_factory=dict)
    by_model: dict = field(default_factory=dict)

    def add(self, purpose: str, p: int, c: int, cached: bool, model: str = "") -> None:
        self.calls += 1
        self.cached_calls += int(cached)
        self.prompt_tokens += p
        self.completion_tokens += c
        self.by_purpose[purpose] = self.by_purpose.get(purpose, 0) + 1
        if model:
            self.by_model[model] = self.by_model.get(model, 0) + 1

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "by_purpose": dict(self.by_purpose),
            "by_model": dict(self.by_model),
        }


class LLM:
    def __init__(self, cfg, cache: Cache | None = None):
        self.cfg = cfg
        self.model = cfg.llm_model
        self.cache = cache or NullCache()
        self.usage = Usage()
        self._client = None
        self._usage_lock = threading.Lock()
        self._tl = threading.local()  # per-thread: which model answered the latest call

    @property
    def last_model(self) -> str:
        return getattr(self._tl, "model", self.model)

    @last_model.setter
    def last_model(self, m: str) -> None:
        self._tl.model = m

    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import: tests run without the package

            kwargs = {}
            if self.cfg.openai_base_url:
                kwargs["base_url"] = self.cfg.openai_base_url
            if not os.environ.get("OPENAI_API_KEY") and not self.cfg.openai_base_url:
                raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
            self._client = OpenAI(**kwargs)
        return self._client

    def _models_to_try(self) -> list[str]:
        """Primary model first, then LLM_FALLBACK_MODELS (in order, no duplicates)."""
        out = []
        for m in [self.model, *(self.cfg.llm_fallback_models or [])]:
            if m and m not in out:
                out.append(m)
        return out

    def _call_api(self, system: str, prompt: str, json_mode: bool, max_tokens: int) -> tuple[str, int, int]:
        client = self._get_client()
        base = dict(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=self.cfg.llm_temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            base["response_format"] = {"type": "json_object"}
        models = self._models_to_try()
        errors = []
        for i, model in enumerate(models):
            has_next = i < len(models) - 1
            name = model
            attempts = 2 if has_next else 6  # with a fallback available, give up on a busy model sooner
            for attempt in range(attempts):
                try:
                    resp = client.chat.completions.create(model=name, **base)
                    text = resp.choices[0].message.content or ""
                    u = resp.usage
                    self.last_model = model
                    if model != self.model and self.cfg.verbose:
                        print(f"[llm] fell back to {model}")
                    return text, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0
                except Exception as e:
                    status = getattr(e, "status_code", None)
                    body = getattr(e, "body", None) or getattr(getattr(e, "response", None), "text", "")
                    if status == 404 and isinstance(self, GeminiLLM) and not name.startswith("models/"):
                        name = "models/" + name  # some Gemini endpoints want the prefix
                        continue
                    if status in (400, 401, 403):  # bad request / bad key: same for every model
                        raise RuntimeError(f"{type(self).__name__} call failed (HTTP {status}, not retrying): "
                                           f"{e} | details: {body}") from e
                    errors.append(f"{model}: HTTP {status} {str(body)[:160]}")
                    if status == 404:
                        break  # model not available: go straight to the next one
                    if attempt < attempts - 1:  # 429 rate limit / 503 overloaded / transient
                        time.sleep(min(5 * 2 ** attempt, 60) if status in (429, 503) else min(2 ** attempt, 30))
        hint = " Run `python -m wkgfc.run models` to list valid names." if any("HTTP 404" in x for x in errors) else ""
        raise RuntimeError("LLM call failed for all models: " + " || ".join(errors[-len(models):]) + hint)

    # ------------------------------------------------------------------
    def complete(self, prompt: str, *, system: str = "You are a careful fact-checking assistant.",
                 json_mode: bool = False, purpose: str = "other", max_tokens: int | None = None) -> str:
        max_tokens = max_tokens or self.cfg.llm_max_tokens
        key = Cache.make_key("llm", self.model, self.cfg.llm_temperature, system, prompt, json_mode, max_tokens)
        hit = self.cache.get(key)
        if hit is not None:
            with self._usage_lock:
                self.usage.add(purpose, hit["p"], hit["c"], cached=True, model=hit.get("model", self.model))
            return hit["text"]
        self.last_model = self.model
        text, p, c = self._call_api(system, prompt, json_mode, max_tokens)
        self.cache.set(key, {"text": text, "p": p, "c": c, "model": self.last_model})
        with self._usage_lock:
            self.usage.add(purpose, p, c, cached=False, model=self.last_model)
        return text

    def complete_json(self, prompt: str, *, purpose: str = "other", default=None, **kw):
        text = self.complete(prompt, json_mode=True, purpose=purpose, **kw)
        parsed = parse_json(text)
        return default if parsed is None else parsed


class BedrockLLM(LLM):
    """Claude (or any chat model) on Amazon Bedrock via the Converse API.

    Credentials (any one works, boto3 picks them up automatically):
      - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (IAM user keys), or
      - AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key), or
      - an AWS profile configured with `aws configure`.
    Region: AWS_REGION (e.g. us-east-1). Model: an inference-profile ID such as
    us.anthropic.claude-haiku-4-5-20251001-v1:0 (Claude 4.5 models need the profile ID, not the base ID)."""

    def _get_client(self):
        if self._client is None:
            import boto3  # lazy import

            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
            self._client = boto3.client("bedrock-runtime", region_name=region)
        return self._client

    def _call_api(self, system, prompt, json_mode, max_tokens):
        client = self._get_client()
        if json_mode:
            prompt += "\n\nRespond with a single valid JSON object and nothing else."
        models = self._models_to_try()
        errors = []
        for i, model in enumerate(models):
            attempts = 2 if i < len(models) - 1 else 6  # with a fallback available, give up sooner
            for attempt in range(attempts):
                try:
                    resp = client.converse(
                        modelId=model,
                        system=[{"text": system}],
                        messages=[{"role": "user", "content": [{"text": prompt}]}],
                        inferenceConfig={"maxTokens": max_tokens, "temperature": self.cfg.llm_temperature},
                    )
                    parts = resp["output"]["message"]["content"]
                    text = "".join(p.get("text", "") for p in parts)
                    u = resp.get("usage", {})
                    self.last_model = model
                    return text, u.get("inputTokens", 0), u.get("outputTokens", 0)
                except Exception as e:
                    code = type(e).__name__ + str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
                    if any(k in code for k in ("AccessDenied", "UnrecognizedClient", "NoCredentials")):
                        raise RuntimeError(f"Bedrock call failed (credentials/permissions, not retrying): {e}") from e
                    errors.append(f"{model}: {code} {str(e)[:160]}")
                    if any(k in code for k in ("ResourceNotFound", "ValidationException")):
                        break  # wrong/unavailable model ID: try the next model
                    if attempt < attempts - 1:  # throttling / overloaded / transient
                        time.sleep(min(5 * 2 ** attempt, 60))
        raise RuntimeError("Bedrock call failed for all models: " + " || ".join(errors[-len(models):]))


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiLLM(LLM):
    """Google Gemini via its OpenAI-compatible endpoint. Free tier from Google AI Studio.

    Key: GEMINI_API_KEY. Model: e.g. gemini-3.8-flash (check AI Studio for current names).
    Use for development and pipeline testing only; do not mix its numbers with Claude results."""

    def _get_client(self):
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env (Google AI Studio key).")
            from openai import OpenAI  # lazy import

            self._client = OpenAI(api_key=key, base_url=GEMINI_BASE_URL)
        return self._client


class MockLLM(LLM):
    """Offline LLM for tests. `responder(prompt, purpose) -> str` decides the reply."""

    def __init__(self, cfg, responder: Callable[[str, str], str]):
        super().__init__(cfg, NullCache())
        self.responder = responder
        self.log: list[tuple[str, str]] = []

    def _call_api(self, system, prompt, json_mode, max_tokens):
        raise AssertionError("unused")

    def complete(self, prompt, *, system="", json_mode=False, purpose="other", max_tokens=None):
        self.log.append((purpose, prompt))
        out = self.responder(prompt, purpose)
        with self._usage_lock:
            self.usage.add(purpose, len(prompt) // 4, len(out) // 4, cached=False)
        return out


def make_llm(cfg, cache: Cache | None = None, responder=None) -> LLM:
    if cfg.llm_backend == "mock":
        if responder is None:
            raise ValueError("mock backend needs a responder")
        return MockLLM(cfg, responder)
    if cfg.llm_backend == "bedrock":
        return BedrockLLM(cfg, cache)
    if cfg.llm_backend == "gemini":
        return GeminiLLM(cfg, cache)
    return LLM(cfg, cache)


_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


def parse_json(text: str):
    """Parse JSON from an LLM reply, tolerating code fences and surrounding prose."""
    if text is None:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?", "", t).rstrip("`").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(t)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
