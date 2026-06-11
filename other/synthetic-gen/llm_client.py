from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

from google import genai
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)


class GeminiRateLimitError(RuntimeError):
    pass


class GeminiTransientError(RuntimeError):
    pass


class GeminiAuthOrAccessError(RuntimeError):
    pass


def _classify_exception(exc: Exception) -> Exception:
    msg = str(exc).lower()
    # Best-effort classification across SDK/HTTP error styles.
    if "429" in msg or "rate" in msg and "limit" in msg:
        return GeminiRateLimitError(str(exc))
    if "503" in msg or "timeout" in msg or "temporarily" in msg or "unavailable" in msg:
        return GeminiTransientError(str(exc))
    if "403" in msg or "permission" in msg or "api key not valid" in msg:
        return GeminiAuthOrAccessError(str(exc))
    return exc


class AsyncTokenBucket:
    """
    Simple async token bucket for request-per-minute throttling.
    Not perfectly precise, but stable and safe under concurrency.
    """

    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        # Hard-rate limiter with no burst: at most 1 token in bucket.
        self.capacity = 1.0
        self.tokens = 1.0
        self.refill_per_sec = self.rpm / 60.0
        self.updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.updated_at
                self.updated_at = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                # Need to wait for enough tokens.
                missing = 1.0 - self.tokens
                wait_s = max(0.05, missing / self.refill_per_sec)
            await asyncio.sleep(wait_s)


@dataclass(frozen=True)
class LLMConfig:
    api_key_env: str = "GEMINI_API_KEY"
    api_keys_env: str = "GEMINI_API_KEYS"
    # Prefer the Gemma 27B instruction-tuned variant that is commonly available.
    model: str = "gemma-3-27b-it"
    rpm: int = 60
    max_concurrency: int = 8
    timeout_s: int = 60


def _on_retry(retry_state: RetryCallState) -> None:
    # Keep minimal to avoid log spam; caller logs attempts.
    _ = retry_state


class GeminiNarrationClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig(
            model=os.getenv("GEMINI_MODEL", "").strip() or "AUTO",
            rpm=int(os.getenv("GEMINI_RPM", "60")),
            max_concurrency=int(os.getenv("GEMINI_MAX_CONCURRENCY", "8")),
            timeout_s=int(os.getenv("GEMINI_TIMEOUT_S", "60")),
        )
        api_keys = self._load_api_keys()
        if not api_keys:
            raise RuntimeError(
                f"Missing API keys. Set {self.config.api_keys_env} (comma-separated) or {self.config.api_key_env}."
            )

        self._clients = [genai.Client(api_key=k) for k in api_keys]
        resolved = self._resolve_model(self.config.model, self._clients[0])
        object.__setattr__(self.config, "model", resolved)
        per_key_conc = max(1, self.config.max_concurrency // len(self._clients))
        self._semas = [asyncio.Semaphore(per_key_conc) for _ in self._clients]
        self._buckets = [AsyncTokenBucket(self.config.rpm) for _ in self._clients]
        self._rr_lock = asyncio.Lock()
        self._rr_index = 0
        self._key_cooldown_until: List[float] = [0.0 for _ in self._clients]
        self._blocked_models_per_key: List[set[str]] = [set() for _ in self._clients]

    def _load_api_keys(self) -> List[str]:
        many = os.getenv(self.config.api_keys_env, "").strip()
        if many:
            return [x.strip() for x in many.split(",") if x.strip()]
        single = os.getenv(self.config.api_key_env, "").strip()
        return [single] if single else []

    def _resolve_model(self, configured: str, client) -> str:
        """
        Resolve a usable model name.

        - If GEMINI_MODEL is set and not AUTO, use it.
        - Otherwise, pick the closest available Gemma 27B-ish model from the API.
        """
        configured = (configured or "").strip()
        if configured and configured.upper() != "AUTO":
            return configured

        try:
            models = list(client.models.list())
        except Exception:
            # If we can't list models (e.g. 403), fall back to the most-likely
            # Gemma 27B instruction-tuned model variant.
            return "gemma-3-27b-it"

        names: list[str] = []
        for m in models:
            name = getattr(m, "name", None) or getattr(m, "model", None) or ""
            if isinstance(name, str) and name:
                names.append(name)

        # Prefer any Gemma 27B instruction-tuned variant.
        preferred = [
            n
            for n in names
            if "gemma" in n.lower()
            and ("27" in n or "27b" in n.lower())
            and ("it" in n.lower() or "instr" in n.lower() or "instruct" in n.lower())
        ]
        if preferred:
            return preferred[0]

        gemma_any = [n for n in names if "gemma" in n.lower()]
        if gemma_any:
            return gemma_any[0]

        # Closest available general model if no Gemma is listed.
        for fallback in ["gemini-2.0-pro", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]:
            if any(fallback in n for n in names):
                return fallback
        return configured or "gemma-3-27b-it"

    @retry(
        reraise=True,
        stop=stop_after_attempt(8),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type((GeminiRateLimitError, GeminiTransientError)),
        before_sleep=_on_retry,
    )
    async def generate_one(self, prompt: str) -> str:
        """
        Generate EXACTLY ONE narration string (no batching).
        """
        client_idx = await self._next_client_index()
        async with self._semas[client_idx]:
            await self._buckets[client_idx].acquire()
            try:
                # The SDK call is sync; run it off the event loop.
                text = await asyncio.to_thread(self._generate_sync, self._clients[client_idx], prompt, client_idx)
                return text
            except Exception as e:  # noqa: BLE001
                classified = _classify_exception(e)
                raise classified from e

    async def _next_client_index(self) -> int:
        async with self._rr_lock:
            now = time.monotonic()
            n = len(self._clients)
            for _ in range(n):
                idx = self._rr_index
                self._rr_index = (self._rr_index + 1) % n
                if self._key_cooldown_until[idx] <= now:
                    return idx
            idx = self._rr_index
            self._rr_index = (self._rr_index + 1) % n
            return idx

    def _generate_sync(self, client, prompt: str, client_idx: int) -> str:
        # Light jitter to avoid burst synchronization under high concurrency.
        time.sleep(random.uniform(0.0, 0.15))
        model_candidates = [
            self.config.model,
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]
        model_candidates = [m for m in model_candidates if m not in self._blocked_models_per_key[client_idx]]
        if not model_candidates:
            model_candidates = ["gemini-1.5-flash"]
        last_exc: Optional[Exception] = None
        for model_name in model_candidates:
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    # Keep output deterministic enough to pass validation, but still diverse.
                    config={
                        "temperature": 0.9,
                        "top_p": 0.9,
                        "max_output_tokens": 160,
                    },
                )
                text = getattr(resp, "text", None)
                if not text:
                    raise GeminiTransientError("Empty response text from model")
                return str(text).strip()
            except Exception as e:  # noqa: BLE001
                last_exc = e
                msg = str(e).lower()
                # Try next model only when failure is likely model-access related.
                if "403" in msg or "404" in msg or "not found" in msg:
                    self._blocked_models_per_key[client_idx].add(model_name)
                    continue
                if "429" in msg or "rate" in msg:
                    self._key_cooldown_until[client_idx] = time.monotonic() + 12.0
                    raise GeminiRateLimitError(str(e))
                raise
        if last_exc is not None:
            raise last_exc
        raise GeminiTransientError("Model call failed for unknown reason")

