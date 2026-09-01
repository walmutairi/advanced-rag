"""Thin, dependency-free client for the local Ollama server.

Only three things are needed from Ollama, and each has a sharp edge worth
handling once here rather than at every call site:

* ``chat``      — plain text generation.
* ``chat_json`` — generation constrained to a JSON schema. gemma4 is a
  thinking model, so raw output can carry reasoning preamble and fenced code
  blocks; the parser strips both before decoding.
* ``embed``     — batch embeddings, L2-normalised so a dot product is cosine
  similarity.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterable

import requests

import config


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns something unusable."""


# --------------------------------------------------------------------------
# JSON salvage helpers
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_json(raw: str) -> Any:
    """Pull the first well-formed JSON value out of a model response.

    Tries, in order: the whole string, any fenced block, then a brace/bracket
    scan that respects string literals and escapes. The scan matters because
    thinking models like to narrate around their JSON.
    """
    if not raw or not raw.strip():
        raise OllamaError("model returned an empty response")

    text = _THINK_RE.sub("", raw).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for block in _FENCE_RE.findall(text):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    raise OllamaError(f"could not parse JSON from model output: {text[:400]}")


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.model = model or config.ANSWER_MODEL
        self.embed_model = embed_model or config.EMBED_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        self._session = requests.Session()

    # -- infrastructure ----------------------------------------------------

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        url = f"{self.host}{path}"
        last_error: Exception | None = None
        wait = timeout if timeout is not None else self.timeout
        retries = config.OLLAMA_MAX_RETRIES if max_retries is None else max(1, max_retries)

        for attempt in range(retries):
            try:
                response = self._session.post(url, json=payload, timeout=wait)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        raise OllamaError(
            f"Ollama request to {url} failed after "
            f"{retries} attempts: {last_error}"
        ) from last_error

    def health(self) -> tuple[bool, str]:
        """Check the server is up and the configured models are pulled."""
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            names = {m.get("name", "") for m in response.json().get("models", [])}
        except requests.RequestException as exc:
            return False, f"Cannot reach Ollama at {self.host}: {exc}"

        def _present(model: str) -> bool:
            # "gemma4:31b" should also match a bare "gemma4" listing.
            return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)

        missing = [m for m in (self.model, self.embed_model) if not _present(m)]
        if missing:
            return False, "Missing Ollama models: " + ", ".join(
                f"{m} (run `ollama pull {m}`)" for m in missing
            )
        return True, f"Ollama ready — {self.model} + {self.embed_model}"

    # -- generation --------------------------------------------------------

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        think: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {
                "temperature": (
                    config.LLM_TEMPERATURE if temperature is None else temperature
                ),
                "num_ctx": num_ctx or config.LLM_NUM_CTX,
            },
        }
        data = self._post("/api/chat", payload)
        return (data.get("message") or {}).get("content", "").strip()

    def chat_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        think: bool = False,
    ):
        """Yield content tokens as Ollama generates them."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": {
                "temperature": (
                    config.LLM_TEMPERATURE if temperature is None else temperature
                ),
                "num_ctx": num_ctx or config.LLM_NUM_CTX,
            },
        }
        url = f"{self.host}/api/chat"
        try:
            response = self._session.post(
                url, json=payload, stream=True, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama stream to {url} failed: {exc}") from exc

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            message = data.get("message") or {}
            content = message.get("content") or ""
            if content:
                yield content
            if data.get("done"):
                break

    def chat_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        retries: int = 2,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> Any:
        """Generate and decode JSON, retrying with a blunter nudge on failure.

        Passing ``schema`` uses Ollama's structured-output mode, which
        constrains decoding to valid JSON rather than merely asking for it.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema if schema else "json",
            "options": {
                "temperature": (
                    config.LLM_TEMPERATURE if temperature is None else temperature
                ),
                "num_ctx": num_ctx or config.LLM_NUM_CTX,
            },
        }

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                data = self._post(
                    "/api/chat",
                    payload,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                content = (data.get("message") or {}).get("content", "")
                return _extract_json(content)
            except OllamaError as exc:
                last_error = exc
                if attempt < retries:
                    payload["messages"] = messages + [
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply was not valid JSON. Reply with "
                                "the JSON value only — no prose, no code fences."
                            ),
                        }
                    ]
                    payload["options"]["temperature"] = 0.0

        raise OllamaError(f"structured generation failed: {last_error}")

    # -- embeddings --------------------------------------------------------

    def embed(self, texts: Iterable[str], batch_size: int = 16) -> list[list[float]]:
        """Embed texts, returning L2-normalised vectors in input order."""
        items = [t if t and t.strip() else " " for t in texts]
        if not items:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            data = self._post(
                "/api/embed", {"model": self.embed_model, "input": batch}
            )
            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise OllamaError(
                    f"embedding endpoint returned {len(embeddings or [])} vectors "
                    f"for {len(batch)} inputs"
                )
            vectors.extend(_normalise(v) for v in embeddings)

        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _normalise(vector: list[float]) -> list[float]:
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return list(vector)
    return [v / norm for v in vector]


#: Process-wide default client for query-time generation and embeddings.
_default_client: OllamaClient | None = None
_extract_client: OllamaClient | None = None


def get_client() -> OllamaClient:
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient(model=config.ANSWER_MODEL)
    return _default_client


def get_extract_client() -> OllamaClient:
    """Client for ingest-time extraction; shared when extract == answer model."""
    global _extract_client
    if config.EXTRACT_MODEL == config.ANSWER_MODEL:
        return get_client()
    if _extract_client is None:
        _extract_client = OllamaClient(model=config.EXTRACT_MODEL)
    return _extract_client


__all__ = ["OllamaClient", "OllamaError", "get_client", "get_extract_client", "_extract_json"]
