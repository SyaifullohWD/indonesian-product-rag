from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

DEFAULT_REWRITE_MODEL = "qwen3.5-9b-64k:latest"
NONE_REPLY = "NONE"
# Bump when the prompt changes so cached rewrites from an old prompt are not reused.
PROMPT_VERSION = "2"

SYSTEM_PROMPT = (
    "You turn an Indonesian social media post into a short shopping need. Reply with ONE line in "
    "Indonesian, max 12 words: the product type and the benefit the person needs, e.g. 'lampu "
    "meja yang terang untuk membaca di malam hari'. If the post does not ask for or need a "
    "product someone can buy (for example it wants a professional, a place, a service, or just "
    "shares a story), reply exactly: NONE. Do not name brands. No explanation."
)

_QUOTES = "\"'“”‘’"


class Rewriter(Protocol):
    def rewrite(self, text: str) -> str:  # pragma: no cover - protocol
        ...


def clean_reply(raw: str) -> str:
    """Keep the first line, drop quotes and trailing punctuation; empty means NONE."""
    text = (raw or "").strip()
    if not text:
        return NONE_REPLY
    line = text.splitlines()[0].strip().strip(_QUOTES).strip()
    line = line.rstrip(".,;:!?…").strip().strip(_QUOTES).strip()
    if not line or line.upper() == NONE_REPLY:
        return NONE_REPLY
    return line


@dataclass
class OllamaRewriter:
    base_url: str
    model: str = field(
        default_factory=lambda: os.environ.get("RAG_REWRITE_MODEL", DEFAULT_REWRITE_MODEL)
    )

    def rewrite(self, text: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }
        with httpx.Client(base_url=self.base_url, timeout=120.0) as client:
            resp = client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("invalid rewrite response")
        return clean_reply(content)


class CachedRewriter:
    """JSON-file cache so a repeated post never calls the model twice."""

    def __init__(self, inner: Rewriter, path: str | Path = ".cache/rewrites.json"):
        self.inner = inner
        self.path = Path(path)
        self._entries: dict[str, str] | None = None

    def _key(self, text: str) -> str:
        model = getattr(self.inner, "model", "")
        raw = f"{model}\n{PROMPT_VERSION}\n{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries or {}, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

    def rewrite(self, text: str) -> str:
        if self._entries is None:
            self._entries = self._load()
        key = self._key(text)
        if key in self._entries:
            return self._entries[key]
        need = self.inner.rewrite(text)
        self._entries[key] = need
        self._save()
        return need
