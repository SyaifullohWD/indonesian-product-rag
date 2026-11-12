from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import httpx


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - protocol
        ...


def validate_vector(vec: list[float], dim: int, label: str = "embedding") -> list[float]:
    if not isinstance(vec, (list, tuple)):
        raise ValueError(f"{label} must be a list of numbers")
    if len(vec) != dim:
        raise ValueError(f"{label} dim mismatch: got {len(vec)} expected {dim}")
    out: list[float] = []
    for x in vec:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"{label} contains a non-numeric value")
        f = float(x)
        if not math.isfinite(f):
            raise ValueError(f"{label} contains a nonfinite value")
        out.append(f)
    if not any(f != 0.0 for f in out):
        raise ValueError(f"{label} must not be a zero vector")
    return out


@dataclass
class OllamaEmbedding:
    base_url: str
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        with httpx.Client(base_url=self.base_url, timeout=120.0) as client:
            for t in texts:
                resp = client.post("/api/embeddings", json={"model": self.model, "prompt": t})
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding")
                if not isinstance(emb, list):
                    raise ValueError("invalid embedding response")
                results.append(validate_vector(emb, self.dim))
        return results
