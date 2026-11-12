from __future__ import annotations


class FakeEmbedder:
    """Deterministic pseudo-vectors derived from the input text."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for t in texts:
            h = sum(ord(c) for c in t)
            out.append([float((h + i * 31) % 100) / 100.0 for i in range(self.dim)])
        return out


class MappedEmbedder:
    """Explicit vectors per text; unmapped texts become zero vectors."""

    def __init__(self, dim: int = 1024, mapping: dict[str, list[float]] | None = None):
        self.dim = dim
        self.mapping = mapping or {}
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [list(self.mapping.get(t, [0.0] * self.dim)) for t in texts]


class OneHotEmbedder(MappedEmbedder):
    """Maps known texts to orthogonal one-hot vectors."""

    def __init__(self, dim: int = 1024, mapping: dict[str, int] | None = None):
        self.indices = mapping or {}
        super().__init__(dim=dim, mapping={})

    def vec(self, idx: int) -> list[float]:
        v = [0.0] * self.dim
        if 0 <= idx < self.dim:
            v[idx] = 1.0
        return v

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for t in texts:
            idx = self.indices.get(t)
            out.append(self.vec(idx) if idx is not None else [0.0] * self.dim)
        return out
