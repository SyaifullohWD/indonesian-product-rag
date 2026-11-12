from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    database_url: str = os.environ.get(
        "RAG_DATABASE_URL",
        "postgresql:///product_rag?host=localhost&port=5433",
    )
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    embedding_dim: int = int(os.environ.get("EMBEDDING_DIM", "1024"))
    schema_name: str = os.environ.get("RAG_SCHEMA", "rag_slice")
    guard_threshold: float = float(os.environ.get("RAG_GUARD_THRESHOLD", "0.55"))


def get_config() -> Config:
    return Config()
