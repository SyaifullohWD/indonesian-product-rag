from __future__ import annotations

import psycopg
from psycopg import sql

from .db import SearchResult
from .embedding import EmbeddingClient, validate_vector
from .eval import validate_threshold
from .rewrite import NONE_REPLY, Rewriter

REWRITE_MODES = ("off", "replace", "concat")


class GuardedNoMatch(Exception):
    pass


def resolve_query(
    text: str, rewriter: Rewriter | None, mode: str
) -> tuple[str | None, str | None]:
    """Return (query_text, rewrite). query_text is None when the rewrite is NONE."""
    if mode not in REWRITE_MODES:
        raise ValueError(f"rewrite mode must be one of {', '.join(REWRITE_MODES)}")
    if mode == "off":
        return text, None
    if rewriter is None:
        raise ValueError("a rewriter is required when rewrite mode is not off")
    need = rewriter.rewrite(text)
    if need == NONE_REPLY:
        return None, NONE_REPLY
    if mode == "replace":
        return need, need
    return f"{text}\n{need}", need


def hybrid_search(
    conn: psycopg.Connection,
    schema: str,
    query_text: str,
    embedder: EmbeddingClient,
    model: str,
    dim: int,
    top_k: int = 3,
) -> list[SearchResult]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 3:
        raise ValueError("top_k must be an integer between 1 and 3")
    vecs = embedder.embed([query_text])
    if len(vecs) != 1:
        raise ValueError("embedder must return one vector for query")
    qvec = validate_vector(vecs[0], dim, label="query embedding")
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT product_id, cosine_similarity, rrf_score FROM {}.hybrid_search(%s::public.vector, %s, %s, %s, %s)"
            ).format(sql.Identifier(schema)),
            (qvec, query_text, model, dim, top_k),
        )
        rows = cur.fetchall()
    results = [
        SearchResult(
            product_id=r["product_id"],
            cosine_similarity=float(r["cosine_similarity"]),
            rrf_score=float(r["rrf_score"]),
        )
        for r in rows
    ]
    return results


def guard_results(results: list[SearchResult], threshold: float) -> tuple[bool, list[SearchResult]]:
    """Return (ok, results). ok False means no_match (guard triggered).
    Guard uses MAX cosine of returned top3 strictly below threshold; empty gives no_match.
    """
    threshold = validate_threshold(threshold)
    if not results:
        return False, []
    max_cos = max(r.cosine_similarity for r in results)
    if max_cos < threshold:
        return False, []
    return True, results
