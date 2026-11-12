from __future__ import annotations

from dataclasses import dataclass

import psycopg
from psycopg import sql

from .chunking import ProductChunk, read_csv_products
from .embedding import EmbeddingClient, validate_vector


@dataclass
class IngestStats:
    total: int
    inserted: int
    updated: int
    unchanged: int
    embedded: int


def ensure_catalog_identity(
    conn: psycopg.Connection, schema: str, model: str, dim: int
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT embedding_model, embedding_dim FROM {}.catalog_meta WHERE id = 1").format(
                sql.Identifier(schema)
            )
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {}.catalog_meta (id, embedding_model, embedding_dim) VALUES (1, %s, %s)"
                ).format(sql.Identifier(schema)),
                (model, dim),
            )
        else:
            if row["embedding_model"] != model or int(row["embedding_dim"]) != int(dim):
                raise ValueError(
                    f"catalog model/dimension mismatch: existing={row['embedding_model']}/{row['embedding_dim']} "
                    f"requested={model}/{dim}"
                )


def import_catalog(
    conn: psycopg.Connection,
    schema: str,
    csv_path: str,
    embedder: EmbeddingClient,
    model: str,
    dim: int,
) -> IngestStats:
    chunks = read_csv_products(csv_path)
    total = len(chunks)
    # transactional ingestion
    with conn.transaction():
        ensure_catalog_identity(conn, schema, model, dim)
        existing: dict[str, tuple[str, str]] = {}
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT product_id, content_hash, embedding_model, embedding_dim FROM {}.products").format(
                    sql.Identifier(schema)
                )
            )
            for r in cur.fetchall():
                if r["embedding_model"] != model or int(r["embedding_dim"]) != int(dim):
                    raise ValueError(
                        f"existing row model/dimension mismatch for {r['product_id']}: "
                        f"{r['embedding_model']}/{r['embedding_dim']} != {model}/{dim}"
                    )
                existing[r["product_id"]] = (r["content_hash"], r["embedding_model"])

        to_embed: list[ProductChunk] = []
        for c in chunks:
            prev = existing.get(c.product_id)
            if prev and prev[0] == c.content_hash:
                continue
            to_embed.append(c)

        embeddings = embedder.embed([c.content for c in to_embed]) if to_embed else []
        if len(embeddings) != len(to_embed):
            raise ValueError("embedder returned wrong number of vectors")
        emb_iter = iter(embeddings)

        inserted = 0
        updated = 0
        unchanged = total - len(to_embed)
        with conn.cursor() as cur:
            for c in chunks:
                prev = existing.get(c.product_id)
                if prev and prev[0] == c.content_hash:
                    continue
                emb = next(emb_iter)
                emb = validate_vector(emb, dim)
                if prev is None:
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {}.products
                              (product_id, content, content_hash, embedding, embedding_model, embedding_dim, updated_at)
                            VALUES (%s, %s, %s, %s::vector, %s, %s, now())
                            """
                        ).format(sql.Identifier(schema)),
                        (c.product_id, c.content, c.content_hash, emb, model, dim),
                    )
                    inserted += 1
                else:
                    cur.execute(
                        sql.SQL(
                            """
                            UPDATE {}.products
                            SET content=%s, content_hash=%s, embedding=%s::vector,
                                embedding_model=%s, embedding_dim=%s, updated_at=now()
                            WHERE product_id=%s
                            """
                        ).format(sql.Identifier(schema)),
                        (c.content, c.content_hash, emb, model, dim, c.product_id),
                    )
                    updated += 1
    return IngestStats(
        total=total,
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        embedded=len(to_embed),
    )
