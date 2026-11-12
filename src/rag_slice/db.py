from __future__ import annotations

from dataclasses import dataclass

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

SCHEMA_DEFAULT = "rag_slice"


@dataclass(frozen=True)
class SearchResult:
    product_id: str
    cosine_similarity: float
    rrf_score: float


def _ident(schema: str) -> sql.Identifier:
    return sql.Identifier(schema)


def init_db(conn: psycopg.Connection, schema: str = SCHEMA_DEFAULT, dim: int = 1024) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(_ident(schema)))
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {}.catalog_meta (
              id smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
              embedding_model text NOT NULL,
              embedding_dim integer NOT NULL,
              updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        ).format(_ident(schema)))
        cur.execute(sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {}.products (
              product_id text PRIMARY KEY,
              content text NOT NULL,
              content_hash text NOT NULL,
              content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
              embedding public.vector({dim}) NOT NULL,
              embedding_model text NOT NULL,
              embedding_dim integer NOT NULL,
              updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        ).format(_ident(schema), dim=sql.SQL(str(int(dim)))))
        cur.execute(sql.SQL(
            "CREATE INDEX IF NOT EXISTS products_embedding_hnsw ON {}.products USING hnsw (embedding public.vector_cosine_ops)"
        ).format(_ident(schema)))
        cur.execute(sql.SQL(
            "CREATE INDEX IF NOT EXISTS products_content_tsv_gin ON {}.products USING gin (content_tsv)"
        ).format(_ident(schema)))
        # search_path is empty at runtime, so every operator/type is schema-qualified.
        func_sql = sql.SQL(
            r"""
CREATE OR REPLACE FUNCTION {schema}.hybrid_search(
  query_embedding public.vector({dim}),
  query_text text,
  query_model text,
  query_dim integer,
  top_k integer DEFAULT 3
)
RETURNS TABLE(product_id text, cosine_similarity double precision, rrf_score double precision)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
DECLARE
  meta_model text;
  meta_dim integer;
  tsq tsquery;
  k_rrf constant double precision := 60.0;
BEGIN
  IF top_k IS NULL OR top_k < 1 OR top_k > 3 THEN
    RAISE EXCEPTION 'top_k must be an integer between 1 and 3: got %', top_k
      USING ERRCODE = '22023';
  END IF;

  SELECT m.embedding_model, m.embedding_dim INTO meta_model, meta_dim
  FROM {schema}.catalog_meta m WHERE m.id = 1;
  IF meta_model IS NULL THEN
    RETURN;
  END IF;
  -- Enforce the real vector width, not the caller-supplied query_dim metadata.
  IF public.vector_dims(query_embedding) IS DISTINCT FROM meta_dim THEN
    RAISE EXCEPTION 'query embedding dimension mismatch: catalog=%, query=%',
      meta_dim, public.vector_dims(query_embedding)
      USING ERRCODE = '22023';
  END IF;
  IF meta_model IS DISTINCT FROM query_model OR meta_dim IS DISTINCT FROM query_dim THEN
    RAISE EXCEPTION 'model/dimension mismatch: catalog=%,% query=%,%', meta_model, meta_dim, query_model, query_dim
      USING ERRCODE = '22023';
  END IF;

  -- OR every token (an implicit AND would drop partial matches); no tokens -> no keyword leg.
  SELECT to_tsquery('simple', string_agg(quote_literal(tok), ' | '))
    INTO tsq
  FROM unnest(regexp_split_to_array(lower(coalesce(query_text, '')), '[^[:alnum:]]+')) AS tok
  WHERE tok <> '';

  PERFORM set_config('hnsw.iterative_scan', 'relaxed_order', true);

  RETURN QUERY
  WITH vec AS (
    SELECT p.product_id,
           1 - (p.embedding OPERATOR(public.<=>) query_embedding) AS sim,
           row_number() OVER (ORDER BY p.embedding OPERATOR(public.<=>) query_embedding) AS rnk
    FROM {schema}.products p
    ORDER BY p.embedding OPERATOR(public.<=>) query_embedding
    LIMIT 20
  ),
  kw AS (
    SELECT p.product_id,
           1 - (p.embedding OPERATOR(public.<=>) query_embedding) AS sim,
           ts_rank_cd(p.content_tsv, tsq) AS kw_rank,
           row_number() OVER (ORDER BY ts_rank_cd(p.content_tsv, tsq) DESC, p.product_id) AS rnk
    FROM {schema}.products p
    WHERE tsq IS NOT NULL AND p.content_tsv @@ tsq
    ORDER BY ts_rank_cd(p.content_tsv, tsq) DESC, p.product_id
    LIMIT 20
  ),
  fused AS (
    SELECT COALESCE(v.product_id, k.product_id) AS pid,
           COALESCE(v.sim, k.sim) AS sim,
           (COALESCE(1.0 / (k_rrf + v.rnk), 0.0) + COALESCE(1.0 / (k_rrf + k.rnk), 0.0)) AS rrf
    FROM vec v
    FULL OUTER JOIN kw k ON v.product_id = k.product_id
  )
  SELECT f.pid AS product_id,
         f.sim::double precision AS cosine_similarity,
         f.rrf::double precision AS rrf_score
  FROM fused f
  ORDER BY f.rrf DESC, f.sim DESC, f.pid
  LIMIT top_k;
END;
$$;
            """
        ).format(schema=_ident(schema), dim=sql.SQL(str(int(dim))))
        cur.execute(func_sql)
        cur.execute(sql.SQL(
            "REVOKE ALL ON FUNCTION {}.hybrid_search(public.vector, text, text, integer, integer) FROM PUBLIC"
        ).format(_ident(schema)))
    conn.commit()


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url, row_factory=dict_row)
