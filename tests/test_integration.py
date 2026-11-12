from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from psycopg import sql

from rag_slice.chunking import read_csv_products
from rag_slice.config import get_config
from rag_slice.db import connect, init_db
from rag_slice.ingest import import_catalog
from rag_slice.search import guard_results, hybrid_search
from tests.support import FakeEmbedder, MappedEmbedder, OneHotEmbedder

CATALOG_HEADER = ["product_id", "name", "category", "description", "price_idr"]


def write_csv(path: Path, rows: list[dict], header: list[str]):
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_eval(path: Path, cases: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
        encoding="utf-8",
    )
    return path


def test_init_db_creates_schema_and_function(db_conn, test_schema):
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT routine_name FROM information_schema.routines WHERE routine_schema=%s AND routine_name='hybrid_search'",
            (test_schema,),
        )
        assert cur.fetchone() is not None
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname=%s", (test_schema,))
        tables = {r["tablename"] for r in cur.fetchall()}
        assert {"products", "catalog_meta"} <= tables


def test_import_and_content_hash_skip_uses_zero_embed_calls(db_conn, test_schema, tmp_path):
    csvp = tmp_path / "c.csv"
    write_csv(
        csvp,
        [
            {"product_id": "P1", "name": "Sabun", "category": "Wajah", "description": "wangi", "price_idr": "1000"},
            {"product_id": "P2", "name": "Sampo", "category": "Rambut", "description": "harum", "price_idr": "2000"},
        ],
        CATALOG_HEADER,
    )
    embedder = FakeEmbedder(dim=1024)
    stats = import_catalog(db_conn, test_schema, str(csvp), embedder, "m", 1024)
    assert stats.total == 2 and stats.inserted == 2 and stats.embedded == 2
    calls_after_first = len(embedder.calls)

    stats2 = import_catalog(db_conn, test_schema, str(csvp), embedder, "m", 1024)
    assert stats2.unchanged == 2 and stats2.embedded == 0
    assert len(embedder.calls) == calls_after_first, "unchanged rows must not call embed"

    write_csv(
        csvp,
        [
            {"product_id": "P1", "name": "Sabun", "category": "Wajah", "description": "wangi sekali", "price_idr": "1000"},
            {"product_id": "P2", "name": "Sampo", "category": "Rambut", "description": "harum", "price_idr": "2000"},
        ],
        CATALOG_HEADER,
    )
    stats3 = import_catalog(db_conn, test_schema, str(csvp), embedder, "m", 1024)
    assert stats3.updated == 1 and stats3.unchanged == 1 and stats3.embedded == 1
    assert len(embedder.calls) == calls_after_first + 1
    assert len(embedder.calls[-1]) == 1


def test_transactional_rollback_on_embed_failure(db_conn, test_schema, tmp_path):
    csvp = tmp_path / "c.csv"
    write_csv(
        csvp,
        [{"product_id": "P1", "name": "A", "category": "C", "description": "x", "price_idr": "1"}],
        CATALOG_HEADER,
    )

    class BadEmbedder:
        def embed(self, texts):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        import_catalog(db_conn, test_schema, str(csvp), BadEmbedder(), "m", 1024)
    with db_conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT count(*) c FROM {}.products").format(sql.Identifier(test_schema)))
        assert cur.fetchone()["c"] == 0
        cur.execute(sql.SQL("SELECT count(*) c FROM {}.catalog_meta").format(sql.Identifier(test_schema)))
        assert cur.fetchone()["c"] == 0


@pytest.mark.parametrize(
    "vec",
    [
        [0.0] * 1024,
        [float("nan")] + [0.0] * 1023,
        [float("inf")] + [0.0] * 1023,
        [1.0] * 3,
    ],
)
def test_import_rejects_bad_embeddings(db_conn, test_schema, tmp_path, vec):
    csvp = tmp_path / "c.csv"
    write_csv(
        csvp,
        [{"product_id": "P1", "name": "A", "category": "C", "description": "x", "price_idr": "1"}],
        CATALOG_HEADER,
    )

    class BadVecEmbedder:
        def embed(self, texts):
            return [vec for _ in texts]

    with pytest.raises(ValueError):
        import_catalog(db_conn, test_schema, str(csvp), BadVecEmbedder(), "m", 1024)


def test_model_dimension_identity_prevents_mixed(db_conn, test_schema, tmp_path):
    csvp = tmp_path / "c.csv"
    write_csv(
        csvp,
        [{"product_id": "P1", "name": "A", "category": "C", "description": "x", "price_idr": "1"}],
        CATALOG_HEADER,
    )
    embedder = FakeEmbedder(dim=1024)
    import_catalog(db_conn, test_schema, str(csvp), embedder, "m1", 1024)
    with pytest.raises(ValueError):
        import_catalog(db_conn, test_schema, str(csvp), embedder, "m2", 1024)
    with db_conn.cursor() as cur:
        vec = [0.0] * 1024
        with pytest.raises(Exception):
            cur.execute(
                sql.SQL("SELECT * FROM {}.hybrid_search(%s::vector, %s, %s, %s, %s)").format(
                    sql.Identifier(test_schema)
                ),
                (vec, "q", "m2", 1024, 3),
            )
    db_conn.rollback()


def test_hybrid_search_or_fusion_and_guard(db_conn, test_schema, tmp_path):
    csvp = tmp_path / "c.csv"
    rows = [
        {"product_id": "VEC", "name": "Alpha", "category": "C", "description": "unikvec", "price_idr": "1"},
        {"product_id": "KW", "name": "Kopi Bubuk", "category": "C", "description": "minuman harum khas", "price_idr": "2"},
        {"product_id": "BOTH", "name": "Kopi Alpha", "category": "C", "description": "campuran spesial", "price_idr": "3"},
    ]
    write_csv(csvp, rows, CATALOG_HEADER)
    chunks = read_csv_products(csvp)
    vec_map = {}
    for c in chunks:
        if c.product_id == "VEC":
            vec_map[c.content] = 0
        elif c.product_id == "KW":
            vec_map[c.content] = 1
        elif c.product_id == "BOTH":
            vec_map[c.content] = 2
    emb = OneHotEmbedder(dim=1024, mapping=vec_map)
    import_catalog(db_conn, test_schema, str(csvp), emb, "m", 1024)

    q_text = "kopi"
    q_emb = OneHotEmbedder(dim=1024, mapping={q_text: 0})
    results = hybrid_search(db_conn, test_schema, q_text, q_emb, "m", 1024, top_k=3)
    ids = [r.product_id for r in results]
    assert "VEC" in ids
    assert ("KW" in ids) or ("BOTH" in ids)
    ok, final = guard_results(results, 0.55)
    assert ok and len(final) > 0
    ok2, final2 = guard_results(results, 1.1)
    assert not ok2 and final2 == []


def test_hybrid_search_keyword_leg_is_or_not_implicit_and(db_conn, test_schema, tmp_path):
    csvp = tmp_path / "or.csv"
    rows = [
        {"product_id": f"p{i:02d}", "name": "Produk", "category": "C", "description": "lain", "price_idr": "1"}
        for i in range(20)
    ]
    rows.append({"product_id": "p-kw", "name": "Toner", "category": "C", "description": "toner toner mawar", "price_idr": "1"})
    rows.append({"product_id": "p-sabun", "name": "Sabun", "category": "C", "description": "muka", "price_idr": "1"})
    write_csv(csvp, rows, CATALOG_HEADER)

    chunks = read_csv_products(csvp)
    vec_map = {c.content: 0 for c in chunks if c.product_id.startswith("p") and c.product_id[1:].isdigit()}
    vec_map.update(
        {
            next(c.content for c in chunks if c.product_id == "p-kw"): 1,
            next(c.content for c in chunks if c.product_id == "p-sabun"): 2,
        }
    )
    import_catalog(db_conn, test_schema, str(csvp), OneHotEmbedder(dim=1024, mapping=vec_map), "m", 1024)

    q_text = "sabun toner"
    q_emb = OneHotEmbedder(dim=1024, mapping={q_text: 0})
    results = hybrid_search(db_conn, test_schema, q_text, q_emb, "m", 1024, top_k=3)
    ids = [r.product_id for r in results]

    assert "p-kw" in ids, "product matching only the second token must still be retrieved"
    assert any(i[1:].isdigit() for i in ids), "cosine top20 must still contribute"
    with db_conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT count(*) c FROM {}.products WHERE content_tsv @@ to_tsquery('simple', 'sabun & toner')"
            ).format(sql.Identifier(test_schema))
        )
        assert cur.fetchone()["c"] == 0, "implicit AND would match nothing"

    empty_kw = hybrid_search(
        db_conn, test_schema, "!!!", OneHotEmbedder(dim=1024, mapping={"!!!": 0}), "m", 1024, top_k=3
    )
    assert len(empty_kw) == 3
    assert all(r.product_id != "p-kw" for r in empty_kw)


def _seed_guard_catalog(db_conn, schema, tmp_path) -> MappedEmbedder:
    csvp = tmp_path / "catalog.csv"
    write_csv(
        csvp,
        [
            {
                "product_id": "P1",
                "name": "Sabun",
                "category": "Wajah",
                "description": "sabun muka",
                "price_idr": "1000",
            }
        ],
        CATALOG_HEADER,
    )
    content = read_csv_products(csvp)[0].content
    cfg = get_config()
    dim = cfg.embedding_dim
    hit = [1.0] + [0.0] * (dim - 1)
    half = [0.5, math.sqrt(0.75)] + [0.0] * (dim - 2)
    emb = MappedEmbedder(dim, {content: hit, "sabun": hit, "sepatu": half})
    import_catalog(db_conn, schema, str(csvp), emb, cfg.embedding_model, dim)
    return emb


EVAL_CASES = [
    {"id": "r1", "text": "sabun", "lang": "id", "expected_product_id": "P1"},
    {"id": "n1", "text": "sepatu", "lang": "id", "expected_product_id": "no_match"},
]


def test_cli_synthetic_eval_executes_search_and_guard_without_metrics(
    db_conn, test_schema, db_url, tmp_path, monkeypatch, capsys
):
    from rag_slice.cli import main

    emb = _seed_guard_catalog(db_conn, test_schema, tmp_path)
    monkeypatch.setattr("rag_slice.cli._embedder", lambda cfg: emb)
    path = write_eval(tmp_path / "smoke_synthetic.jsonl", EVAL_CASES)

    rc = main(["eval", "--file", str(path), "--database-url", db_url, "--schema", test_schema, "--threshold", "0.55"])
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert set(out) == {"synthetic", "executed", "cases"}
    assert out["synthetic"] is True
    assert out["executed"] == 2
    assert len(out["cases"]) == 2
    for row in out["cases"]:
        assert set(row) == {"id", "status", "guard", "product_ids", "cosines"}
        assert row["status"] == "ok"
        assert len(row["product_ids"]) == len(row["cosines"]) >= 1
    guards = {row["id"]: row["guard"] for row in out["cases"]}
    assert guards == {"r1": "match", "n1": "no_match"}
    for metric in ("retrieval_hitrate", "no_match_recall", "accuracy", "passed", "full_spec_ready"):
        assert metric not in out


def test_sql_hybrid_search_rejects_bad_top_k_and_real_vector_dim(db_conn, test_schema, tmp_path):
    _seed_guard_catalog(db_conn, test_schema, tmp_path)
    vec = [1.0] + [0.0] * 1023
    call = sql.SQL("SELECT * FROM {}.hybrid_search(%s::vector, %s, %s, %s, %s)").format(
        sql.Identifier(test_schema)
    )
    with db_conn.cursor() as cur:
        for top_k in (0, 4, None):
            with pytest.raises(Exception, match="top_k"):
                cur.execute(call, (vec, "q", "m", 1024, top_k))
            db_conn.rollback()

        # Caller metadata claims the catalog dim, but the vector itself is wider.
        cur.execute(
            sql.SQL("UPDATE {}.catalog_meta SET embedding_dim = 768").format(
                sql.Identifier(test_schema)
            )
        )
        with pytest.raises(Exception, match="query embedding dimension"):
            cur.execute(call, (vec, "q", "m", 768, 3))
        db_conn.rollback()


def test_cli_eval_raw_top3_hitrate_independent_of_guard(
    db_conn, test_schema, db_url, tmp_path, monkeypatch, capsys
):
    from rag_slice.cli import main

    emb = _seed_guard_catalog(db_conn, test_schema, tmp_path)
    monkeypatch.setattr("rag_slice.cli._embedder", lambda cfg: emb)
    path = write_eval(tmp_path / "real.jsonl", EVAL_CASES)
    args = ["eval", "--file", str(path), "--database-url", db_url, "--schema", test_schema]

    rc = main(args + ["--threshold", "0.55"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["retrieval_hitrate"] == 1.0
    assert out["no_match_recall"] == 1.0
    assert out["guarded_positive_acceptance"] == 1.0
    assert out["pilot"] is True
    assert out["pilot_passed"] is True
    assert out["full_spec_ready"] is False
    assert out["passed"] is True

    # A threshold above every cosine must not change RAW retrieval hitrate.
    rc = main(args + ["--threshold", "1.1"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["retrieval_hitrate"] == 1.0
    assert out["guarded_positive_acceptance"] == 0.0
    assert out["no_match_recall"] == 1.0


def test_cli_sweep_guard_changes_recall_but_not_hitrate(
    db_conn, test_schema, db_url, tmp_path, monkeypatch, capsys
):
    from rag_slice.cli import main

    emb = _seed_guard_catalog(db_conn, test_schema, tmp_path)
    monkeypatch.setattr("rag_slice.cli._embedder", lambda cfg: emb)
    path = write_eval(tmp_path / "real.jsonl", EVAL_CASES)

    rc = main(
        [
            "sweep",
            "--file",
            str(path),
            "--database-url",
            db_url,
            "--schema",
            test_schema,
            "--min",
            "0.3",
            "--max",
            "0.7",
            "--step",
            "0.2",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    rows = out["sweep"]
    assert [r["threshold"] for r in rows] == [0.3, 0.5, 0.7]
    assert [r["retrieval_hitrate"] for r in rows] == [1.0, 1.0, 1.0]
    assert [r["no_match_recall"] for r in rows] == [0.0, 0.0, 1.0]
    assert [r["guarded_positive_acceptance"] for r in rows] == [1.0, 1.0, 1.0]
