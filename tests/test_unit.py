from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_slice.chunking import canonical_content_hash, read_csv_products, row_to_content
from rag_slice.db import SearchResult
from rag_slice.embedding import validate_vector
from rag_slice.eval import (
    EvalCase,
    SearchOutcome,
    add_real_post,
    is_indonesian,
    is_synthetic_path,
    load_jsonl,
    score_eval,
    threshold_sweep,
    validate_cases,
    validate_threshold,
)
from rag_slice.search import guard_results, hybrid_search

CSV_HEADER = "product_id,name,category,description,price_idr"


def write_csv(path: Path, header: str, rows: str) -> Path:
    path.write_text(header + "\n" + rows, encoding="utf-8")
    return path


def test_canonical_hash_stable_and_order_independent():
    a = {"name": "Produk", "id": "1", "desc": "Enak"}
    b = {"desc": "Enak", "id": "1", "name": "Produk"}
    assert canonical_content_hash(a) == canonical_content_hash(b)


def test_chunk_one_product_per_row(tmp_path: Path):
    p = write_csv(
        tmp_path / "p.csv",
        CSV_HEADER,
        "A,Sabun,Wajah,bersih,1000\nB,Sampo,Rambut,wangi,2000\n",
    )
    chunks = read_csv_products(p)
    assert len(chunks) == 2
    assert chunks[0].product_id == "A"
    assert "name: Sabun" in chunks[0].content
    assert "description: bersih" in chunks[0].content
    assert "price_idr: 1000" in row_to_content({"price_idr": "1000"})


def test_read_csv_requires_header(tmp_path: Path):
    p = tmp_path / "p.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_csv_products(p)


@pytest.mark.parametrize(
    "header",
    [
        "product_id,name,category\nA,a,c\n",
        "name,category,description,price_idr\nA,a,c,1\n",
        "product_id,name,category,description\nA,a,c,d\n",
        "product_id,name,name,category,description,price_idr\n",
        "product_id,name,,description,price_idr\n",
    ],
)
def test_csv_rejects_missing_or_malformed_headers(tmp_path: Path, header: str):
    p = tmp_path / "bad.csv"
    p.write_text(header, encoding="utf-8")
    with pytest.raises(ValueError):
        read_csv_products(p)


@pytest.mark.parametrize(
    "rows",
    [
        "A,Sabun,Wajah,bersih,1000,EXTRA\n",
        "A,Sabun,Wajah,,1000\n",
        "A,Sabun,Wajah,bersih,\n",
        "A,Sabun,Wajah,bersih,bukan-angka\n",
        "A,Sabun,Wajah,bersih,1000\nA,Sampo,Rambut,wangi,2000\n",
    ],
)
def test_csv_rejects_malformed_rows(tmp_path: Path, rows: str):
    p = write_csv(tmp_path / "bad.csv", CSV_HEADER, rows)
    with pytest.raises(ValueError):
        read_csv_products(p)


def test_csv_rejects_missing_optional_value_but_accepts_empty(tmp_path: Path):
    header = CSV_HEADER + ",stock"
    missing = write_csv(tmp_path / "missing.csv", header, "A,Sabun,Wajah,bersih,1000\n")
    with pytest.raises(ValueError, match="malformed row 2"):
        read_csv_products(missing)
    empty = write_csv(tmp_path / "empty.csv", header, "A,Sabun,Wajah,bersih,1000,\n")
    assert "stock:" in read_csv_products(empty)[0].content


def test_csv_accepts_id_column_and_optional_fields(tmp_path: Path):
    p = write_csv(
        tmp_path / "p.csv",
        "id,name,category,description,price,stock",
        "A,Sabun,Wajah,bersih,1000,5\n",
    )
    chunks = read_csv_products(p)
    assert chunks[0].product_id == "A"
    assert "stock: 5" in chunks[0].content


def test_csv_accepts_price_idr_only(tmp_path: Path):
    p = write_csv(
        tmp_path / "p.csv",
        "product_id,name,category,description,price_idr",
        "A,Sabun,Wajah,bersih,1000\n",
    )
    assert read_csv_products(p)[0].product_id == "A"


def test_guard_empty_no_match():
    ok, res = guard_results([], 0.55)
    assert not ok and res == []


def test_guard_uses_max_cosine_not_rrf():
    results = [
        SearchResult("A", cosine_similarity=0.40, rrf_score=0.99),
        SearchResult("B", cosine_similarity=0.54, rrf_score=0.98),
        SearchResult("C", cosine_similarity=0.10, rrf_score=0.97),
    ]
    ok, _ = guard_results(results, 0.55)
    assert not ok
    results2 = [SearchResult("A", 0.55, 0.01)]
    ok2, final2 = guard_results(results2, 0.55)
    assert ok2 and final2[0].product_id == "A"
    results3 = [SearchResult("A", 0.549999, 0.99)]
    assert not guard_results(results3, 0.55)[0]


def test_guard_rejects_nonfinite_threshold():
    with pytest.raises(ValueError):
        guard_results([SearchResult("A", 1.0, 1.0)], float("nan"))
    with pytest.raises(ValueError):
        guard_results([], float("inf"))


def test_validate_threshold_rejects_bool_and_nonfinite():
    for bad in (True, float("nan"), float("inf"), None, "nope"):
        with pytest.raises(ValueError):
            validate_threshold(bad)
    assert validate_threshold(0.55) == 0.55


def test_indonesian_detection():
    assert is_indonesian("id")
    assert is_indonesian("id-mixed")
    assert is_indonesian("code-mixed-id-en")
    assert not is_indonesian("en")


def test_eval_retrieval_hitrate_is_raw_and_guard_is_reported_separately():
    cases = [EvalCase("r", "t", "id", "P1"), EvalCase("n", "t", "id", "no_match")]
    outcomes = [SearchOutcome(["P1"], False), SearchOutcome(["P1"], True)]
    rep = score_eval(cases, outcomes)
    assert rep.retrieval_hitrate == 1.0
    assert rep.retrieval_guarded_accepted == 0
    assert rep.guarded_positive_acceptance == 0.0
    assert rep.no_match_recall == 0.0

    good = [SearchOutcome(["P1"], True), SearchOutcome([], False)]
    rep2 = score_eval(cases, good)
    assert rep2.retrieval_hitrate == 1.0
    assert rep2.guarded_positive_acceptance == 1.0
    assert rep2.no_match_recall == 1.0


def _pilot_fixture(retrieval: int, negative: int):
    cases = [EvalCase(f"r{i}", "t", "id", "P") for i in range(retrieval)]
    cases += [EvalCase(f"n{i}", "t", "id", "no_match") for i in range(negative)]
    outcomes = [SearchOutcome(["P"], True) for _ in range(retrieval)]
    outcomes += [SearchOutcome([], False) for _ in range(negative)]
    return cases, outcomes


def test_pilot_detection_boundaries_and_full_spec_readiness():
    cases, outcomes = _pilot_fixture(99, 50)
    rep = score_eval(cases, outcomes)
    assert rep.pilot is True
    assert rep.passed is True
    assert rep.pilot_passed is True
    assert rep.full_spec_ready is False

    cases, outcomes = _pilot_fixture(100, 49)
    rep = score_eval(cases, outcomes)
    assert rep.pilot is True
    assert rep.passed is True
    assert rep.pilot_passed is True
    assert rep.full_spec_ready is False

    cases, outcomes = _pilot_fixture(100, 50)
    rep = score_eval(cases, outcomes)
    assert rep.pilot is False
    assert rep.passed is True
    assert rep.pilot_passed is False
    assert rep.full_spec_ready is True


def test_eval_scoring_and_cohorts():
    cases = [
        EvalCase("1", "t", "id", "P1"),
        EvalCase("2", "t", "en", "no_match"),
    ]
    outcomes = [SearchOutcome(["P1"], True), SearchOutcome([], False)]
    rep = score_eval(cases, outcomes)
    assert rep.retrieval_hitrate == 1.0
    assert rep.no_match_recall == 1.0
    assert not rep.passed
    assert any("Indonesian share" in r for r in rep.reasons)

    cases2 = [EvalCase("1", "t", "id", "P1")]
    rep2 = score_eval(cases2, [SearchOutcome(["P1"], True)])
    assert not rep2.passed
    assert "missing negative (no_match) cohort" in rep2.reasons

    with pytest.raises(ValueError):
        score_eval(cases2, [])


def test_validate_cases_catalog_ref():
    cases = [EvalCase("1", "t", "id", "P9")]
    errs = validate_cases(cases, catalog_ids={"P1"})
    assert any("not in catalog" in e for e in errs)


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '"nope"',
        "null",
        '{"id":1,"text":"x","lang":"id","expected_product_id":"P1"}',
        '{"id":"1","text":null,"lang":"id","expected_product_id":"P1"}',
        '{"id":"1","text":"   ","lang":"id","expected_product_id":"P1"}',
        '{"id":"","text":"x","lang":"id","expected_product_id":"P1"}',
        '{"id":"TODO","text":"x","lang":"id","expected_product_id":"P1"}',
        '{"id":"1","text":"TODO tulis","lang":"id","expected_product_id":"P1"}',
        '{"id":"1","text":"x","lang":"fr","expected_product_id":"P1"}',
        '{"id":"1","text":"x","lang":"id","expected_product_id":""}',
        '{"id":"1","text":"x","lang":"id","expected_product_id":"TBD"}',
        '{"id":"1","text":"x","lang":"id"}',
        '{"id":"1","text":"x","lang":"id","expected_product_id":"P1"}\n'
        '{"id":"1","text":"y","lang":"id","expected_product_id":"P1"}',
    ],
)
def test_load_jsonl_rejects_invalid_cases(tmp_path: Path, payload: str):
    p = tmp_path / "bad.jsonl"
    p.write_text(payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(p)


def test_load_jsonl_accepts_valid_cases(tmp_path: Path):
    p = tmp_path / "ok.jsonl"
    p.write_text(
        '{"id":"1","text":"sabun","lang":"id","expected_product_id":"P1"}\n'
        '{"id":"2","text":"soap","lang":"en","expected_product_id":"no_match"}\n'
        '{"id":"3","text":"sabun murah","lang":"id-mixed","expected_product_id":"P1"}\n',
        encoding="utf-8",
    )
    cases = load_jsonl(p)
    assert [c.id for c in cases] == ["1", "2", "3"]
    assert cases[2].lang == "id-mixed"


def _catalog_csv(tmp_path: Path) -> Path:
    return write_csv(
        tmp_path / "catalog.csv",
        CSV_HEADER,
        "DEMO-001,Sabun Muka,Wajah,sabun lembut,35000\n"
        "DEMO-002,Serum Cerah,Wajah,serum vitamin c,89000\n",
    )


def test_catalog_command_prints_one_product_per_line(tmp_path: Path, capsys):
    from rag_slice.cli import main

    rc = main(["catalog", "--catalog", str(_catalog_csv(tmp_path))])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out == ["DEMO-001\tSabun Muka\tWajah", "DEMO-002\tSerum Cerah\tWajah"]


def test_add_post_noninteractive_appends_valid_json(tmp_path: Path, capsys, monkeypatch):
    import rag_slice.cli as cli

    def no_db(url):
        raise AssertionError("add-post must never run search")

    monkeypatch.setattr(cli, "connect", no_db)
    path = tmp_path / "nested" / "real" / "posts.jsonl"
    rc = cli.main(
        [
            "add-post",
            "--file",
            str(path),
            "--catalog",
            str(_catalog_csv(tmp_path)),
            "--text",
            "Sabunnya lembut — suka!",
            "--label",
            "DEMO-001",
        ]
    )
    assert rc == 0
    raw = path.read_text(encoding="utf-8")
    assert "\\u2014" not in raw, "must write UTF-8, not ASCII escapes"
    case = load_jsonl(path)[0]
    assert case == EvalCase("real-001", "Sabunnya lembut — suka!", "id", "DEMO-001")

    rc = cli.main(
        [
            "add-post",
            "--file",
            str(path),
            "--catalog",
            str(_catalog_csv(tmp_path)),
            "--text",
            "Sepatu lari ini enak",
            "--label",
            "no_match",
        ]
    )
    assert rc == 0
    cases = load_jsonl(path)
    assert [c.id for c in cases] == ["real-001", "real-002"]
    assert cases[1].lang == "id"
    assert cases[1].expected_product_id == "no_match"


def test_add_post_continues_ids_from_file(tmp_path: Path):
    path = tmp_path / "posts.jsonl"
    # no trailing newline on purpose: the append must still produce two records
    path.write_text(
        '{"id":"real-007","text":"lama","lang":"id","expected_product_id":"no_match"}',
        encoding="utf-8",
    )
    case = add_real_post(path, "baru", None, "no_match", {"DEMO-001"})
    assert case.id == "real-008"
    assert case.lang == "id"
    assert [c.id for c in load_jsonl(path)] == ["real-007", "real-008"]


@pytest.mark.parametrize(
    "text,lang,label",
    [
        ("", "id", "DEMO-001"),
        ("   ", "id", "DEMO-001"),
        ("teks", "id", "DEMO-999"),
        ("teks", "id", ""),
        ("teks", "fr", "DEMO-001"),
        ("sudah ada", "id", "DEMO-001"),
    ],
)
def test_add_post_rejects_invalid_input(tmp_path: Path, text: str, lang: str, label: str):
    path = tmp_path / "posts.jsonl"
    path.write_text(
        '{"id":"real-001","text":"sudah ada","lang":"id","expected_product_id":"no_match"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        add_real_post(path, text, lang, label, {"DEMO-001"})
    assert len(load_jsonl(path)) == 1


def test_add_post_rejection_does_not_create_file(tmp_path: Path):
    path = tmp_path / "real" / "posts.jsonl"
    with pytest.raises(ValueError):
        add_real_post(path, "teks", "id", "DEMO-999", {"DEMO-001"})
    assert not path.exists()


def test_add_post_interactive_loops_until_n(tmp_path: Path, capsys, monkeypatch):
    import rag_slice.cli as cli

    path = tmp_path / "posts.jsonl"
    answers = iter(["Sabun", "bagus", "", "id", "DEMO-001", "y", "Serum", "", "", "no_match", "n"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    rc = cli.main(["add-post", "--file", str(path), "--catalog", str(_catalog_csv(tmp_path))])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DEMO-001\tSabun Muka\tWajah" in out
    cases = load_jsonl(path)
    assert [(c.id, c.text, c.lang, c.expected_product_id) for c in cases] == [
        ("real-001", "Sabun\nbagus", "id", "DEMO-001"),
        ("real-002", "Serum", "id", "no_match"),
    ]


def test_add_post_interactive_reprompts_on_unknown_label(tmp_path: Path, capsys, monkeypatch):
    import rag_slice.cli as cli

    path = tmp_path / "posts.jsonl"
    answers = iter(["Teks", "", "en", "NOPE", "DEMO-002", "n"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    rc = cli.main(["add-post", "--file", str(path), "--catalog", str(_catalog_csv(tmp_path))])
    out = capsys.readouterr().out
    assert rc == 0
    assert "label must be a catalog product_id or no_match" in out
    assert load_jsonl(path)[0].expected_product_id == "DEMO-002"


def test_threshold_sweep_boundary_and_raw_retrieval():
    cases = [EvalCase("n", "t", "id", "no_match")]
    rows = threshold_sweep(cases, [(["P1"], 0.5)], [0.5, 0.500001])
    assert rows[0]["no_match_recall"] == 0.0  # max cosine == threshold counts as a match
    assert rows[1]["no_match_recall"] == 1.0

    cases = [EvalCase("r", "t", "id", "P1")]
    rows = threshold_sweep(cases, [(["P1"], 0.1)], [0.9])
    assert rows[0]["retrieval_hitrate"] == 1.0
    assert rows[0]["guarded_positive_acceptance"] == 0.0


def test_threshold_sweep_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        threshold_sweep([EvalCase("1", "t", "id", "P1")], [], [0.5])


def test_threshold_sweep_rejects_nonfinite_thresholds():
    cases = [EvalCase("1", "t", "id", "P1")]
    with pytest.raises(ValueError):
        threshold_sweep(cases, [(["P1"], 1.0)], [float("nan")])
    with pytest.raises(ValueError):
        threshold_sweep(cases, [(["P1"], float("nan"))], [0.5])


def test_synthetic_path_detection():
    assert is_synthetic_path("data/smoke_synthetic.jsonl")
    assert not is_synthetic_path("evals/real.jsonl")


def test_frange_boundaries_and_invalid_steps():
    from rag_slice.cli import frange

    assert list(frange(0.3, 0.5, 0.1)) == [0.3, 0.4, 0.5]
    for step in (0.0, -0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            list(frange(0.3, 0.5, step))
    with pytest.raises(ValueError):
        list(frange(0.5, 0.3, 0.1))
    with pytest.raises(ValueError):
        list(frange(float("nan"), 0.5, 0.1))


def test_frange_rejects_oversized_stalled_and_nonadvancing_ranges():
    from rag_slice.cli import MAX_THRESHOLDS, frange

    assert len(list(frange(0.0, float(MAX_THRESHOLDS - 1), 1.0))) == MAX_THRESHOLDS
    with pytest.raises(ValueError, match="too many thresholds"):
        list(frange(0.0, float(MAX_THRESHOLDS), 1.0))
    with pytest.raises(ValueError, match="too small to advance"):
        list(frange(1e16, 1e16 + 2, 1.0))
    with pytest.raises(ValueError, match="too large"):
        list(frange(0.0, 0.5, 5e-324))


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def _patch_ollama(monkeypatch, payload):
    import rag_slice.embedding as embedding

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, path, json):
            return _FakeResponse(payload)

    monkeypatch.setattr(embedding.httpx, "Client", FakeClient)
    return embedding.OllamaEmbedding(base_url="http://localhost:11434", model="m", dim=3)


@pytest.mark.parametrize(
    "vec",
    [
        [True, 0.5, 0.5],
        [1.0, "0.5", 0.5],
        [1.0, float("nan"), 0.5],
        [1.0, 2.0],
        [0.0, 0.0, 0.0],
    ],
)
def test_ollama_embedding_validates_raw_values_before_coercion(monkeypatch, vec):
    emb = _patch_ollama(monkeypatch, {"embedding": vec})
    with pytest.raises(ValueError):
        emb.embed(["q"])


def test_ollama_embedding_accepts_numeric_raw_values(monkeypatch):
    emb = _patch_ollama(monkeypatch, {"embedding": [1, 2.5, 3]})
    assert emb.embed(["q"]) == [[1.0, 2.5, 3.0]]


def test_validate_vector_rejects_bad_vectors():
    assert validate_vector([1, 2, 3], 3) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        validate_vector([0.0, 0.0, 0.0], 3)
    with pytest.raises(ValueError):
        validate_vector([1.0, float("nan"), 0.0], 3)
    with pytest.raises(ValueError):
        validate_vector([1.0, float("inf"), 0.0], 3)
    with pytest.raises(ValueError):
        validate_vector([1.0], 3)
    with pytest.raises(ValueError):
        validate_vector(["x", 1.0, 2.0], 3)


class _NoEmbed:
    def embed(self, texts):
        raise AssertionError("embed must not be called")


def test_hybrid_search_rejects_bad_top_k_before_embedding():
    for top_k in (0, 4, -1, 1.5, True):
        with pytest.raises(ValueError):
            hybrid_search(None, "s", "q", _NoEmbed(), "m", 3, top_k=top_k)


def test_hybrid_search_rejects_bad_query_vectors_before_sql():
    class BadEmbed:
        def __init__(self, vec):
            self.vec = vec

        def embed(self, texts):
            return [self.vec]

    with pytest.raises(ValueError, match="nonfinite"):
        hybrid_search(None, "s", "q", BadEmbed([float("nan")] * 3), "m", 3)
    with pytest.raises(ValueError, match="zero vector"):
        hybrid_search(None, "s", "q", BadEmbed([0.0] * 3), "m", 3)
    with pytest.raises(ValueError, match="dim mismatch"):
        hybrid_search(None, "s", "q", BadEmbed([1.0, 2.0]), "m", 3)


def test_sweep_synthetic_refuses_metrics(tmp_path: Path, capsys):
    from rag_slice.cli import main

    p = tmp_path / "smoke_synthetic.jsonl"
    p.write_text(
        '{"id":"1","text":"sabun","lang":"id","expected_product_id":"no_match"}\n',
        encoding="utf-8",
    )
    rc = main(["sweep", "--file", str(p)])
    captured = capsys.readouterr()
    assert rc != 0
    assert captured.out == ""
    assert "synthetic" in json.loads(captured.err)["error"].lower()


def test_init_db_never_prints_credentials(monkeypatch, capsys):
    import rag_slice.cli as cli

    class DummyConn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(cli, "connect", lambda url: DummyConn())
    monkeypatch.setattr(cli, "init_db", lambda conn, schema, dim: None)
    rc = cli.main(
        [
            "init-db",
            "--database-url",
            "postgresql://user:supersecret@localhost/db",
            "--schema",
            "s",
            "--dim",
            "3",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "supersecret" not in out
    assert "database_url" not in out
    assert "postgresql://" not in out


def test_error_output_redacts_credentials(monkeypatch, capsys):
    import rag_slice.cli as cli

    def boom(url):
        raise RuntimeError(f"cannot connect to {url}")

    monkeypatch.setattr(cli, "connect", boom)
    rc = cli.main(
        [
            "init-db",
            "--database-url",
            "postgresql://user:supersecret@localhost/db",
            "--schema",
            "s",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "supersecret" not in err
    assert "***" in err
