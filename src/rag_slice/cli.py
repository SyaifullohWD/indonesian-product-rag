from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from .config import get_config
from .db import connect, init_db
from .embedding import OllamaEmbedding
from .eval import (
    ALLOWED_LANGS,
    NO_MATCH,
    CatalogEntry,
    EvalCase,
    SearchOutcome,
    add_real_post,
    is_synthetic_path,
    load_catalog_labels,
    load_jsonl,
    score_eval,
    threshold_sweep,
    validate_cases,
    validate_threshold,
)
from .ingest import import_catalog
from .search import guard_results, hybrid_search

MAX_THRESHOLDS = 10001


def _embedder(cfg):
    return OllamaEmbedding(base_url=cfg.ollama_base_url, model=cfg.embedding_model, dim=cfg.embedding_dim)


def _redact_secrets(text: str) -> str:
    text = re.sub(r"(://[^/\s:@]+:)[^@\s/]+@", r"\1***@", text)
    return re.sub(r"(password=)[^\s&;]+", r"\1***", text, flags=re.IGNORECASE)


def cmd_init_db(args) -> int:
    cfg = get_config()
    db_url = args.database_url or cfg.database_url
    schema = args.schema or cfg.schema_name
    dim = args.dim or cfg.embedding_dim
    with connect(db_url) as conn:
        init_db(conn, schema=schema, dim=dim)
    # Never echo the database URL: it may contain credentials.
    print(json.dumps({"ok": True, "schema": schema, "dim": dim}))
    return 0


def cmd_import_catalog(args) -> int:
    cfg = get_config()
    db_url = args.database_url or cfg.database_url
    schema = args.schema or cfg.schema_name
    with connect(db_url) as conn:
        stats = import_catalog(
            conn,
            schema=schema,
            csv_path=args.csv,
            embedder=_embedder(cfg),
            model=cfg.embedding_model,
            dim=cfg.embedding_dim,
        )
    print(json.dumps({"ok": True, "csv": args.csv, **stats.__dict__}))
    return 0


def cmd_search(args) -> int:
    cfg = get_config()
    db_url = args.database_url or cfg.database_url
    schema = args.schema or cfg.schema_name
    threshold = validate_threshold(
        args.threshold if args.threshold is not None else cfg.guard_threshold
    )
    with connect(db_url) as conn:
        results = hybrid_search(
            conn,
            schema=schema,
            query_text=args.query,
            embedder=_embedder(cfg),
            model=cfg.embedding_model,
            dim=cfg.embedding_dim,
            top_k=args.top_k,
        )
        ok, final = guard_results(results, threshold)
    if not ok:
        out = {"ok": False, "guard": "no_match", "threshold": threshold, "results": []}
        print(json.dumps(out, ensure_ascii=False))
        return 0
    out = {
        "ok": True,
        "threshold": threshold,
        "results": [
            {"product_id": r.product_id, "cosine_similarity": r.cosine_similarity, "rrf_score": r.rrf_score}
            for r in final
        ],
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


def _print_catalog(entries: list[CatalogEntry]) -> None:
    for e in entries:
        print(f"{e.product_id}\t{e.name}\t{e.category}")


def cmd_catalog(args) -> int:
    _print_catalog(load_catalog_labels(args.catalog))
    return 0


def _prompt_text() -> str:
    print("Post text (finish with an empty line):")
    lines: list[str] = []
    while True:
        line = input()
        if line == "":
            return "\n".join(lines)
        lines.append(line)


def _prompt_lang() -> str:
    while True:
        raw = input("Lang [id/en/id-mixed] (default id): ").strip().lower()
        if not raw:
            return "id"
        if raw in ALLOWED_LANGS:
            return raw
        print("unknown lang; use id, en, or id-mixed")


def _prompt_label(entries: list[CatalogEntry], catalog_ids: set[str]) -> str:
    _print_catalog(entries)
    while True:
        raw = input("Label (product_id or no_match): ").strip()
        if raw == NO_MATCH or raw in catalog_ids:
            return raw
        print("label must be a catalog product_id or no_match")


def _interactive_add_posts(path: str, entries: list[CatalogEntry]) -> list[EvalCase]:
    catalog_ids = {e.product_id for e in entries}
    added: list[EvalCase] = []
    while True:
        try:
            text = _prompt_text()
            if not text.strip():
                print("post text must not be empty")
                continue
            case = add_real_post(
                path, text, _prompt_lang(), _prompt_label(entries, catalog_ids), catalog_ids
            )
        except EOFError:
            break
        except ValueError as e:
            print(f"rejected: {e}")
            continue
        added.append(case)
        print(f"added {case.id} -> {case.expected_product_id}")
        try:
            if input("Add another post? [Y/n]: ").strip().lower() == "n":
                break
        except EOFError:
            break
    return added


def cmd_add_post(args) -> int:
    entries = load_catalog_labels(args.catalog)
    catalog_ids = {e.product_id for e in entries}
    if args.text is not None:
        if args.label is None:
            raise ValueError("--label is required when --text is given")
        added = [add_real_post(args.file, args.text, args.lang, args.label, catalog_ids)]
    else:
        added = _interactive_add_posts(args.file, entries)
    print(
        json.dumps(
            {"ok": True, "file": args.file, "added": [c.id for c in added]}, ensure_ascii=False
        )
    )
    return 0


def _run_eval_cases(args, cases: list[EvalCase], threshold: float, db_url: str, schema: str):
    cfg = get_config()
    embedder = _embedder(cfg)
    with connect(db_url) as conn:
        with conn.cursor() as cur:
            from psycopg import sql
            cur.execute(sql.SQL("SELECT product_id FROM {}.products").format(sql.Identifier(schema)))
            catalog_ids = {r["product_id"] for r in cur.fetchall()}
        verrors = validate_cases(cases, catalog_ids=catalog_ids)
        if verrors:
            raise ValueError("invalid eval labels: " + "; ".join(verrors[:20]))
        outcomes: list[SearchOutcome] = []
        raw_for_sweep = []
        for c in cases:
            results = hybrid_search(
                conn,
                schema=schema,
                query_text=c.text,
                embedder=embedder,
                model=cfg.embedding_model,
                dim=cfg.embedding_dim,
                top_k=3,
            )
            raw_ids = [r.product_id for r in results]
            ok, _ = guard_results(results, threshold)
            outcomes.append(SearchOutcome(product_ids=raw_ids, ok=ok))
            max_cos = max((r.cosine_similarity for r in results), default=float("-inf"))
            raw_for_sweep.append((raw_ids, max_cos))
    return outcomes, raw_for_sweep, catalog_ids


def _run_synthetic_eval(
    cases: list[EvalCase], db_url: str, schema: str, threshold: float
) -> tuple[dict, bool]:
    cfg = get_config()
    embedder = _embedder(cfg)
    rows = []
    executed = 0
    with connect(db_url) as conn:
        for c in cases:
            try:
                results = hybrid_search(
                    conn,
                    schema=schema,
                    query_text=c.text,
                    embedder=embedder,
                    model=cfg.embedding_model,
                    dim=cfg.embedding_dim,
                    top_k=3,
                )
            except Exception:
                rows.append({"id": c.id, "status": "error"})
                continue
            executed += 1
            ok, _ = guard_results(results, threshold)
            rows.append(
                {
                    "id": c.id,
                    "status": "ok",
                    "guard": "match" if ok else "no_match",
                    "product_ids": [r.product_id for r in results],
                    "cosines": [r.cosine_similarity for r in results],
                }
            )
    return {"synthetic": True, "executed": executed, "cases": rows}, executed == len(cases)


def cmd_eval(args) -> int:
    cfg = get_config()
    db_url = args.database_url or cfg.database_url
    schema = args.schema or cfg.schema_name
    threshold = validate_threshold(
        args.threshold if args.threshold is not None else cfg.guard_threshold
    )
    path = args.file
    cases = load_jsonl(path)
    if not cases:
        raise ValueError("eval file is empty or has no runnable cases")

    if is_synthetic_path(path):
        out, all_executed = _run_synthetic_eval(cases, db_url, schema, threshold)
        print(json.dumps(out, ensure_ascii=False))
        return 0 if all_executed else 1

    outcomes, _, _ = _run_eval_cases(args, cases, threshold, db_url, schema)
    report = score_eval(cases, outcomes, path=path)
    print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 0 if report.passed else 1


def cmd_sweep(args) -> int:
    cfg = get_config()
    db_url = args.database_url or cfg.database_url
    schema = args.schema or cfg.schema_name
    path = args.file
    if is_synthetic_path(path):
        raise ValueError("synthetic fixtures cannot be swept; metrics are refused")
    cases = load_jsonl(path)
    if not cases:
        raise ValueError("eval file is empty or has no runnable cases")
    thresholds = [round(x, 10) for x in frange(args.min, args.max, args.step)]
    # Guard is applied per sweep threshold; raw top3 is collected once.
    outcomes, raw, _ = _run_eval_cases(args, cases, 0.0, db_url, schema)
    rows = threshold_sweep(cases, raw, thresholds)
    print(json.dumps({"path": path, "sweep": rows}, ensure_ascii=False))
    return 0


def frange(start, stop, step):
    start = validate_threshold(start)
    stop = validate_threshold(stop)
    step = validate_threshold(step)
    if step <= 0:
        raise ValueError("step must be > 0")
    if start > stop:
        raise ValueError("min must be <= max")
    if start + step == start:
        raise ValueError("step is too small to advance the range")
    span = (stop - start) / step
    if not math.isfinite(span):
        raise ValueError("range is too large for the step")
    count = int(math.floor(span + 1e-9)) + 1
    if count > MAX_THRESHOLDS:
        raise ValueError(f"too many thresholds: {count} > {MAX_THRESHOLDS}")
    # Index arithmetic: repeated addition drifts and can stall forever on tiny steps.
    for i in range(count):
        yield start + i * step


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-slice")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init-db")
    sp.add_argument("--database-url")
    sp.add_argument("--schema")
    sp.add_argument("--dim", type=int)
    sp.set_defaults(func=cmd_init_db)

    sp = sub.add_parser("import-catalog")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--database-url")
    sp.add_argument("--schema")
    sp.set_defaults(func=cmd_import_catalog)

    sp = sub.add_parser("search")
    sp.add_argument("--query", required=True)
    sp.add_argument("--database-url")
    sp.add_argument("--schema")
    sp.add_argument("--threshold", type=float)
    sp.add_argument("--top-k", type=int, default=3)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("catalog")
    sp.add_argument("--catalog", default="data/catalog_demo_id.csv")
    sp.set_defaults(func=cmd_catalog)

    sp = sub.add_parser("add-post")
    sp.add_argument("--file", default="evals/real/posts.jsonl")
    sp.add_argument("--catalog", default="data/catalog_demo_id.csv")
    sp.add_argument("--text")
    sp.add_argument("--lang")
    sp.add_argument("--label")
    sp.set_defaults(func=cmd_add_post)

    sp = sub.add_parser("eval")
    sp.add_argument("--file", required=True)
    sp.add_argument("--database-url")
    sp.add_argument("--schema")
    sp.add_argument("--threshold", type=float)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("sweep")
    sp.add_argument("--file", required=True)
    sp.add_argument("--database-url")
    sp.add_argument("--schema")
    sp.add_argument("--min", type=float, default=0.30)
    sp.add_argument("--max", type=float, default=0.80)
    sp.add_argument("--step", type=float, default=0.05)
    sp.set_defaults(func=cmd_sweep)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(
            json.dumps({"ok": False, "error": _redact_secrets(str(e)), "type": type(e).__name__}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
