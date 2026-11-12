from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("name", "category", "description")


@dataclass(frozen=True)
class ProductChunk:
    product_id: str
    content: str
    content_hash: str


def canonical_content_hash(row: dict[str, str]) -> str:
    # Canonicalize: sort keys, normalize values to strings, stable JSON.
    canon = {k: ("" if v is None else str(v)) for k, v in sorted(row.items(), key=lambda kv: kv[0])}
    payload = json.dumps(canon, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_to_content(row: dict[str, str]) -> str:
    # One product per headered chunk: include header:value pairs in stable order.
    parts = []
    for k, v in sorted(row.items(), key=lambda kv: kv[0]):
        val = "" if v is None else str(v).strip()
        parts.append(f"{k}: {val}")
    return "\n".join(parts).strip()


def _validate_header(raw_fields: list[str | None], id_field: str | None) -> tuple[str, str]:
    if any(f is None for f in raw_fields):
        raise ValueError("CSV header contains an empty column name")
    fields = [str(f).strip() for f in raw_fields]
    if any(not f for f in fields):
        raise ValueError("CSV header contains an empty column name")
    if len(set(fields)) != len(fields):
        raise ValueError("CSV header contains duplicate column names")

    chosen = id_field.strip() if id_field is not None else None
    if chosen is not None:
        if chosen not in {"id", "product_id"}:
            raise ValueError("CSV id field must be 'id' or 'product_id'")
        if chosen not in fields:
            raise ValueError(f"CSV id field {chosen!r} not found in header")
    elif "product_id" in fields:
        chosen = "product_id"
    elif "id" in fields:
        chosen = "id"
    else:
        raise ValueError("CSV must contain an 'id' or 'product_id' column")

    missing = [c for c in REQUIRED_COLUMNS if c not in fields]
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")
    if "price" not in fields and "price_idr" not in fields:
        raise ValueError("CSV must contain a 'price' or 'price_idr' column")
    price_field = "price_idr" if "price_idr" in fields else "price"
    return chosen, price_field


def read_csv_products(path: str | Path, id_field: str | None = None) -> list[ProductChunk]:
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV must have a header row")
        chosen, price_field = _validate_header(list(reader.fieldnames), id_field)
        chunks: list[ProductChunk] = []
        seen: set[str] = set()
        for i, row in enumerate(reader, start=2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"malformed row {i}: wrong number of values")
            norm = {(k or "").strip(): (v if v is not None else "").strip() for k, v in row.items()}
            pid = norm.get(chosen, "")
            if not pid:
                raise ValueError(f"missing product id at row {i} (field {chosen})")
            if pid in seen:
                raise ValueError(f"duplicate product id {pid!r} at row {i}")
            seen.add(pid)
            for col in REQUIRED_COLUMNS:
                if not norm.get(col, ""):
                    raise ValueError(f"missing {col} at row {i}")
            price = norm.get(price_field, "")
            if not price:
                raise ValueError(f"missing {price_field} at row {i}")
            try:
                price_value = float(price)
            except ValueError:
                raise ValueError(f"invalid {price_field} at row {i}: {price!r}") from None
            if not math.isfinite(price_value) or price_value < 0:
                raise ValueError(f"invalid {price_field} at row {i}: {price!r}")

            content = row_to_content(norm)
            ch = canonical_content_hash(norm)
            chunks.append(ProductChunk(product_id=pid, content=content, content_hash=ch))
        return chunks
