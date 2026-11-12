from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

NO_MATCH = "no_match"
SYNTHETIC_MARKER = "smoke_synthetic"
ALLOWED_LANGS = frozenset({"id", "en", "id-mixed"})
PLACEHOLDERS = frozenset(
    {"", "todo", "tbd", "placeholder", "fixme", "n/a", "na", "none", "null", "lorem ipsum", "..."}
)

SPEC_RETRIEVAL = 100
SPEC_NEGATIVE = 50
MIN_RETRIEVAL_HITRATE = 0.85
MIN_NO_MATCH_RECALL = 0.90
MIN_INDONESIAN_SHARE = 0.70


@dataclass
class EvalCase:
    id: str
    text: str
    lang: str
    expected_product_id: str


@dataclass
class EvalReport:
    path: str
    total: int
    retrieval_total: int
    retrieval_hits: int
    retrieval_hitrate: float | None
    retrieval_guarded_accepted: int
    guarded_positive_acceptance: float | None
    negative_total: int
    negative_correct: int
    no_match_recall: float | None
    indonesian_share: float | None
    passed: bool
    pilot: bool
    pilot_passed: bool
    full_spec_ready: bool
    reasons: list[str]
    cohorts_present: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_threshold(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("threshold must be a finite number")
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError("threshold must be a finite number") from None
    if not math.isfinite(threshold):
        raise ValueError("threshold must be a finite number")
    return threshold


def is_synthetic_path(path: str | Path) -> bool:
    p = str(Path(path).name)
    return SYNTHETIC_MARKER in p


def _is_placeholder(value: str) -> bool:
    v = value.strip().lower()
    return v in PLACEHOLDERS or v.startswith(("todo", "tbd", "placeholder", "lorem ipsum"))


def load_jsonl(path: str | Path) -> list[EvalCase]:
    p = Path(path)
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON at {p}:{lineno}: {e}") from e
            where = f"{p}:{lineno}"
            if not isinstance(obj, dict):
                raise ValueError(f"case must be a JSON object at {where}")
            for key in ("id", "text", "lang", "expected_product_id"):
                if key not in obj:
                    raise ValueError(f"missing field {key!r} at {where}")
                if not isinstance(obj[key], str):
                    raise ValueError(f"field {key!r} must be a string at {where}")
            case_id = obj["id"].strip()
            text = obj["text"].strip()
            lang = obj["lang"].strip().lower()
            expected = obj["expected_product_id"].strip()
            if _is_placeholder(case_id):
                raise ValueError(f"empty or placeholder id at {where}")
            if case_id in seen:
                raise ValueError(f"duplicate id {case_id!r} at {where}")
            if _is_placeholder(text):
                raise ValueError(f"{case_id}: empty or placeholder text at {where}")
            if lang not in ALLOWED_LANGS:
                raise ValueError(f"{case_id}: unknown lang {lang!r} at {where}")
            if expected != NO_MATCH and _is_placeholder(expected):
                raise ValueError(f"{case_id}: empty or placeholder expected_product_id at {where}")
            seen.add(case_id)
            cases.append(EvalCase(id=case_id, text=text, lang=lang, expected_product_id=expected))
    return cases


@dataclass(frozen=True)
class CatalogEntry:
    product_id: str
    name: str
    category: str


def load_catalog_labels(path: str | Path) -> list[CatalogEntry]:
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("catalog CSV must have a header row")
        fields = [str(x).strip() for x in reader.fieldnames]
        id_field = "product_id" if "product_id" in fields else "id"
        if id_field not in fields:
            raise ValueError("catalog CSV must contain an 'id' or 'product_id' column")
        for col in ("name", "category"):
            if col not in fields:
                raise ValueError(f"catalog CSV missing required column {col!r}")
        entries: list[CatalogEntry] = []
        for lineno, row in enumerate(reader, start=2):
            pid = (row.get(id_field) or "").strip()
            if not pid:
                raise ValueError(f"missing product id at {p}:{lineno}")
            entries.append(
                CatalogEntry(
                    product_id=pid,
                    name=(row.get("name") or "").strip(),
                    category=(row.get("category") or "").strip(),
                )
            )
        return entries


def _next_real_id(cases: list[EvalCase]) -> str:
    nums = [int(m.group(1)) for c in cases if (m := re.fullmatch(r"real-(\d+)", c.id))]
    return f"real-{max(nums, default=0) + 1:03d}"


def add_real_post(
    path: str | Path,
    text: str,
    lang: str | None,
    label: str,
    catalog_ids: set[str],
) -> EvalCase:
    text = (text or "").strip()
    if not text:
        raise ValueError("post text must not be empty")
    lang = (lang or "").strip().lower() or "id"
    if lang not in ALLOWED_LANGS:
        raise ValueError(f"unknown lang {lang!r}; use id, en, or id-mixed")
    label = (label or "").strip()
    if label != NO_MATCH and label not in catalog_ids:
        raise ValueError(f"label {label!r} is not a catalog product_id and not no_match")
    p = Path(path)
    existing = load_jsonl(p) if p.exists() else []
    if any(c.text == text for c in existing):
        raise ValueError("post text already exists in the eval file")
    case = EvalCase(id=_next_real_id(existing), text=text, lang=lang, expected_product_id=label)
    p.parent.mkdir(parents=True, exist_ok=True)
    # A hand-edited file may lack the trailing newline; never glue two records together.
    prefix = "\n" if p.exists() and p.stat().st_size and not p.read_bytes().endswith(b"\n") else ""
    with p.open("a", encoding="utf-8") as f:
        f.write(prefix + json.dumps(asdict(case), ensure_ascii=False) + "\n")
    return case


def validate_cases(cases: list[EvalCase], catalog_ids: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for c in cases:
        if not c.id or _is_placeholder(c.id):
            errors.append("empty or placeholder id")
        if c.id in seen:
            errors.append(f"duplicate id {c.id}")
        seen.add(c.id)
        if _is_placeholder(c.text):
            errors.append(f"{c.id}: empty or placeholder text")
        if c.lang not in ALLOWED_LANGS:
            errors.append(f"{c.id}: unknown lang {c.lang!r}")
        exp = c.expected_product_id
        if exp != NO_MATCH:
            if _is_placeholder(exp):
                errors.append(f"{c.id}: empty or placeholder expected_product_id")
            elif catalog_ids is not None and exp not in catalog_ids:
                errors.append(f"{c.id}: expected_product_id {exp!r} not in catalog")
    return errors


def is_indonesian(lang: str) -> bool:
    l = (lang or "").lower().strip()
    if not l:
        return False
    if l in {"id", "ind", "indonesian", "id-id"}:
        return True
    if l.startswith("id") and (len(l) == 2 or not l[2].isalpha()):
        return True
    # code-mixed Indonesian
    if "id" in l and ("mix" in l or "code" in l or "+" in l or "/" in l):
        return True
    if "mixed" in l and "en" in l and "id" in l:
        return True
    return False


@dataclass
class SearchOutcome:
    product_ids: list[str]
    ok: bool  # guard ok


def score_eval(
    cases: list[EvalCase],
    outcomes: list[SearchOutcome],
    path: str = "",
) -> EvalReport:
    if len(cases) != len(outcomes):
        raise ValueError("cases and outcomes length mismatch")
    total = len(cases)
    retrieval_cases = [(c, o) for c, o in zip(cases, outcomes) if c.expected_product_id != NO_MATCH]
    negative_cases = [(c, o) for c, o in zip(cases, outcomes) if c.expected_product_id == NO_MATCH]
    retrieval_total = len(retrieval_cases)
    negative_total = len(negative_cases)
    # Retrieval hitrate is RAW top3 and must not depend on the guard.
    retrieval_hits = sum(1 for c, o in retrieval_cases if c.expected_product_id in o.product_ids)
    # Positive acceptance after the guard, reported separately.
    retrieval_guarded_accepted = sum(1 for _, o in retrieval_cases if o.ok)
    negative_correct = sum(1 for _, o in negative_cases if not o.ok)
    retrieval_hitrate = (retrieval_hits / retrieval_total) if retrieval_total else None
    guarded_positive_acceptance = (
        retrieval_guarded_accepted / retrieval_total if retrieval_total else None
    )
    no_match_recall = (negative_correct / negative_total) if negative_total else None
    id_count = sum(1 for c in cases if is_indonesian(c.lang))
    ind_share = (id_count / total) if total else None

    pilot = retrieval_total < SPEC_RETRIEVAL or negative_total < SPEC_NEGATIVE
    cohorts_present = {
        "retrieval": retrieval_total > 0,
        "negative": negative_total > 0,
    }

    reasons: list[str] = []
    passed = True
    if retrieval_total == 0:
        passed = False
        reasons.append("missing retrieval cohort")
    if negative_total == 0:
        passed = False
        reasons.append("missing negative (no_match) cohort")

    if retrieval_hitrate is not None and retrieval_hitrate < MIN_RETRIEVAL_HITRATE:
        passed = False
        reasons.append(f"retrieval hitrate {retrieval_hitrate:.3f} < {MIN_RETRIEVAL_HITRATE}")
    if retrieval_hitrate is None:
        passed = False
    if no_match_recall is not None and no_match_recall < MIN_NO_MATCH_RECALL:
        passed = False
        reasons.append(f"no_match recall {no_match_recall:.3f} < {MIN_NO_MATCH_RECALL}")
    if no_match_recall is None:
        passed = False
    if ind_share is not None and ind_share < MIN_INDONESIAN_SHARE:
        passed = False
        reasons.append(f"Indonesian share {ind_share:.3f} < {MIN_INDONESIAN_SHARE}")
    if ind_share is None:
        passed = False
        reasons.append("empty eval set")

    full_spec_ready = passed and not pilot
    pilot_passed = passed and pilot
    if pilot:
        short: list[str] = []
        if retrieval_total < SPEC_RETRIEVAL:
            short.append(f"retrieval={retrieval_total}/{SPEC_RETRIEVAL}")
        if negative_total < SPEC_NEGATIVE:
            short.append(f"negative={negative_total}/{SPEC_NEGATIVE}")
        reasons.append(
            f"pilot count: {', '.join(short)}; pilot thresholds "
            f"{'passed' if passed else 'failed'}, full spec not ready"
        )

    return EvalReport(
        path=str(path),
        total=total,
        retrieval_total=retrieval_total,
        retrieval_hits=retrieval_hits,
        retrieval_hitrate=retrieval_hitrate,
        retrieval_guarded_accepted=retrieval_guarded_accepted,
        guarded_positive_acceptance=guarded_positive_acceptance,
        negative_total=negative_total,
        negative_correct=negative_correct,
        no_match_recall=no_match_recall,
        indonesian_share=ind_share,
        passed=passed,
        pilot=pilot,
        pilot_passed=pilot_passed,
        full_spec_ready=full_spec_ready,
        reasons=reasons,
        cohorts_present=cohorts_present,
    )


def threshold_sweep(
    cases: list[EvalCase],
    # per-case: (raw product_ids, max_cosine_of_top3 or -inf)
    case_results: list[tuple[list[str], float]],
    thresholds: Iterable[float],
) -> list[dict[str, Any]]:
    if len(cases) != len(case_results):
        raise ValueError("cases and case_results length mismatch")
    checked = [validate_threshold(t) for t in thresholds]
    for _, max_cos in case_results:
        if math.isnan(max_cos):
            raise ValueError("max cosine must not be NaN")
    rows = []
    for th in checked:
        outcomes: list[SearchOutcome] = []
        for (pids, max_cos), c in zip(case_results, cases):
            ok = bool(pids) and max_cos >= th
            # Keep RAW product ids so retrieval hitrate does not depend on the guard.
            outcomes.append(SearchOutcome(product_ids=list(pids), ok=ok))
        rep = score_eval(cases, outcomes)
        rows.append(
            {
                "threshold": th,
                "passed": rep.passed,
                "pilot": rep.pilot,
                "pilot_passed": rep.pilot_passed,
                "full_spec_ready": rep.full_spec_ready,
                "retrieval_hitrate": rep.retrieval_hitrate,
                "guarded_positive_acceptance": rep.guarded_positive_acceptance,
                "no_match_recall": rep.no_match_recall,
                "indonesian_share": rep.indonesian_share,
                "reasons": rep.reasons,
            }
        )
    return rows
