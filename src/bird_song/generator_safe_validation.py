from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROTOCOL_FORMAT_VERSION = 1
EXPECTED_SOURCE_VALIDATION_ROWS = 519
EXPECTED_GENERATOR_SAFE_VALIDATION_ROWS = 510
EXPECTED_EXCLUDED_VALIDATION_ROWS = 9
EXPECTED_HISTORICAL_TRAIN_COUNTERPART_ROWS = 17
EXPECTED_TEST_ROWS = 489
EXCLUSION_RULE = (
    "Exclude each validation row identified by the content_safe_v2 exact-audio duplicate "
    "ledger as the retained counterpart of one or more rows from the historical training split."
)
CAVEAT = (
    "This boundary removes only the nine exact historical-train duplicate relationships already "
    "recorded in the content_safe_v2 ledger. It does not detect near-duplicates, perceptually "
    "similar recordings, or other semantic overlap, and validation results based on 510 rows are "
    "not directly interchangeable with earlier 519-row validation results."
)


def protocol_path_for_manifest(validation_manifest: Path) -> Path:
    return validation_manifest.with_suffix(".protocol.json")


def _portable(path: Path, project_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _species_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(species): int(count)
        for species, count in frame["name"].value_counts().sort_index().items()
    }


def _json_record(row: pd.Series) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            record[str(key)] = None
        elif hasattr(value, "item"):
            record[str(key)] = value.item()
        else:
            record[str(key)] = value
    return record


def _read_manifest(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    frame = pd.read_csv(path)
    required = {"split", "name", "id", "relative_wav_path"}
    missing = required - set(frame.columns)
    if frame.empty or missing or frame[list(required)].isna().any().any():
        raise ValueError(f"{label} is incomplete: missing={sorted(missing)}")
    if frame["relative_wav_path"].astype(str).duplicated().any():
        raise ValueError(f"{label} contains duplicate relative_wav_path values")
    return frame


def prepare_generator_safe_validation(
    *,
    project_root: Path,
    source_validation_manifest: Path,
    source_test_manifest: Path,
    historical_manifest: Path,
    exact_duplicate_protocol: Path,
    output_validation_manifest: Path,
    output_protocol: Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic validation-only safety boundary from the existing duplicate ledger."""
    project_root = project_root.resolve()
    source_validation_manifest = source_validation_manifest.resolve()
    source_test_manifest = source_test_manifest.resolve()
    historical_manifest = historical_manifest.resolve()
    exact_duplicate_protocol = exact_duplicate_protocol.resolve()
    output_validation_manifest = output_validation_manifest.resolve()
    output_protocol = (
        output_protocol.resolve()
        if output_protocol is not None
        else protocol_path_for_manifest(output_validation_manifest)
    )

    validation = _read_manifest(source_validation_manifest, "source validation manifest")
    test = _read_manifest(source_test_manifest, "test manifest")
    historical = _read_manifest(historical_manifest, "historical manifest")
    if len(validation) != EXPECTED_SOURCE_VALIDATION_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_VALIDATION_ROWS} source validation rows, found {len(validation)}"
        )
    if len(test) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS} untouched test rows, found {len(test)}")
    if not exact_duplicate_protocol.is_file():
        raise FileNotFoundError(f"Exact-duplicate protocol is missing: {exact_duplicate_protocol}")
    ledger = json.loads(exact_duplicate_protocol.read_text(encoding="utf-8"))
    counterpart_links = [
        row
        for row in ledger.get("removed_rows", [])
        if row.get("removed_split") == "train" and row.get("retained_split") == "validation"
    ]
    if len(counterpart_links) != EXPECTED_HISTORICAL_TRAIN_COUNTERPART_ROWS:
        raise ValueError(
            "Exact-duplicate ledger must contain exactly "
            f"{EXPECTED_HISTORICAL_TRAIN_COUNTERPART_ROWS} historical-train/validation links"
        )

    excluded_paths = sorted(
        {str(row["retained_relative_wav_path"]) for row in counterpart_links}
    )
    historical_paths = sorted(
        {str(row["removed_relative_wav_path"]) for row in counterpart_links}
    )
    if len(excluded_paths) != EXPECTED_EXCLUDED_VALIDATION_ROWS:
        raise ValueError(
            f"Exact-duplicate ledger must identify exactly {EXPECTED_EXCLUDED_VALIDATION_ROWS} "
            "unique validation rows"
        )
    if len(historical_paths) != EXPECTED_HISTORICAL_TRAIN_COUNTERPART_ROWS:
        raise ValueError("Historical-train counterpart paths must be unique")

    excluded = validation[validation["relative_wav_path"].astype(str).isin(excluded_paths)].copy()
    if len(excluded) != EXPECTED_EXCLUDED_VALIDATION_ROWS or set(
        excluded["relative_wav_path"].astype(str)
    ) != set(excluded_paths):
        raise ValueError("Source validation manifest does not contain every ledger exclusion exactly once")
    historical_counterparts = historical[
        historical["relative_wav_path"].astype(str).isin(historical_paths)
    ].copy()
    if len(historical_counterparts) != EXPECTED_HISTORICAL_TRAIN_COUNTERPART_ROWS or set(
        historical_counterparts["relative_wav_path"].astype(str)
    ) != set(historical_paths):
        raise ValueError("Historical manifest does not contain every ledger counterpart exactly once")
    if set(historical_counterparts["split"].astype(str)) != {"train"}:
        raise ValueError("Every historical counterpart must belong to the historical training split")

    generator_safe = validation[
        ~validation["relative_wav_path"].astype(str).isin(excluded_paths)
    ].copy()
    if len(generator_safe) != EXPECTED_GENERATOR_SAFE_VALIDATION_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_GENERATOR_SAFE_VALIDATION_ROWS} generator-safe rows, "
            f"found {len(generator_safe)}"
        )
    if set(generator_safe["relative_wav_path"].astype(str)) & set(excluded_paths):
        raise AssertionError("Generator-safe validation still contains a ledger exclusion")

    historical_by_path = {
        str(row["relative_wav_path"]): row
        for _, row in historical_counterparts.iterrows()
    }
    link_records = []
    for link in sorted(
        counterpart_links,
        key=lambda row: (
            str(row["retained_relative_wav_path"]),
            str(row["removed_relative_wav_path"]),
        ),
    ):
        historical_path = str(link["removed_relative_wav_path"])
        link_records.append(
            {
                "excluded_validation_relative_wav_path": str(
                    link["retained_relative_wav_path"]
                ),
                "historical_train_row": _json_record(historical_by_path[historical_path]),
            }
        )

    before_counts = _species_counts(validation)
    after_counts = _species_counts(generator_safe)
    test_counts = _species_counts(test)
    protocol = {
        "format_version": PROTOCOL_FORMAT_VERSION,
        "title": "Generator-safe validation boundary",
        "source_validation_manifest": _portable(source_validation_manifest, project_root),
        "output_validation_manifest": _portable(output_validation_manifest, project_root),
        "exact_duplicate_protocol": _portable(exact_duplicate_protocol, project_root),
        "historical_manifest": _portable(historical_manifest, project_root),
        "rule": EXCLUSION_RULE,
        "validation_counts": {
            "before": {"rows": int(len(validation)), "species": before_counts},
            "after": {"rows": int(len(generator_safe)), "species": after_counts},
        },
        "excluded_validation_row_count": int(len(excluded)),
        "historical_train_counterpart_row_count": int(len(link_records)),
        "excluded_validation_rows": [
            _json_record(row)
            for _, row in excluded.sort_values("relative_wav_path", kind="stable").iterrows()
        ],
        "historical_train_counterpart_rows": link_records,
        "test_unchanged": {
            "manifest": _portable(source_test_manifest, project_root),
            "assertion": "The preparation is validation-only: the test manifest is read-only, no test rows are excluded or rewritten, and all 489 rows remain the held-out test set.",
            "rows": int(len(test)),
            "excluded_rows": 0,
            "species": test_counts,
        },
        "caveat": CAVEAT,
    }

    output_validation_manifest.parent.mkdir(parents=True, exist_ok=True)
    csv_temp = output_validation_manifest.with_suffix(output_validation_manifest.suffix + ".tmp")
    generator_safe.to_csv(csv_temp, index=False, lineterminator="\n")
    csv_temp.replace(output_validation_manifest)
    output_protocol.parent.mkdir(parents=True, exist_ok=True)
    json_temp = output_protocol.with_suffix(output_protocol.suffix + ".tmp")
    json_temp.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    json_temp.replace(output_protocol)
    return protocol


def load_generator_safe_validation_identity(
    validation_manifest: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Validate a prepared manifest and return its semantic protocol identity."""
    validation_manifest = validation_manifest.resolve()
    project_root = project_root.resolve()
    protocol_path = protocol_path_for_manifest(validation_manifest)
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Generator-safe validation protocol is missing: {protocol_path}. "
            "Run scripts/prepare_generator_safe_validation.py first."
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    required = {
        "format_version",
        "source_validation_manifest",
        "output_validation_manifest",
        "exact_duplicate_protocol",
        "historical_manifest",
        "rule",
        "validation_counts",
        "excluded_validation_row_count",
        "historical_train_counterpart_row_count",
        "excluded_validation_rows",
        "historical_train_counterpart_rows",
        "test_unchanged",
        "caveat",
    }
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"Generator-safe validation protocol is incomplete: {sorted(missing)}")
    if int(protocol["format_version"]) != PROTOCOL_FORMAT_VERSION:
        raise ValueError("Unsupported generator-safe validation protocol version")
    if protocol["output_validation_manifest"] != _portable(validation_manifest, project_root):
        raise ValueError("Generator-safe validation protocol does not identify this manifest")
    if protocol["rule"] != EXCLUSION_RULE:
        raise ValueError("Generator-safe validation protocol has a non-canonical exclusion rule")

    frame = _read_manifest(validation_manifest, "generator-safe validation manifest")
    after = protocol["validation_counts"].get("after", {})
    before = protocol["validation_counts"].get("before", {})
    excluded_rows = protocol["excluded_validation_rows"]
    counterparts = protocol["historical_train_counterpart_rows"]
    if int(before.get("rows", -1)) != EXPECTED_SOURCE_VALIDATION_ROWS:
        raise ValueError("Generator-safe validation protocol has the wrong source row count")
    if int(after.get("rows", -1)) != EXPECTED_GENERATOR_SAFE_VALIDATION_ROWS:
        raise ValueError("Generator-safe validation protocol has the wrong output row count")
    if len(frame) != EXPECTED_GENERATOR_SAFE_VALIDATION_ROWS:
        raise ValueError("Generator-safe validation manifest has the wrong row count")
    if after.get("species") != _species_counts(frame):
        raise ValueError("Generator-safe validation species counts do not match its protocol")
    if int(protocol["excluded_validation_row_count"]) != EXPECTED_EXCLUDED_VALIDATION_ROWS or len(
        excluded_rows
    ) != EXPECTED_EXCLUDED_VALIDATION_ROWS:
        raise ValueError("Generator-safe validation protocol must identify exactly nine exclusions")
    if int(protocol["historical_train_counterpart_row_count"]) != (
        EXPECTED_HISTORICAL_TRAIN_COUNTERPART_ROWS
    ) or len(counterparts) != EXPECTED_HISTORICAL_TRAIN_COUNTERPART_ROWS:
        raise ValueError("Generator-safe validation protocol must identify exactly 17 counterparts")
    excluded_paths = sorted(str(row["relative_wav_path"]) for row in excluded_rows)
    if len(set(excluded_paths)) != EXPECTED_EXCLUDED_VALIDATION_ROWS:
        raise ValueError("Generator-safe validation exclusions must be unique")
    if set(frame["relative_wav_path"].astype(str)) & set(excluded_paths):
        raise ValueError("Generator-safe validation manifest still contains an excluded row")
    test_unchanged = protocol["test_unchanged"]
    if int(test_unchanged.get("rows", -1)) != EXPECTED_TEST_ROWS or int(
        test_unchanged.get("excluded_rows", -1)
    ) != 0:
        raise ValueError("Generator-safe validation protocol does not preserve the held-out test set")

    semantic_counterparts = [
        {
            "excluded_validation_relative_wav_path": str(
                row["excluded_validation_relative_wav_path"]
            ),
            "historical_train_relative_wav_path": str(
                row["historical_train_row"]["relative_wav_path"]
            ),
        }
        for row in counterparts
    ]
    return {
        "format_version": int(protocol["format_version"]),
        "protocol": _portable(protocol_path, project_root),
        "source_validation_manifest": protocol["source_validation_manifest"],
        "output_validation_manifest": protocol["output_validation_manifest"],
        "exact_duplicate_protocol": protocol["exact_duplicate_protocol"],
        "historical_manifest": protocol["historical_manifest"],
        "rule": protocol["rule"],
        "validation_counts": protocol["validation_counts"],
        "excluded_validation_relative_wav_paths": excluded_paths,
        "historical_train_counterparts": semantic_counterparts,
        "test_unchanged": protocol["test_unchanged"],
        "caveat": protocol["caveat"],
    }
