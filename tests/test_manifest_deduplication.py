from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_groups_keep_recording_ids_together() -> None:
    module = _load_script("create_splits", "01_create_splits.py")
    rows = pd.DataFrame(
        [
            {"id": "same-id"},
            {"id": "same-id"},
            {"id": "other-id"},
            {"id": "independent"},
        ]
    )
    grouped = module.assign_split_groups(rows)
    assert grouped.loc[:1, "split_group"].nunique() == 1
    assert grouped.loc[2, "split_group"] != grouped.loc[0, "split_group"]
    assert grouped.loc[3, "split_group"] != grouped.loc[0, "split_group"]
