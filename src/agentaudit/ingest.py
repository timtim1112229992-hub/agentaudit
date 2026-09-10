"""Reading and fixing the analysed record set.

Every table is hashed on the way in. The digests are the only description of the
corpus that leaves the machine, and they are sufficient to prove that a later run
examined the same records without disclosing any of them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import COLUMN_MAP, SETTINGS, SYNTHETIC_DIR, TABLES


def _canonical(row: pd.Series) -> str:
    return json.dumps({k: ("" if pd.isna(v) else str(v)) for k, v in sorted(row.items())},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_frame(df: pd.DataFrame) -> dict:
    """Per-record and aggregate digests. Order independent by construction."""
    per = sorted(hashlib.sha256(_canonical(r).encode("utf-8")).hexdigest() for _, r in df.iterrows())
    agg = hashlib.sha256("".join(per).encode("ascii")).hexdigest()
    return {"n_records": int(len(df)), "aggregate_sha256": agg, "record_sha256": per}


def _source_dir() -> Path:
    return SETTINGS.data_dir or SYNTHETIC_DIR


def load_tables() -> tuple[dict[str, pd.DataFrame], dict]:
    """Load every configured table, restricted to the permitted columns."""
    root = _source_dir()
    frames: dict[str, pd.DataFrame] = {}
    digests: dict[str, dict] = {}
    for name, filename in TABLES.items():
        path = root / filename
        if not path.exists():
            path = path.with_suffix(".csv")
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
        keep = [c for c in COLUMN_MAP[name] if c in df.columns]
        df = df[keep].copy()
        frames[name] = df
        digests[name] = digest_frame(df)
    meta = {
        "source": "synthetic" if SETTINGS.using_synthetic else "restricted",
        "tables": digests,
        "column_map": {k: [c for c in v if c in frames[k].columns] for k, v in COLUMN_MAP.items()},
    }
    return frames, meta
