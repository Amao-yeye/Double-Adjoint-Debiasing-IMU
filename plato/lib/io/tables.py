"""Robust small-table parsing helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def timestamp_unit_scale(values: np.ndarray, requested: str = "auto") -> tuple[float, str]:
    """Return multiplier to seconds and the inferred unit label."""

    if requested != "auto":
        mapping = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
        if requested not in mapping:
            raise ValueError(f"Unsupported timestamp unit: {requested}")
        return mapping[requested], requested
    median_abs = float(np.median(np.abs(values)))
    if median_abs > 1e17:
        return 1e-9, "ns"
    if median_abs > 1e14:
        return 1e-6, "us"
    if median_abs > 1e11:
        return 1e-3, "ms"
    return 1.0, "s"


def is_float_token(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def canonical_name(name: str) -> str:
    name = name.strip().lstrip("#").strip().lower()
    name = re.sub(r"\[[^\]]*\]", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def read_table(path: Path) -> tuple[Optional[list[str]], np.ndarray]:
    """Read a whitespace/comma table with an optional commented header."""

    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                candidate = line.lstrip("#").strip()
                if not rows and ("timestamp" in candidate.lower() or candidate.lower().startswith("time")):
                    line = candidate
                else:
                    continue
            fields = [item.strip() for item in re.split(r",|\s+", line) if item.strip()]
            if fields:
                rows.append(fields)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    has_header = any(not is_float_token(item) for item in rows[0])
    header = [canonical_name(item) for item in rows[0]] if has_header else None
    data_rows = rows[1:] if has_header else rows
    width = len(data_rows[0])
    numeric: list[list[float]] = []
    for row in data_rows:
        if len(row) < width:
            continue
        try:
            values = [float(item) for item in row[:width]]
        except ValueError:
            continue
        if np.all(np.isfinite(values)):
            numeric.append(values)
    if not numeric:
        raise ValueError(f"No numeric rows found in {path}")
    return header, np.asarray(numeric, dtype=float)


def find_column(header: Sequence[str], aliases: Sequence[str], label: str) -> int:
    alias_set = {canonical_name(item) for item in aliases}
    for idx, name in enumerate(header):
        if name in alias_set:
            return idx
    raise ValueError(f"Could not find {label} column in header {list(header)}")


def ensure_increasing(timestamps: np.ndarray, name: str) -> None:
    if len(timestamps) < 2:
        raise ValueError(f"{name} contains fewer than two timestamps")
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError(f"{name} timestamps are not strictly increasing")

