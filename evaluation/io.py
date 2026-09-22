"""Shared I/O utilities for NexuST tasks: CSV results."""

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def append_csv_row(
    csv_path: Path,
    columns: Sequence[str],
    row: Sequence,
    write_header: bool = False,
) -> None:
    """Append a single row to a CSV file, optionally writing the header first."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(columns)
        writer.writerow(row)


def save_summary_rows(
    csv_path: Path,
    columns: Sequence[str],
    results: List[Dict[str, float]],
    fixed_cols: Dict[str, str],
    metric_keys: Sequence[str],
    fmt: str = ".6f",
) -> None:
    """Append mean and std summary rows to a CSV file.

    Args:
        csv_path: Target CSV file (must already have per-seed rows).
        columns: Column names (must match the CSV header).
        results: List of dicts, one per seed, with at least ``metric_keys``.
        fixed_cols: Dict mapping column name -> fixed value (e.g. dataset, mode).
        metric_keys: Which keys in *results* to aggregate.
        fmt: Format spec for metric values (default ".6f").
    """
    for agg_name, agg_fn in [("mean", np.mean), ("std", np.std)]:
        row = []
        for col in columns:
            if col == "seed":
                row.append(agg_name)
            elif col in fixed_cols:
                row.append(fixed_cols[col])
            elif col in metric_keys:
                vals = [r[col] for r in results]
                row.append(f"{agg_fn(vals):{fmt}}")
            else:
                row.append("")
        append_csv_row(csv_path, columns, row)


def needs_header(csv_path: Path) -> bool:
    """Return True if the CSV file does not exist or is empty."""
    return not csv_path.exists() or csv_path.stat().st_size == 0
