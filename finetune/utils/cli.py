"""Shared CLI infrastructure for finetune tasks."""

import argparse
import csv
from pathlib import Path



def add_shared_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments common to all finetune tasks."""
    # Data
    parser.add_argument('--data_path', type=str, required=True)

    # Training
    parser.add_argument("--phase", choices=["sweep", "final"], default="final")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulate_grad_batches', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--max_gene_len', type=int, default=300)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    # Compute
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--accelerator', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--precision', default='bf16-mixed')
    parser.add_argument('--logger', choices=['csv', 'wandb'], default='csv')

    # Logging
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--project', type=str, default='HiGeST-finetune')
    parser.add_argument('--ckpt_root', type=str, default='checkpoints/finetune')

    return parser


def build_ckpt_dir(ckpt_root: str, *segments: str) -> Path:
    """Build checkpoint directory from root + path segments, creating it if needed.

    Example:
        build_ckpt_dir(args.ckpt_root, f"cls_{timestamp}", mode, dataset, f"seed{seed}")
    """
    ckpt_dir = Path(ckpt_root).joinpath(*segments)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def append_csv(output_dir: str | Path, filename: str,
               header: list[str], row: list) -> Path:
    """Append a row to a CSV file, writing header if the file is new.

    Caller is responsible for rank checking (only call from rank 0).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    return csv_path


def append_per_group_csv(
    output_dir: str | Path,
    filename_prefix: str,
    index_cols: list[str],
    index_values: list,
    metric_names: list[str],
    per_group: dict[str, dict[str, float]],
    float_fmt: dict[str, str] | None = None,
    default_fmt: str = '.6f',
) -> list[Path]:
    """Write one CSV per metric: ``{filename_prefix}_{metric}.csv``.

    Rows = one per run (identified by ``index_values``), columns = group names.
    Columns are stable across runs: sorted alphabetically from ``per_group`` keys
    on first write. Later runs with extra/missing groups append 'nan' for absent
    columns rather than rewriting the header.

    Only writes groups that are present both in ``metric_names`` and in the
    per-group dicts. Use when groups are a small, enumerable set (celltypes,
    niche centers). Not suitable for high-cardinality groups (e.g. per-gene).

    Args:
        output_dir: Directory to write CSVs into (created if missing).
        filename_prefix: e.g. ``f"{dataset}_center"`` → ``{dataset}_center_mse.csv``.
        index_cols: Header names for the leading index columns (e.g.
            ``['dataset', 'mode', 'radius_idx', 'radius', 'seed']``).
        index_values: Row values matching ``index_cols``, same length.
        metric_names: Which metrics from per_group[*] to emit (one CSV each).
        per_group: ``{group_name: {metric: float_or_nan, ...}, ...}``.
        float_fmt: Per-metric format override (e.g. ``{'r2': '.4f'}``).
        default_fmt: Format for metrics without an override.

    Returns:
        List of written paths, one per metric.
    """
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    float_fmt = float_fmt or {}
    written: list[Path] = []

    for metric in metric_names:
        path = output_dir / f"{filename_prefix}_{metric}.csv"
        file_exists = path.exists() and path.stat().st_size > 0
        fmt = float_fmt.get(metric, default_fmt)

        if file_exists:
            with open(path, 'r', newline='') as f:
                existing_header = next(csv.reader(f), None)
            group_names = existing_header[len(index_cols):] if existing_header else sorted(per_group.keys())
        else:
            group_names = sorted(per_group.keys())

        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(list(index_cols) + group_names)
            row = list(index_values)
            for g in group_names:
                v = per_group.get(g, {}).get(metric)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    row.append('nan')
                else:
                    row.append(f'{v:{fmt}}')
            writer.writerow(row)
        written.append(path)

    return written
