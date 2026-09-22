from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import scanpy as sc
from tqdm import tqdm


def extract_niche_labels(
    adatas: list,
    niche_key: str = "X_niche_0",
) -> Tuple[np.ndarray, Dict]:
    """Extract proportion-normalized niche labels from pre-loaded adatas.

    Args:
        adatas: List of AnnData objects (already in memory).
        niche_key: Key in adata.obsm for niche counts.

    Returns:
        proportions: (N, n_cell_types) float32, rows sum to 1 (or 0 for isolated cells)
        metadata:    dict with "n_cell_types" and "cell_type_names"
    """
    labels_list = []
    center_ct_list = []
    n_cell_types = None
    cell_type_names = []
    ct_col = None

    for i, adata in enumerate(adatas):
        counts = adata.obsm[niche_key]
        if hasattr(counts, "toarray"):
            counts = counts.toarray()
        counts = np.asarray(counts, dtype=np.float32)

        # Proportion normalization: count / sum
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid div-by-zero
        labels_list.append(counts / row_sums)

        niche_info = adata.uns.get("niche_info", {})
        file_ct_names = list(niche_info.get("columns", []))

        if n_cell_types is None:
            cell_type_names = file_ct_names
            ct_col = niche_info.get("cell_type_col")
            n_cell_types = counts.shape[1]
        elif file_ct_names and file_ct_names != cell_type_names:
            raise ValueError(
                f"adata[{i}] niche column order mismatch: "
                f"{file_ct_names} vs {cell_type_names}"
            )

        center_ct_list.append(adata.obs[ct_col].values)

    return np.concatenate(labels_list, axis=0), {
        "n_cell_types": n_cell_types,
        "cell_type_names": cell_type_names,
        "center_cell_types": np.concatenate(center_ct_list, axis=0),
    }


def center_grouped_mae(val_preds, val_lab, center_types, *, cell_type_names=None):
    """MAE grouped by center cell type.

    If cell_type_names is given, center_types is treated as int ids into that
    list, and the returned dict is keyed by name with NaN entries for names
    that do not appear in center_types — this guarantees a fixed full-set
    output regardless of which cells happen to land in val. Without
    cell_type_names, the dict is keyed by whatever values appear in
    center_types (legacy behavior).
    """
    center_types = np.asarray(center_types)

    if cell_type_names is None:
        unique_types = np.unique(center_types)
        results = {}
        for ct in unique_types:
            mask = center_types == ct
            if not mask.any():
                continue
            abs_err = np.abs(val_preds[mask] - val_lab[mask])
            results[ct] = {
                "mae": abs_err.mean(),
                "mae_per_target": abs_err.mean(axis=0),
            }
        return results

    n_targets = val_preds.shape[1]
    results = {}
    for idx, name in enumerate(cell_type_names):
        mask = center_types == idx
        if mask.any():
            abs_err = np.abs(val_preds[mask] - val_lab[mask])
            results[name] = {
                "mae": float(abs_err.mean()),
                "mae_per_target": abs_err.mean(axis=0),
            }
        else:
            results[name] = {
                "mae": float("nan"),
                "mae_per_target": np.full(n_targets, np.nan),
            }
    return results
