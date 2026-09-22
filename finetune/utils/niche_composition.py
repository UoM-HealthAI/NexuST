from pathlib import Path
from typing import Optional
import sys
import gc
import argparse

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import scanpy as sc
from anndata import AnnData
from pandas import get_dummies
from scipy.sparse import csr_matrix
from sklearn.neighbors import radius_neighbors_graph
from tqdm import tqdm
from pandas import Categorical


def _normalize_radii(radius: list[float] | float) -> list[float]:
    if isinstance(radius, (int, float)):
        radius = [radius]
    return [float(r) for r in radius]


def _niche_keys(niche_key: str, radii: list[float]) -> list[str]:
    return [f"{niche_key}_{i}" for i in range(len(radii))]


def _matching_niche_config(
    adata: AnnData,
    niche_key: str,
    radii: list[float],
    cell_type_col: str,
    spatial_key: str,
    expected_columns: list | None = None,
) -> bool:
    niche_info = adata.uns.get("niche_info")
    if not isinstance(niche_info, dict):
        return False

    if niche_info.get("niche_key") != niche_key:
        return False
    if niche_info.get("cell_type_col") != cell_type_col:
        return False
    if niche_info.get("spatial_key") != spatial_key:
        return False
    if expected_columns is not None and niche_info.get("columns") != list(expected_columns):
        return False

    stored_radii = niche_info.get("radii")
    if stored_radii is None:
        stored_radii = []
        n_radii = niche_info.get("n_radii", 0)
        for i in range(n_radii):
            niche_i = niche_info.get(f"niche_{i}", {})
            if "radius" not in niche_i:
                return False
            stored_radii.append(float(niche_i["radius"]))

    if len(stored_radii) != len(radii):
        return False

    if not np.allclose(np.asarray(stored_radii, dtype=float), np.asarray(radii, dtype=float)):
        return False

    return all(key in adata.obsm for key in _niche_keys(niche_key, radii))


def _clear_existing_niche_entries(adata: AnnData, niche_key: str) -> None:
    # Remove old single-radius key (e.g. "X_niche") and indexed keys (e.g. "X_niche_0")
    if niche_key in adata.obsm:
        del adata.obsm[niche_key]
    stale_keys = [key for key in adata.obsm.keys() if key.startswith(f"{niche_key}_")]
    for key in stale_keys:
        del adata.obsm[key]


def niche_composition(
    adata: AnnData,
    cell_type_col: str,
    radius: list[float] | float = 0.5,
    spatial_key: str = "spatial",
    niche_key: str = "X_niche",
    global_cell_types: list = None,
) -> list:
    """
    Compute spatial neighborhood cell type composition for each cell at multiple radii.

    Args:
        adata: AnnData object with cell_type_col in obs and spatial coordinates in obsm
        cell_type_col: Column name in adata.obs containing cell type labels
        radius: Single radius or list of radii for multi-scale niche computation.
               Each radius produces a separate obsm entry: X_niche_0, X_niche_1, ...
        spatial_key: Key in adata.obsm for spatial coordinates
        niche_key: Base key in adata.obsm (entries stored as {niche_key}_0, {niche_key}_1, ...)
        global_cell_types: Global list of cell types to use for one-hot encoding.
                          If provided, ensures consistent column order across all files.
                          If None, uses cell types present in this file only.

    Returns:
        List of niche composition sparse matrices, one per radius.
    """
    radius = _normalize_radii(radius)

    if global_cell_types is not None:
        cat = Categorical(adata.obs[cell_type_col], categories=global_cell_types)
        one_hot_ct = get_dummies(cat, dtype=float, sparse=True)
        cell_types = global_cell_types
    else:
        one_hot_ct = get_dummies(adata.obs[cell_type_col], dtype=float, sparse=True)
        cell_types = list(one_hot_ct.columns)

    coords = adata.obsm[spatial_key]
    one_hot_sparse = csr_matrix(one_hot_ct)
    _clear_existing_niche_entries(adata, niche_key)

    uns_niche = {
        "columns": cell_types,
        "cell_type_col": cell_type_col,
        "niche_key": niche_key,
        "spatial_key": spatial_key,
        "n_radii": len(radius),
        "radii": radius,
        "obsm_keys": _niche_keys(niche_key, radius),
    }

    results = []
    for i, r in enumerate(radius):
        connectivities = radius_neighbors_graph(
            coords, radius=r, mode="connectivity", include_self=False
        )
        niche = connectivities @ one_hot_sparse
        adata.obsm[f"{niche_key}_{i}"] = niche
        uns_niche[f"niche_{i}"] = {
            "radius": float(r),
            "mean_neighbors": float(np.mean(niche.sum(axis=1))),
        }
        results.append(niche)

    adata.uns["niche_info"] = uns_niche
    return results


def niche_dir(
    data_dir: str,
    cell_type_col: str,
    radius: list[float] | float = 0.5,
    spatial_key: str = "spatial",
    niche_key: str = "X_niche",
    force: bool = False,
    global_cell_types: list = None,
    collect_global_cell_types: bool = True,
) -> None:
    """
    Recursively find all h5ad files in a directory and compute niche composition for each.

    Args:
        data_dir: Root directory containing h5ad files (recursive search)
        cell_type_col: Column name in adata.obs containing cell type labels
        radius: Single radius or list of radii for multi-scale niche computation
        spatial_key: Key in adata.obsm for spatial coordinates
        niche_key: Base key in adata.obsm (entries stored as {niche_key}_0, ...)
        force: If True, recompute even if niche already exists
        global_cell_types: Global list of cell types. If provided, use this list for all files.
        collect_global_cell_types: If True and global_cell_types is None, first scan all files
                                   to collect unique cell types, then use that for encoding.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Recursively find all h5ad files
    h5ad_files = sorted(data_dir.rglob("*.h5ad"))
    if not h5ad_files:
        return

    # Collect global cell types if needed
    if global_cell_types is None and collect_global_cell_types:
        all_cell_types = set()
        for h5ad_path in tqdm(h5ad_files, desc="Scanning cell types"):
            adata = sc.read_h5ad(h5ad_path, backed='r')
            if cell_type_col in adata.obs:
                all_cell_types.update(adata.obs[cell_type_col].unique())
            del adata
        global_cell_types = sorted(all_cell_types)

    for h5ad_path in tqdm(h5ad_files, desc="Processing files"):
        # Load adata
        adata = sc.read_h5ad(h5ad_path)

        # Check if cell_type_col exists
        if cell_type_col not in adata.obs:
            del adata
            gc.collect()
            continue

        expected_columns = (
            list(global_cell_types)
            if global_cell_types is not None
            else list(get_dummies(adata.obs[cell_type_col], dtype=float, sparse=True).columns)
        )

        radii = _normalize_radii(radius)
        if not force and _matching_niche_config(
            adata,
            niche_key=niche_key,
            radii=radii,
            cell_type_col=cell_type_col,
            spatial_key=spatial_key,
            expected_columns=expected_columns,
        ):
            del adata
            gc.collect()
            continue

        # Compute niche composition
        niche_composition(
            adata=adata,
            cell_type_col=cell_type_col,
            radius=radius,
            spatial_key=spatial_key,
            niche_key=niche_key,
            global_cell_types=global_cell_types,
        )

        # Save back to file
        adata.write_h5ad(h5ad_path)
        del adata
        gc.collect()


def niche_from_config(
    dataset_name: Optional[str] = None,
    spatial_key: str = "spatial",
    niche_key: str = "X_niche",
    force: bool = False,
    global_cell_types: list = None,
    collect_global_cell_types: bool = True,
) -> None:
    from finetune.utils.config import DATASETS

    if dataset_name is not None:
        if dataset_name not in DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASETS. "
                f"Available: {list(DATASETS.keys())}"
            )
        datasets = {dataset_name: DATASETS[dataset_name]}
    else:
        datasets = DATASETS

    for name, cfg in datasets.items():
        niche_dir(
            data_dir=cfg.downstream_dir,
            cell_type_col=cfg.label_col,
            radius=cfg.niche_radius,
            spatial_key=spatial_key,
            niche_key=niche_key,
            force=force,
            global_cell_types=global_cell_types,
            collect_global_cell_types=collect_global_cell_types,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute niche composition from dataset config")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--radii", type=float, nargs="+", default=None)
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name in finetune.utils.config.DATASETS")
    parser.add_argument("--spatial_key", type=str, default="spatial", help="Key in adata.obsm for spatial coordinates")
    parser.add_argument("--niche_key", type=str, default="X_niche", help="Base key for saving niche matrices in adata.obsm")
    parser.add_argument("--force", action="store_true", help="Recompute niche even if matching cached results exist")
    parser.add_argument(
        "--no_collect_global_cell_types",
        action="store_true",
        help="Disable the initial scan that builds a global cell-type ordering across files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.data_dir is not None:
        if args.label_col is None or args.radii is None:
            raise ValueError("--data_dir requires --label_col and --radii")
        niche_dir(args.data_dir, args.label_col, args.radii,
                  spatial_key=args.spatial_key, niche_key=args.niche_key,
                  force=args.force,
                  collect_global_cell_types=not args.no_collect_global_cell_types)
    else:
        niche_from_config(
            dataset_name=args.dataset_name,
            spatial_key=args.spatial_key,
            niche_key=args.niche_key,
            force=args.force,
            collect_global_cell_types=not args.no_collect_global_cell_types,
        )
