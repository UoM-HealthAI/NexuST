"""
Split downstream datasets into FOVs and train/val sets.
"""

from argparse import ArgumentParser
from pathlib import Path
import shutil

import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.model_selection import train_test_split

from finetune.utils.config import DATASETS


def scale_coords(adata: AnnData, spatial_key: str = "spatial") -> str:
    """Normalize spatial coords to [0, 100]."""
    key = spatial_key if spatial_key in adata.obsm else ("X_spatial" if "X_spatial" in adata.obsm else None)
    if key is None:
        raise KeyError("No spatial coordinates in adata.obsm ('spatial' or 'X_spatial').")

    coords = adata.obsm[key].astype("float32", copy=True)
    coords = coords - coords.min(axis=0)
    maxc = coords.max()
    if maxc > 0:
        coords = coords / maxc * 100.0
    adata.obsm[key] = coords
    if key != "spatial":
        adata.obsm["spatial"] = coords
    return key


def create_pseudo_fov(adata: AnnData, n_bins: int = 10, spatial_key: str = "spatial") -> np.ndarray:
    """Create pseudo FOV labels based on spatial coordinate grid."""
    coords = adata.obsm[spatial_key]

    x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_bins + 1)
    y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n_bins + 1)

    x_idx = np.clip(np.digitize(coords[:, 0], x_bins) - 1, 0, n_bins - 1)
    y_idx = np.clip(np.digitize(coords[:, 1], y_bins) - 1, 0, n_bins - 1)

    pseudo_fov = x_idx * n_bins + y_idx
    print(f"  Created {len(np.unique(pseudo_fov))} pseudo FOVs from {n_bins}x{n_bins} grid")

    return pseudo_fov


def filter_low_quality_cells(adata: AnnData, min_nonzero: int = 10) -> AnnData:
    """Filter cells with too few non-zero genes."""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    nonzero_count = (X != 0).sum(axis=1)
    keep_mask = nonzero_count >= min_nonzero
    n_filtered = (~keep_mask).sum()
    if n_filtered > 0:
        print(f"  Filtering {n_filtered} cells with < {min_nonzero} non-zero genes")
        adata = adata[keep_mask].copy()
    return adata


def select_hvg(adata: AnnData, n_top_genes: int = 300, flavor: str = "seurat") -> AnnData:
    """Select top HVG from the whole slide before splitting."""
    adata_copy = adata.copy()
    sc.pp.highly_variable_genes(
        adata_copy,
        n_top_genes=n_top_genes,
        flavor=flavor,
        subset=False,
    )

    hvg_mask = adata_copy.var["highly_variable"]
    n_selected = hvg_mask.sum()
    print(f"  Selected {n_selected} HVG from {adata.n_vars} genes")
    return adata[:, hvg_mask].copy()


def split_slide_by_fov(
    adata: AnnData,
    library_key: str,
    output_dir: Path,
    pseudo_fov_bins: int = 10,
    n_hvg: int = 300,
    imputation_min_nonzero: int = 10,
):
    """
    Split a single slide by FOV/sample/pseudo_fov.

    Steps:
        1. Select top HVG on the whole slide (ensures consistent gene set across FOVs)
        2. Filter low-quality cells
        3. Scale coordinates to [0, 100]
        4. Split by FOV and save each as h5ad
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Select HVG on whole slide first
    if n_hvg > 0 and adata.n_vars > n_hvg:
        adata = select_hvg(adata, n_top_genes=n_hvg)

    # Step 2: Filter low-quality cells
    adata = filter_low_quality_cells(adata, min_nonzero=imputation_min_nonzero)

    # Step 3: Scale coordinates
    scale_coords(adata, spatial_key="spatial")

    # Step 4: Create pseudo FOV if no library_key
    if library_key is None:
        pseudo_key = f"pseudo_fov_{pseudo_fov_bins}x{pseudo_fov_bins}"
        adata.obs[pseudo_key] = create_pseudo_fov(adata, n_bins=pseudo_fov_bins)
        library_key = pseudo_key

    # Step 5: Split and save
    fovs = adata.obs[library_key].unique()
    print(f"  Splitting into {len(fovs)} FOVs...")

    for fov in fovs:
        subslide = adata[adata.obs[library_key] == fov].copy()
        out_path = output_dir / f"{library_key}_{fov}.h5ad"
        # Never overwrite existing outputs
        if out_path.exists():
            continue
        subslide.write_h5ad(out_path)

    print(f"  Saved to: {output_dir}")


def split_all_by_fov(
    n_hvg: int = 300,
    imputation_min_nonzero: int = 10,
):
    """Split all datasets from val_processed to downstream by FOV."""
    for name, cfg in DATASETS.items():
        print(f"\n[{name}]")
        print(f"  Source: {cfg.val_h5ad_path}")

        if not cfg.val_h5ad_path.exists():
            print(f"  [SKIP] File not found")
            continue

        adata = sc.read_h5ad(cfg.val_h5ad_path)
        print(f"  Cells: {adata.n_obs:,}, Genes: {adata.n_vars:,}")

        if cfg.library_key is None:
            print(f"  Using pseudo FOV: {cfg.pseudo_fov_bins}x{cfg.pseudo_fov_bins} grid")

        split_slide_by_fov(
            adata,
            library_key=cfg.library_key,
            output_dir=cfg.downstream_dir,
            pseudo_fov_bins=cfg.pseudo_fov_bins,
            n_hvg=n_hvg,
            imputation_min_nonzero=imputation_min_nonzero,
        )

    print(f"\nAll done!")


def split_train_val(
    data_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Path, Path]:
    """Split FOV h5ad files into train/val directories (move files)."""
    adata_paths = list(data_dir.glob("*.h5ad"))
    if not adata_paths:
        raise RuntimeError(f"No h5ad files found in {data_dir}")

    train_paths, val_paths = train_test_split(adata_paths, test_size=test_size, random_state=random_state)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    def _move(src: Path, dst_dir: Path):
        dst = dst_dir / src.name
        # Never overwrite existing outputs
        if dst.exists():
            return
        shutil.move(str(src), str(dst))

    for path in train_paths:
        _move(path, train_dir)
    for path in val_paths:
        _move(path, val_dir)

    print(f"  Train: {len(train_paths)} files -> {train_dir}")
    print(f"  Val: {len(val_paths)} files -> {val_dir}")

    return train_dir, val_dir


def split_all_train_val(test_size: float = 0.2):
    for name, cfg in DATASETS.items():
        print(f"\n[{name}]")

        if not cfg.downstream_dir.exists():
            print(f"  [SKIP] Directory not found: {cfg.downstream_dir}")
            continue

        root_h5ads = list(cfg.downstream_dir.glob("*.h5ad"))
        if not root_h5ads:
            print(f"  [SKIP] No h5ad files to split")
            continue

        split_train_val(cfg.downstream_dir, test_size=test_size)

    print(f"\nAll done!")


def prepare_all_datasets(
    n_hvg: int = 300,
    test_size: float = 0.2,
    imputation_min_nonzero: int = 10,
):
    """
    Full pipeline: split by FOV then split into train/val.

    1. Read from val_processed
    2. Select HVG on whole slide
    3. Filter low-quality cells
    4. Split by FOV and save to downstream
    5. Split FOVs into train/val
    """
    print("Step 1: Split by FOV")
    split_all_by_fov(
        n_hvg=n_hvg,
        imputation_min_nonzero=imputation_min_nonzero,
    )

    print("\n" + "=" * 60)
    print("Step 2: Split into train/val")
    split_all_train_val(test_size=test_size)

    print("Done! Dataset structure:")
    for name, cfg in DATASETS.items():
        if cfg.downstream_dir.exists():
            n_train = len(list(cfg.train_dir.glob("*.h5ad"))) if cfg.train_dir.exists() else 0
            n_val = len(list(cfg.val_dir.glob("*.h5ad"))) if cfg.val_dir.exists() else 0
            print(f"  {name}: train={n_train}, val={n_val}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_hvg", type=int, default=0,
                        help="Number of HVG to select per slide (0 to disable)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of FOVs for validation")
    parser.add_argument("--min_nonzero", type=int, default=10,
                        help="Minimum non-zero genes to keep a cell")
    args = parser.parse_args()

    prepare_all_datasets(
        n_hvg=args.n_hvg,
        test_size=args.test_size,
        imputation_min_nonzero=args.min_nonzero,
    )
