"""
Subset validation h5ad files to only keep HVG genes.

Usage:

python data/process/subset_hvg.py \
    -d corpus/val_processed \
    -o corpus/val_hvg \
    --n_hvg 300
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import scanpy as sc
from scipy import sparse
import gc


def subset_to_hvg(adata, n_hvg: int = 300, hvg_flavor: str = 'seurat_v3'):
    """
    Subset AnnData to only keep top HVG genes.

    Args:
        adata: Processed AnnData (already normalized and log1p)
        n_hvg: Number of HVG to keep
        hvg_flavor: Flavor for HVG selection

    Returns:
        Subsetted AnnData with only HVG genes
    """
    n_genes = adata.n_vars

    # Select HVG
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(n_hvg, n_genes),
        flavor=hvg_flavor,
    )

    # Subset to HVG only
    adata = adata[:, adata.var['highly_variable']].copy()

    return adata


def process_directory(data_dir: str, out_dir: str, n_hvg: int = 300):
    """Process all h5ad files in directory."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[subset_hvg] Input: {data_dir}")
    print(f"[subset_hvg] Output: {out_dir}")
    print(f"[subset_hvg] n_hvg: {n_hvg}")

    total_files = 0

    for platform_dir in sorted(data_dir.iterdir()):
        if not platform_dir.is_dir():
            continue

        platform = platform_dir.name
        files = sorted(platform_dir.glob("**/*.h5ad"))
        print(f"Found {len(files)} files in {platform}/")

        for f in tqdm(files, desc=f"Processing {platform}"):
            adata = sc.read_h5ad(f)
            original_shape = adata.shape

            adata = subset_to_hvg(adata, n_hvg=n_hvg)

            # Ensure sparse
            if not sparse.issparse(adata.X):
                adata.X = sparse.csr_matrix(adata.X)

            # Output path
            relative_path = f.relative_to(data_dir)
            output_path = out_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            adata.write_h5ad(output_path)

            if total_files == 0:
                print(f"\nFirst file: {original_shape} -> {adata.shape}")

            del adata
            gc.collect()
            total_files += 1

    print(f"\n[subset_hvg] Complete! Processed {total_files} files")


def main():
    parser = argparse.ArgumentParser(description='Subset h5ad files to HVG only')
    parser.add_argument('--data_dir', '-d', type=str, required=True)
    parser.add_argument('--out_dir', '-o', type=str, required=True)
    parser.add_argument('--n_hvg', type=int, default=300)

    args = parser.parse_args()
    process_directory(args.data_dir, args.out_dir, args.n_hvg)


if __name__ == '__main__':
    main()
