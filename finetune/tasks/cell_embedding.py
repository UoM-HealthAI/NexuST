"""NexuST cell embedding extraction."""
import argparse
import sys
from pathlib import Path

import scanpy as sc

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inference import run_inference


def embed(
    data_path: str,
    output_dir: str,
    ckpt: str,
    subsample: int = 100000,
    gpus: str = "0",
    max_cells: int = 1024,
    max_gene_len: int = 300,
    n_top_genes: int = 300,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_name = Path(data_path).stem

    adata = sc.read_h5ad(data_path)
    if subsample and adata.n_obs > subsample:
        sc.pp.subsample(adata, n_obs=subsample)

    # Select HVGs if more genes than n_top_genes
    if adata.n_vars > n_top_genes:
        n_orig = adata.n_vars
        # seurat_v3 works on raw counts directly
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3", span=0.3)
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"Selected {adata.n_vars} HVGs from {n_orig} genes")

    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    adata = run_inference(
        adata,
        ckpt=ckpt,
        gpus=gpus,
        max_cells=max_cells,
        max_gene_len=max_gene_len,
        obsm_key="X_nexust",
    )

    out_path = output_dir / f"nexust_{data_name}.h5ad"
    adata.write_h5ad(out_path)
    print(f"Saved: {out_path}")
    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", default="./output")
    parser.add_argument("--subsample", type=int, default=100000)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--n_top_genes", type=int, default=300, help="Number of HVGs to select")
    args = parser.parse_args()

    embed(
        args.data, args.output, args.ckpt, args.subsample, args.gpus,
        n_top_genes=args.n_top_genes,
    )
