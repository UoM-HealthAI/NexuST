"""Data loading utilities."""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scanpy as sc
from anndata import AnnData
from tqdm import tqdm


def load_h5ads_from_dir(dir_path: Union[str, Path]) -> AnnData:
    """Load and concatenate all .h5ad files from a directory."""
    paths = sorted(Path(dir_path).glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No .h5ad files in {dir_path}")
    print(f"Loading {len(paths)} h5ad files from {dir_path}")
    adatas = [sc.read_h5ad(p) for p in paths]
    return sc.concat(adatas, join="inner")


def embed_h5ad_dir(
    adapter,
    data_dir: Union[str, Path],
    spatial_key: str = "spatial",
) -> Tuple[np.ndarray, List[AnnData]]:
    """Embed all h5ad files in a directory and return (embeddings, adatas).

    The adatas are kept in memory so subclasses can extract task-specific
    labels without re-reading from disk.

    Returns:
        embeddings: (N, d_model) float32
        adatas:     list of AnnData (one per h5ad file, same order)
    """
    data_dir = Path(data_dir)
    h5ad_files = sorted(data_dir.rglob("*.h5ad"))
    if not h5ad_files:
        raise FileNotFoundError(f"No h5ad files found in {data_dir}")

    emb_list = []
    adata_list = []

    for path in tqdm(h5ad_files, desc=f"Embedding {data_dir.name}"):
        adata = sc.read_h5ad(path)
        emb = adapter.embed(adata, spatial_key=spatial_key)
        emb_list.append(emb)
        adata_list.append(adata)

    return np.concatenate(emb_list, axis=0), adata_list


def extract_expression_labels(
    adatas: List[AnnData],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Extract gene expression from adata.X as dense float32 array.

    Args:
        adatas: list of AnnData objects (from embed_h5ad_dir).

    Returns:
        expression: (N_cells, n_genes) float32 dense array
        metadata:   {"n_genes": int, "gene_names": list[str]}
    """
    expr_list = []
    gene_names = None

    for i, adata in enumerate(adatas):
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        expr_list.append(X)

        current_genes = list(adata.var_names)
        if gene_names is None:
            gene_names = current_genes
        elif current_genes != gene_names:
            raise ValueError(
                f"adata[{i}] gene order mismatch: "
                f"expected {len(gene_names)} genes, got {len(current_genes)}"
            )

    return np.concatenate(expr_list, axis=0), {
        "n_genes": len(gene_names),
        "gene_names": gene_names,
    }
