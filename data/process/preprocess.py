import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm
import scanpy as sc
from scipy import sparse
import gc
from anndata import AnnData

import sys
sys.path.append(str(Path(__file__).parent.parent))


class Preprocessor:
    """
    Preprocessor for spatial transcriptomics data.
    Gene IDs are encoded later in the dataset/tokenizer

    Processing pipeline:
        1. filter_to_vocab - filter genes to vocabulary
        2. filter_cell_by_counts (optional) - filter cells by minimum counts (based on vocab genes)
        3. normalize_total - cell-wise total count normalization
        4. log1p - log transformation
        5. subset_hvg (optional) - highly variable gene selection
    """
    def __init__(
        self,
        filter_cell_by_counts: Union[int, bool] = False,
        target_sum: float = 1e4,
        normalize_total: bool = True,
        log1p: bool = True,
        subset_hvg: Union[int, bool] = False,
        hvg_flavor: str = 'seurat_v3',
        filter_to_vocab: bool = True,
    ):
        """
        Initialize Preprocessor.

        Args:
            filter_cell_by_counts: Minimum counts per cell, or False to skip filtering
            target_sum: Target sum for normalize_total (default: 1e4)
            normalize_total: Whether to apply normalize_total
            log1p: Whether to apply log1p transformation
            subset_hvg: Number of HVG to select, or False to skip
            hvg_flavor: Flavor for HVG selection ('seurat_v3', 'cell_ranger', etc.)
            filter_to_vocab: Whether to filter genes to vocabulary
        """
        self.filter_cell_by_counts = filter_cell_by_counts
        self.target_sum = target_sum
        self.normalize_total = normalize_total
        self.log1p = log1p
        self.subset_hvg = subset_hvg
        self.hvg_flavor = hvg_flavor
        self.filter_to_vocab = filter_to_vocab

        # Load gene vocabulary for filtering (no ID encoding here)
        gene_vocab_path = Path(__file__).parent.parent / "gene_vocab.json"
        with gene_vocab_path.open("r") as f:
            gene_dict = json.load(f)
        self.vocab_genes = list(gene_dict.keys())
        assert len(self.vocab_genes) > 0, "Gene vocabulary is not loaded"

    def __call__(self, adata: AnnData) -> AnnData:
        adata = adata.copy()

        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()

        def _totals(mat):
            return np.asarray(mat.sum(axis=1)).ravel() if sparse.issparse(mat) else mat.sum(axis=1)
        
        # 1. Filter to vocabulary genes
        if self.filter_to_vocab:
            gene_mask = adata.var_names.isin(self.vocab_genes)
            adata = adata[:, gene_mask].copy()

        # Compute total_counts (after gene filtering, vocab genes only)
        total_counts = _totals(adata.layers["counts"])
        adata.obs["total_counts"] = total_counts
        adata.obs["log1p_total_counts"] = np.log1p(total_counts)

        # 2. Filter cells by counts (use total_counts_all)
        if isinstance(self.filter_cell_by_counts, int) and self.filter_cell_by_counts > 0:
            keep = adata.obs["total_counts"].values >= self.filter_cell_by_counts
            adata = adata[keep].copy()

        # 3. Normalize total
        if self.normalize_total:
            sc.pp.normalize_total(adata, target_sum=self.target_sum)

        # 4. Log1p
        if self.log1p:
            sc.pp.log1p(adata)

        # 5. Subset HVG
        if self.subset_hvg:
            try:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=self.subset_hvg if isinstance(self.subset_hvg, int) else None,
                    flavor=self.hvg_flavor,
                    layer="counts",
                )
            except TypeError:
                X_backup = adata.X
                adata.X = adata.layers["counts"].copy()
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=self.subset_hvg if isinstance(self.subset_hvg, int) else None,
                    flavor=self.hvg_flavor,
                )
                adata.X = X_backup
                
            adata = adata[:, adata.var.highly_variable].copy()
        if not sparse.issparse(adata.X):
            adata.X = sparse.csr_matrix(adata.X)

        return adata

    def process_directory(
        self,
        data_dir: str,
        out_dir: Optional[str] = None,
    ) -> Path:
        """
        Process all h5ad files in a directory, organized by platform subdirectories.

        Expected structure:
            data_dir/
            ├── cosmx/
            │   └── *.h5ad
            ├── merfish/
            │   └── *.h5ad
            └── xenium/
                └── *.h5ad

        Args:
            data_dir: Input directory containing platform subdirectories
            out_dir: Output directory (default: <data_dir>_processed)

        Returns:
            Path to output directory
        """
        data_dir = Path(data_dir)
        out_dir = Path(out_dir) if out_dir else data_dir.parent / f"{data_dir.name}_processed"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Preprocessor] Input directory: {data_dir}")
        print(f"[Preprocessor] Output directory: {out_dir}")

        total_files = 0

        # Iterate over platform subdirectories
        for platform_dir in sorted(data_dir.iterdir()):
            if not platform_dir.is_dir():
                continue

            platform = platform_dir.name  # e.g., "cosmx", "merfish", "xenium"

            # Find all h5ad files in this platform directory (recursive)
            files = sorted(platform_dir.glob("**/*.h5ad"))
            print(f"[Preprocessor] Found {len(files)} h5ad files in {platform}/")

            for f in tqdm(files, desc=f"Processing {platform}"):
                # Preserve directory structure in output
                relative_path = f.relative_to(data_dir)
                output_path = out_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Process
                adata = sc.read_h5ad(f)
                adata = self(adata)

                # transfer to sparse
                if not sparse.issparse(adata.X):
                    adata.X = sparse.csr_matrix(adata.X)

                # Save
                adata.write_h5ad(output_path)

                del adata
                gc.collect()

            total_files += len(files)

        print(f"\n[Preprocessor] Complete! Processed {total_files} files")
        return out_dir


def main():
    parser = argparse.ArgumentParser(
        description='Preprocessing pipeline for spatial transcriptomics data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', '-d', type=str, required=True,
                       help='Directory containing platform subdirectories (cosmx/, merfish/, xenium/)')
    parser.add_argument('--out_dir', '-o', type=str, default=None,
                       help='Output directory (default: <data_dir>_processed)')
    parser.add_argument('--target_sum', '-t', type=float, default=1e4,
                       help='Target sum for normalization')
    parser.add_argument('--filter_cell_by_counts', type=int, default=0,
                       help='Minimum counts per cell (0 to skip)')
    parser.add_argument('--no_normalize', action='store_true',
                       help='Skip normalize_total')
    parser.add_argument('--no_log1p', action='store_true',
                       help='Skip log1p transformation')
    parser.add_argument('--hvg', action='store_true',
                       help='Enable HVG selection (use 2000 genes)')

    args = parser.parse_args()

    preprocessor = Preprocessor(
        filter_cell_by_counts=args.filter_cell_by_counts if args.filter_cell_by_counts > 0 else False,
        target_sum=args.target_sum,
        normalize_total=not args.no_normalize,
        log1p=not args.no_log1p,
        subset_hvg=2000 if args.hvg else False,
    )

    preprocessor.process_directory(  
        data_dir=args.data_dir,
        out_dir=args.out_dir,
    )


if __name__ == '__main__':
    main()
