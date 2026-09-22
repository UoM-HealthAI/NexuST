import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Optional, Dict
import scanpy as sc
from tqdm import tqdm
from pathlib import Path

from data.tokenizer import Tokenizer


class SlideContainer:
    """Container for a single slide with lazy loading"""
    def __init__(self, h5ad_path: str, tokenizer: Tokenizer):
        self.h5ad_path = h5ad_path
        adata = sc.read_h5ad(h5ad_path, backed='r')

        if 'spatial' in adata.obsm:
            coords = adata.obsm['spatial'][:].astype('float32')
        elif 'X_spatial' in adata.obsm:
            coords = adata.obsm['X_spatial'][:].astype('float32')
        else:
            raise KeyError(f"No spatial coordinates found in {h5ad_path}")

        coords[:, 0] = coords[:, 0] - coords[:, 0].min()
        coords[:, 1] = coords[:, 1] - coords[:, 1].min()
        maxc = coords.max()
        if maxc > 0:
            coords = coords / maxc * 100.0
        self.coords = coords

        self.gene_ids = [tokenizer.encode_gene(gene) for gene in adata.var_names]

        organ_str = adata.uns.get('organ', 'unknown')
        self.organ_id = tokenizer.encode_metadata('organ', organ_str)

        platform_str = adata.uns.get('platform', 'unknown')
        self.platform_id = tokenizer.encode_metadata('platform', platform_str)

        # Load log1p_total_counts for baseline comparison
        if 'log1p_total_counts' in adata.obs:
            self.log1p_total_counts = adata.obs['log1p_total_counts'].values.astype('float32')
        elif 'total_counts' in adata.obs:
            self.log1p_total_counts = np.log1p(adata.obs['total_counts'].values).astype('float32')
        else:
            self.log1p_total_counts = None

        self._adata_backed = None

    def open_backed(self):
        """Lazily open a backed='r' AnnData handle."""
        if self._adata_backed is None:
            self._adata_backed = sc.read_h5ad(self.h5ad_path, backed='r')
        return self._adata_backed

    @property
    def n_spots(self) -> int:
        return self.coords.shape[0]


class NexuSTDataset(Dataset):
    def __init__(
        self,
        patch_index_path: str,
        tokenizer: Tokenizer,
    ):
        """
        Args:
            patch_index_path: Path to precomputed patch_index.pt file
            tokenizer: Tokenizer for gene/metadata encoding
        """
        super().__init__()
        self.tokenizer = tokenizer

        self._load_patch_index(patch_index_path)
        self._build_slide_containers()

        self.fps_run_idx = 0

    def _load_patch_index(self, patch_index_path: str):
        """Load precomputed patch index from file."""
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(f"Loading patch index from {patch_index_path}")
        else:
            print(f"Loading patch index from {patch_index_path}")
            
        patch_index = torch.load(patch_index_path, weights_only=False)

        self.metadata = patch_index['metadata']
        self.n_fps_runs = self.metadata['n_fps_runs']
        self.n_centers = self.metadata['n_centers']
        self.n_spots = self.metadata['n_spots']

        # Build flat list of (slide_idx, subslide_idx) for indexing
        # Each subslide has n_centers patches
        self.patch_index_data = patch_index['slides']

        # Build index mapping: global_patch_idx -> (slide_idx, subslide_idx, center_idx)
        self.index_map = []
        for slide_idx, slide_data in enumerate(self.patch_index_data):
            for subslide_idx, subslide in enumerate(slide_data['subslides']):
                for center_idx in range(self.n_centers):
                    self.index_map.append((slide_idx, subslide_idx, center_idx))

        total_subslides = sum(len(s['subslides']) for s in self.patch_index_data)
        print(f"{len(self.patch_index_data)} slides, {total_subslides} sub-slides,"
              f"{len(self.index_map)} patches, {self.n_fps_runs} FPS runs")

    def _build_slide_containers(self):
        """Build SlideContainer for each slide in patch index."""
        self.slides: List[SlideContainer] = []
        self.path_to_slide_idx: Dict[str, int] = {}

        for slide_idx, slide_data in enumerate(tqdm(
            self.patch_index_data, desc="Loading slides"
        )):
            path = slide_data['path']
            container = SlideContainer(path, self.tokenizer)
            self.slides.append(container)
            self.path_to_slide_idx[path] = slide_idx

    def set_epoch(self, epoch: int):
        """
        Set current epoch to rotate through FPS runs.
        Call this at the start of each epoch.

        Args:
            epoch: Current epoch number
        """
        self.fps_run_idx = epoch % self.n_fps_runs
        print(f"[NexuSTDataset] Epoch {epoch}: using FPS run {self.fps_run_idx}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        slide_idx, subslide_idx, center_idx = self.index_map[idx]

        # Get patch indices from precomputed data
        # fps_patches shape: [n_fps_runs, n_centers, n_spots]
        subslide = self.patch_index_data[slide_idx]['subslides'][subslide_idx]
        spot_idx = subslide['fps_patches'][self.fps_run_idx, center_idx]  # [n_spots]

        # Load data from slide
        slide = self.slides[slide_idx]
        coords = torch.from_numpy(slide.coords[spot_idx]).float()

        adata_backed = slide.open_backed()
        X_sub = adata_backed.X[spot_idx, :]  # [n_spots, n_genes]

        # Convert to dense tensor
        arr = X_sub.toarray() if hasattr(X_sub, "toarray") else np.asarray(X_sub)
        gene_values = torch.from_numpy(arr.astype('float32'))

        gene_ids = torch.tensor(slide.gene_ids, dtype=torch.long).unsqueeze(0).expand(self.n_spots, -1)

        batch = {
            'gene_values': gene_values,  # [n_spots, n_genes]
            'coords': coords,            # [n_spots, 2]
            'gene_ids': gene_ids,        # [n_spots, n_genes]
            'batch_label': torch.full((self.n_spots,), slide.platform_id, dtype=torch.long),
            'organ_id': slide.organ_id,
        }

        # Add log1p_total_counts if available
        if slide.log1p_total_counts is not None:
            batch['log1p_total_counts'] = torch.from_numpy(slide.log1p_total_counts[spot_idx])

        return batch
