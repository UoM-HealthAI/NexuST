import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import scanpy as sc
from tqdm import tqdm
from hilbertcurve.hilbertcurve import HilbertCurve

from data.tokenizer import Tokenizer


def hilbert_order(coords: np.ndarray, grid_size: int = 64) -> np.ndarray:
    """
    Sort cells by Hilbert curve order for spatial locality.

    Args:
        coords: (N, 2) float coordinates
        grid_size: Must be power of 2 (e.g. 32, 64, 128)

    Returns:
        Sorted indices that preserve spatial locality
    """
    eps = 1e-6
    x = coords[:, 0].astype(np.float32)
    y = coords[:, 1].astype(np.float32)

    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    dx = max(x1 - x0, eps)
    dy = max(y1 - y0, eps)

    xn = (x - x0) / dx
    yn = (y - y0) / dy

    qx = np.clip(np.round(xn * (grid_size - 1)), 0, grid_size - 1).astype(np.uint32)
    qy = np.clip(np.round(yn * (grid_size - 1)), 0, grid_size - 1).astype(np.uint32)

    p = int(np.log2(grid_size))
    hc = HilbertCurve(p, 2)

    h = np.array([hc.distance_from_point([int(xi), int(yi)]) for xi, yi in zip(qx, qy)], dtype=np.uint64)
    return np.argsort(h, kind="mergesort")


def sample_or_pad_genes(
    gene_ids: torch.Tensor,
    gene_values: torch.Tensor,
    max_gene_len: int,
    seed: int = 42,
    use_expression_weights: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad or subsample genes to exactly max_gene_len.

    Args:
        use_expression_weights: If True, sample with expression-based weights
            (prefer expressed genes). If False, uniform random permutation.
    """
    n_spots, n_genes = gene_ids.shape

    if n_genes == max_gene_len:
        return gene_ids, gene_values
    elif n_genes < max_gene_len:
        pad_len = max_gene_len - n_genes
        gene_ids = torch.cat([
            gene_ids,
            torch.zeros(n_spots, pad_len, dtype=gene_ids.dtype)
        ], dim=1)
        gene_values = torch.cat([
            gene_values,
            torch.zeros(n_spots, pad_len, dtype=gene_values.dtype)
        ], dim=1)
        return gene_ids, gene_values
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        if use_expression_weights:
            epsilon = 1e-6
            weights = torch.where(gene_values > 0, 1.0, epsilon)
            indices = torch.multinomial(weights, max_gene_len, replacement=False, generator=generator)
            return torch.gather(gene_ids, 1, indices), torch.gather(gene_values, 1, indices)
        else:
            indices = torch.randperm(n_genes, generator=generator)[:max_gene_len]
            return gene_ids[:, indices], gene_values[:, indices]


class FinetuneSlideContainer:
    def __init__(self, h5ad_path: str, tokenizer: Tokenizer):
        self.h5ad_path = h5ad_path
        adata = sc.read_h5ad(h5ad_path, backed='r')

        if 'spatial' in adata.obsm:
            self.coords = adata.obsm['spatial'][:].astype('float32')
        elif 'X_spatial' in adata.obsm:
            self.coords = adata.obsm['X_spatial'][:].astype('float32')
        else:
            raise KeyError(f"No spatial coordinates found in {h5ad_path}")

        self.gene_ids = [tokenizer.encode_gene(gene) for gene in adata.var_names]
        organ_str = adata.uns.get('organ', 'unknown')
        self.organ_id = tokenizer.encode_metadata('organ', organ_str)

        platform_str = adata.uns.get('platform', 'unknown')
        self.platform_id = tokenizer.encode_metadata('platform', platform_str)

        self._adata_backed = None

    def open_backed(self):
        if self._adata_backed is None:
            self._adata_backed = sc.read_h5ad(self.h5ad_path, backed='r')
        return self._adata_backed

    @property
    def n_spots(self) -> int:
        return self.coords.shape[0]


class TrainFinetuneDataset(Dataset):
    """
    Training dataset with online FPS + KNN sampling.
    - Each FOV contributes multiple patches based on its size
    - Small FOV (< n_spots): pad with zeros + return cell_padding_mask
    - Online sampling: different patches each epoch for data augmentation
    """
    def __init__(
        self,
        train_dir: str,
        tokenizer: Tokenizer,
        n_spots: int = 512,
        max_gene_len: int = 1000,
        seed: int = 42,
        device: str = 'cpu',
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_spots = n_spots
        self.max_gene_len = max_gene_len
        self.seed = seed
        self.epoch = 0
        self.train_dir = Path(train_dir)

        from data.sampling import PatchSampler
        self.patch_sampler = PatchSampler(n_spots=n_spots, device=device)

        self._load_slides()
        self._build_index_map()

    def _load_slides(self):
        self.h5ad_paths = sorted(self.train_dir.glob("*.h5ad"))
        if not self.h5ad_paths:
            raise ValueError(f"No h5ad files found in {self.train_dir}")

        self.slides: List[FinetuneSlideContainer] = []
        total_cells = 0

        for path in tqdm(self.h5ad_paths, desc="Loading train FOVs"):
            container = FinetuneSlideContainer(str(path), self.tokenizer)
            self.slides.append(container)
            total_cells += container.n_spots

        print(f"Loaded {len(self.slides)} FOVs, {total_cells:,} cells")

    def _build_index_map(self):
        self.index_map: List[Tuple[int, int, int]] = []  # (fov_idx, n_centers, center_idx)

        for fov_idx, slide in enumerate(self.slides):
            n_centers = max(1, slide.n_spots // self.n_spots)
            for center_idx in range(n_centers):
                self.index_map.append((fov_idx, n_centers, center_idx))

        print(f"Total {len(self.index_map)} patches")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _sample_or_pad_genes(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return sample_or_pad_genes(
            gene_ids, gene_values, self.max_gene_len,
            seed=self.seed + self.epoch, use_expression_weights=True)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fov_idx, n_centers, center_idx = self.index_map[idx]
        slide = self.slides[fov_idx]
        n_cells = slide.n_spots

        # Generate seed for this sample (different each epoch)
        sample_seed = self.seed + self.epoch * 100003 + idx

        # cell_padding_mask: True = padding, False = real cell
        if n_cells >= self.n_spots:
            patches = self.patch_sampler.fps_and_knn(
                slide.coords, n_centers=n_centers, seed=sample_seed
            )  # [n_centers, n_spots]
            spot_idx = patches[center_idx]  # [n_spots]
            cell_padding_mask = torch.zeros(self.n_spots, dtype=torch.bool)
        else:
            spot_idx = np.arange(n_cells)
            cell_padding_mask = torch.ones(self.n_spots, dtype=torch.bool)
            cell_padding_mask[:n_cells] = False

        spot_idx = np.asarray(spot_idx, dtype=np.int64)
        spot_idx_padded = np.full((self.n_spots,), -1, dtype=np.int64)
        spot_idx_padded[: spot_idx.shape[0]] = spot_idx

        # Load data
        coords = torch.from_numpy(slide.coords[spot_idx]).float()
        adata_backed = slide.open_backed()
        X = adata_backed.X[spot_idx, :]
        arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        gene_values = torch.from_numpy(arr.astype('float32'))

        gene_ids = torch.tensor(slide.gene_ids, dtype=torch.long)
        gene_ids = gene_ids.unsqueeze(0).expand(len(spot_idx), -1)
        gene_ids, gene_values = self._sample_or_pad_genes(gene_ids, gene_values)

        # Pad cells (preallocate then fill)
        k = len(spot_idx)
        coords_padded = torch.zeros(self.n_spots, 2, dtype=coords.dtype)
        gene_values_padded = torch.zeros(self.n_spots, self.max_gene_len, dtype=gene_values.dtype)
        gene_ids_padded = torch.zeros(self.n_spots, self.max_gene_len, dtype=gene_ids.dtype)
        coords_padded[:k] = coords
        gene_values_padded[:k] = gene_values
        gene_ids_padded[:k] = gene_ids

        return {
            'coords': coords_padded,              # [n_spots, 2]
            'gene_ids': gene_ids_padded,          # [n_spots, max_gene_len]
            'gene_values': gene_values_padded,   # [n_spots, max_gene_len]
            'cell_padding_mask': cell_padding_mask,        # [n_spots] True=pad, False=real
            'spot_idx': torch.from_numpy(spot_idx_padded).long(),  # [n_spots], -1 for padding
            'organ_id': slide.organ_id,
            'batch_label': torch.full((self.n_spots,), slide.platform_id, dtype=torch.long),
        }


class ValFinetuneDataset(Dataset):
    """
    Validation dataset for downstream tasks.
    Each FOV file is one or more items depending on size:
    - Small FOV (< max_cells): one item with all cells
    - Large FOV (> max_cells): split into chunks using Hilbert curve for spatial locality
    """
    def __init__(
        self,
        val_dir: str,
        tokenizer: Tokenizer,
        max_gene_len: int = 300,
        max_cells: int = 512,
        hilbert_grid_size: int = 512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_gene_len = max_gene_len
        self.max_cells = max_cells
        self.hilbert_grid_size = hilbert_grid_size
        self.val_dir = Path(val_dir)
        self._load_slides()
        self._build_chunks()

    def _load_slides(self):
        self.h5ad_paths = sorted(self.val_dir.glob("*.h5ad"))
        self.slides: List[FinetuneSlideContainer] = []

        for path in tqdm(self.h5ad_paths, desc="Loading val FOVs"):
            container = FinetuneSlideContainer(str(path), self.tokenizer)
            self.slides.append(container)

    def _build_chunks(self):
        """Split large FOVs into chunks using Hilbert curve ordering.
        
        index_map example:
        [
            (0, [0, 1, 2, ..., 299]),           # FOV 0, 全部 300 cells
            (1, [h0, h1, ..., h511]),           # FOV 1, chunk 0 (Hilbert 排序后的前 512)
            (1, [h512, h513, ..., h1023]),      # FOV 1, chunk 1
            (1, [h1024, h1025, ..., h1499]),    # FOV 1, chunk 2 (剩余 476 cells)
            (2, [h0, h1, ..., h511]),           # FOV 2, chunk 0
            (2, [h512, h513, ..., h799]),       # FOV 2, chunk 1 (剩余 288 cells)
        ]
        """
        # index_map: List of (slide_idx, cell_indices)
        self.index_map: List[Tuple[int, np.ndarray]] = []
        total_cells = 0

        for slide_idx, slide in enumerate(self.slides):
            n_cells = slide.n_spots
            total_cells += n_cells

            if n_cells <= self.max_cells:
                # Small FOV: use all cells as one chunk
                self.index_map.append((slide_idx, np.arange(n_cells)))
            else:
                # Large FOV: sort by Hilbert curve and split into chunks
                order = hilbert_order(slide.coords, grid_size=self.hilbert_grid_size)
                for start in range(0, n_cells, self.max_cells):
                    chunk_indices = order[start:start + self.max_cells]
                    self.index_map.append((slide_idx, chunk_indices))

    def _sample_or_pad_genes(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return sample_or_pad_genes(
            gene_ids, gene_values, self.max_gene_len,
            seed=42, use_expression_weights=False)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        slide_idx, cell_indices = self.index_map[idx]
        slide = self.slides[slide_idx]
        n_spots = len(cell_indices)

        # Load chunk data
        coords = torch.from_numpy(slide.coords[cell_indices]).float()

        adata_backed = slide.open_backed()
        X = adata_backed.X[cell_indices, :]
        arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        gene_values = torch.from_numpy(arr.astype('float32'))

        gene_ids = torch.tensor(slide.gene_ids, dtype=torch.long)
        gene_ids = gene_ids.unsqueeze(0).expand(n_spots, -1)

        gene_ids, gene_values = self._sample_or_pad_genes(gene_ids, gene_values)

        return {
            'gene_values': gene_values,   # [N, G]
            'coords': coords,              # [N, 2]
            'gene_ids': gene_ids,          # [N, G]
            'organ_id': slide.organ_id,
            'batch_label': torch.full((n_spots,), slide.platform_id, dtype=torch.long),
            'fov_idx': slide_idx,
            'chunk_idx': idx,
            'cell_indices': cell_indices,  # for mapping back predictions
            'fov_path': str(self.h5ad_paths[slide_idx]),
        }
