import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from typing import List, Tuple
import scanpy as sc
from tqdm import tqdm


@torch.no_grad()
def farthest_point_sampling(points: torch.Tensor, n_samples: int, random_start: bool = True, generator=None) -> torch.Tensor:
    """
    Farthest Point Sampling (Pure PyTorch implementation)

    Args:
        points: (N, D) tensor of points
        n_samples: number of points to sample
        random_start: whether to start from a random point
        generator: optional torch.Generator for reproducibility

    Returns:
        indices: (n_samples,) tensor of selected indices
    """
    N, D = points.shape
    device = points.device
    if N == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    n_samples = min(int(n_samples), N)
    indices = torch.empty((n_samples,), dtype=torch.long, device=device)
    distances = torch.full((N,), float("inf"), device=device)

    # first point
    if random_start:
        first = torch.randint(0, N, (1,), device=device, generator=generator).item()
    else:
        first = 0
    indices[0] = first

    last = points[first]
    distances = torch.minimum(distances, ((points - last) ** 2).sum(dim=1))
    distances[first] = -float("inf")  # prevent re-picking

    for i in range(1, n_samples):
        idx = torch.argmax(distances)
        indices[i] = idx
        last = points[idx]
        distances = torch.minimum(distances, ((points - last) ** 2).sum(dim=1))
        distances[idx] = -float("inf")

    return indices


class PatchSampler:
    """ FPS (Farthest Point Sampling) + KNN based patch sampler """
    def __init__(self, n_spots: int = 512, device: str = 'cpu'):
        self.n_spots = n_spots
        self.device = device 

    def random_patches(self, coords: np.ndarray, n_centers: int, seed: int = None) -> np.ndarray:
        """Randomly sample fixed-size patches (n_spots cells) for each center."""
        n_points = len(coords)
        k = min(self.n_spots, n_points)
        rng = np.random.RandomState(seed) if seed is not None else np.random
        patches = [rng.choice(n_points, size=k, replace=False) for _ in range(int(n_centers))]
        return np.stack(patches, axis=0)  # [n_centers, k]

    def fps_and_knn(self, coords: np.ndarray, n_centers: int, seed: int = None) -> np.ndarray:
        """One-shot FPS center selection + batch KNN for all centers."""
        coords_t = torch.from_numpy(coords).float().to(self.device)
        n_points = len(coords_t)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        center_indices = farthest_point_sampling(
            coords_t,
            n_samples=n_centers,
            random_start=True,
            generator=generator
        )

        if len(center_indices) < n_centers:
            print(f"Warning: FPS returned {len(center_indices)} centers, requested {n_centers}")
            n_centers = len(center_indices)

        center_coords = coords_t[center_indices]  # [n_centers, 2]
        dist = torch.cdist(center_coords, coords_t)  # [n_centers, N]

        k = min(self.n_spots, n_points)
        _, knn_indices = torch.topk(dist, k=k, dim=1, largest=False)  # [n_centers, k]

        return knn_indices.cpu().numpy()  # [n_centers, n_spots]


class KMeansSubSlideSampler:
    """K-means based sub-slide sampler that partitions slides into spatially coherent clusters."""
    def __init__(
        self,
        target_cells_per_cluster: int = 5000,
        random_state: int = 42,
    ):
        self.target_cells = target_cells_per_cluster
        self.random_state = random_state

    def cluster_cells(self, coords: np.ndarray) -> np.ndarray:
        n_cells = len(coords)
        n_clusters = max(1, n_cells // self.target_cells)

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(10000, n_cells),
            max_iter=100,
            random_state=self.random_state,
            verbose=0,
        )

        return kmeans.fit_predict(coords)

    def get_subslides(self, labels: np.ndarray, min_cells: int = 0) -> List[np.ndarray]:
        n_clusters = len(np.unique(labels))
        sub_slides = []
        filtered_count = 0

        for cluster_id in range(n_clusters):
            cluster_mask = (labels == cluster_id)
            indices = np.where(cluster_mask)[0]

            # Filter out clusters that are too small
            if len(indices) >= min_cells:
                sub_slides.append(indices)
            else:
                filtered_count += 1

        if filtered_count > 0:
            print(f"  Filtered out {filtered_count} sub-slides with < {min_cells} cells")

        return sub_slides

    def split_slide(
        self,
        coords: np.ndarray,
        min_cells: int = 0,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        labels = self.cluster_cells(coords)
        sub_slides = self.get_subslides(labels, min_cells=min_cells)
        return labels, sub_slides


