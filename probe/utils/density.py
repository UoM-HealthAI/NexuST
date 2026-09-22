from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.neighbors import radius_neighbors_graph
from tqdm import tqdm


def compute_density(coords: np.ndarray, radius: float = 0.5) -> np.ndarray:
    """Count neighbors within *radius* for each cell."""
    conn = radius_neighbors_graph(
        coords, radius=radius, mode="connectivity", include_self=False
    )
    return np.asarray(conn.sum(axis=1), dtype=np.float32).ravel()
