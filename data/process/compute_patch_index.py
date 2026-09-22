from typing import List
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import h5py
from tqdm import tqdm
from data.sampling import KMeansSubSlideSampler, PatchSampler


def read_spatial_coords(path: str) -> np.ndarray:
    """Read spatial coordinates directly from h5ad using h5py, avoiding uns parsing issues."""
    with h5py.File(path, 'r') as f:
        obsm = f['obsm']
        if 'spatial' in obsm:
            return obsm['spatial'][...].astype('float32')
        elif 'X_spatial' in obsm:
            return obsm['X_spatial'][...].astype('float32')
        else:
            return None


def compute_patch_index(
    h5ad_paths: List[str],
    output_path: str = 'patch_index.pt',
    n_fps_runs: int = 15,
    n_centers: int = 12,
    n_spots: int = 512,
    target_cells_per_subslide: int = 5000,
    device: str = 'cuda',
    base_seed: int = 42,
):
    """
    Precompute KMeans sub-slides and multiple FPS runs for all slides.
    Output cache structure:
        {
            'metadata': {n_fps_runs, n_centers, n_spots, base_seed, ...},
            'slides': [
                {
                    'path': str,
                    'subslides': [
                        {
                            'cell_indices': [N_subslide],
                            'fps_patches': [n_fps_runs, n_centers, n_spots]
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    """
    kmeans_sampler = KMeansSubSlideSampler(
        target_cells_per_cluster=target_cells_per_subslide,
        random_state=base_seed
    )
    patch_sampler = PatchSampler(n_spots=n_spots, device=device)
    
    # Use a dedicated generator for reproducibility across runs
    rng = np.random.RandomState(base_seed)

    cache = {
        'metadata': {
            'n_fps_runs': n_fps_runs,
            'n_centers': n_centers,
            'n_spots': n_spots,
            'target_cells_per_subslide': target_cells_per_subslide,
            'base_seed': base_seed,
        },
        'slides': []
    }

    for path in tqdm(h5ad_paths, desc="Processing slides"):
        coords = read_spatial_coords(path)
        if coords is None:
            print(f"Warning: No spatial coords in {path}, skipping")
            continue

        # Normalize coordinates to [0, 100]
        coords[:, 0] = coords[:, 0] - coords[:, 0].min()
        coords[:, 1] = coords[:, 1] - coords[:, 1].min()
        maxc = coords.max()
        if maxc > 0:
            coords = coords / maxc * 100.0

        # KMeans clustering
        _, subslides = kmeans_sampler.split_slide(coords, min_cells=n_spots)

        slide_data = {
            'path': path,
            'subslides': []
        }

        for subslide_indices in subslides:
            subslide_coords = coords[subslide_indices]

            # Multiple FPS runs with different random starts
            fps_patches_all_runs = []
            for run_idx in range(n_fps_runs):
                # FPS + KNN
                # Explicitly control seed for diversity across runs using the base generator
                seed = rng.randint(0, 2**31 - 1)
                patch_local_indices = patch_sampler.fps_and_knn(
                    subslide_coords, n_centers=n_centers, seed=seed
                )
                # Convert to global indices
                patch_global_indices = subslide_indices[patch_local_indices]
                fps_patches_all_runs.append(patch_global_indices)

            # Stack: [n_fps_runs, n_centers, n_spots]
            fps_patches = np.stack(fps_patches_all_runs, axis=0)

            slide_data['subslides'].append({
                'cell_indices': subslide_indices,
                'fps_patches': fps_patches,
            })

        cache['slides'].append(slide_data)
        print(f"  {path}: {len(coords):,} cells -> {len(subslides)} sub-slides")

    # Save cache
    torch.save(cache, output_path)
    print(f"\nSaved cache to {output_path}")

    # Print summary
    total_subslides = sum(len(s['subslides']) for s in cache['slides'])
    total_patches = total_subslides * n_centers * n_fps_runs
    print(f"Summary: {len(cache['slides'])} slides, {total_subslides} sub-slides, "
          f"{total_patches:,} total patches ({n_fps_runs} runs)")

    return cache


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .h5ad files')
    parser.add_argument('--output_path', type=str, default=None, help='Output path (default: data_dir/patch_index.pt)')
    parser.add_argument('--n_fps_runs', type=int, default=15, help='Number of FPS runs')
    parser.add_argument('--n_centers', type=int, default=12, help='Number of centers')
    
    args = parser.parse_args()

    # Find h5ad files
    data_dir = Path(args.data_dir)
    h5ad_paths = sorted([str(p) for p in data_dir.rglob("*.h5ad")])
    print(f"Found {len(h5ad_paths)} h5ad files in {data_dir}")

    # Default output path: save in data_dir
    output_path = args.output_path if args.output_path else str(data_dir / 'patch_index.pt')

    if len(h5ad_paths) > 0:
        compute_patch_index(
            h5ad_paths=h5ad_paths,
            output_path=output_path,
            n_fps_runs=args.n_fps_runs,
            n_centers=args.n_centers,
            n_spots=512,
            target_cells_per_subslide=5000,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
