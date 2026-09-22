import argparse
import os
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import List

import anndata as ad
import numpy as np
import scanpy as sc
import torch
import torch.multiprocessing as mp

from finetune.utils.tools import get_model_config, load_encoder


def _worker_fn(
    rank: int,
    world_size: int,
    gpu_ids: List[int],
    h5ad_path: str,
    ckpt_path: str,
    max_cells: int,
    max_gene_len: int,
    result_queue: mp.Queue,
):
    """Worker function for multi-GPU inference."""
    from finetune.utils.dataset import ValFinetuneDataset
    from data.tokenizer import get_tokenizer

    # Set GPU
    gpu_id = gpu_ids[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    encoder = load_encoder(ckpt_path, device=device).to(device).eval()

    # Load dataset - create temp symlink dir
    tokenizer = get_tokenizer()
    tmp_dir = tempfile.mkdtemp()
    val_dir = Path(tmp_dir) / "val"
    val_dir.mkdir()
    h5ad_resolved = Path(h5ad_path).resolve()
    link_path = val_dir / h5ad_resolved.name

    try:
        os.symlink(h5ad_resolved, link_path)
    except OSError:
        shutil.copy2(h5ad_resolved, link_path)

    dataset = ValFinetuneDataset(
        val_dir=str(val_dir),
        tokenizer=tokenizer,
        max_gene_len=max_gene_len,
        max_cells=max_cells,
    )

    # Run inference
    cell_idx_all = []
    emb_all = []
    gene_ids_all = []

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )

    with torch.no_grad(), autocast_ctx:
        for i in range(rank, len(dataset), world_size):
            batch = dataset[i]
            out = encoder(
                batch["gene_ids"].unsqueeze(0).to(device),
                batch["gene_values"].unsqueeze(0).to(device),
                torch.tensor([batch["organ_id"]], device=device),
                batch["coords"].unsqueeze(0).to(device),
                mask=None,
            )
            cell_idx_all.append(batch["cell_indices"])
            emb_all.append(out["cls_token"].squeeze(0).cpu().numpy())
            gene_ids_all.append(batch["gene_ids"].numpy())

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Put results in queue
    result_queue.put({
        "rank": rank,
        "cell_idx": cell_idx_all,
        "emb": emb_all,
        "gene_ids": gene_ids_all,
    })


def run_inference(
    adata: ad.AnnData,
    ckpt: str,
    gpus: str = "0",
    max_cells: int = 1024,
    max_gene_len: int = 300,
    obsm_key: str = "X_nexust",
):
    """
    Run multi-GPU inference and return adata with embeddings.

    Args:
        adata: AnnData object
        ckpt: Path to model checkpoint
        gpus: Comma-separated GPU ids (e.g. "0,1,2")
        max_cells: Max cells per chunk
        max_gene_len: Max genes per cell
        obsm_key: Key to store embeddings in adata.obsm

    Returns:
        adata: AnnData with embeddings in obsm[obsm_key]
    """
    # Write adata to temp file for multi-GPU workers
    tmp_h5ad = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
    adata.write_h5ad(tmp_h5ad.name)
    h5ad_path = Path(tmp_h5ad.name)

    gpu_ids = [int(x) for x in gpus.split(",") if x.strip()]
    world_size = len(gpu_ids)

    # Use spawn context to avoid CUDA issues
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    # Spawn workers
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_fn,
            args=(
                rank, world_size, gpu_ids,
                str(h5ad_path), str(ckpt),
                max_cells, max_gene_len,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Collect results
    results = []
    for _ in range(world_size):
        results.append(result_queue.get())

    # Wait for processes
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker failed with exit code {p.exitcode}")

    # Stitch results
    D = int(get_model_config(ckpt).d_model)
    X = np.zeros((adata.n_obs, D), dtype=np.float32)
    gene_ids_stitched = None

    for res in results:
        for cell_idx, emb in zip(res["cell_idx"], res["emb"]):
            X[cell_idx.astype(np.int64)] = emb

        if res["gene_ids"]:
            if gene_ids_stitched is None:
                G = res["gene_ids"][0].shape[1]
                gene_ids_stitched = np.zeros((adata.n_obs, G), dtype=np.int64)
            for cell_idx, gene_ids in zip(res["cell_idx"], res["gene_ids"]):
                gene_ids_stitched[cell_idx.astype(np.int64)] = gene_ids

    adata.obsm[obsm_key] = X
    if gene_ids_stitched is not None:
        adata.obsm["gene_ids"] = gene_ids_stitched

    # Cleanup temp file
    os.unlink(tmp_h5ad.name)

    print(f"Done: computed {obsm_key} with shape {X.shape}")
    return adata


def embed_adata(
    adata: ad.AnnData,
    encoder,
    tokenizer,
    *,
    device: str,
    d_model: int,
    max_cells: int = 1024,
    max_gene_len: int = 300,
    obsm_key: str = "X_nexust",
):
    """Single-GPU, in-process embedding that reuses a preloaded encoder + tokenizer.

    Same Hilbert chunking and forward pass as the multi-GPU run_inference()
    workers, but the caller loads the model ONCE and passes it in. For repeated
    per-FOV embedding (e.g. the probe harness, which calls embed() once per
    h5ad) this avoids reloading the checkpoint and spawning a fresh process for
    every FOV — the dominant cost there, not the actual compute.
    """
    from finetune.utils.dataset import ValFinetuneDataset

    tmp_dir = tempfile.mkdtemp()
    try:
        val_dir = Path(tmp_dir) / "val"
        val_dir.mkdir()
        adata.write_h5ad(val_dir / "fov.h5ad")

        dataset = ValFinetuneDataset(
            val_dir=str(val_dir),
            tokenizer=tokenizer,
            max_gene_len=max_gene_len,
            max_cells=max_cells,
        )

        X = np.zeros((adata.n_obs, int(d_model)), dtype=np.float32)
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            for i in range(len(dataset)):
                batch = dataset[i]
                out = encoder(
                    batch["gene_ids"].unsqueeze(0).to(device),
                    batch["gene_values"].unsqueeze(0).to(device),
                    torch.tensor([batch["organ_id"]], device=device),
                    batch["coords"].unsqueeze(0).to(device),
                    mask=None,
                )
                emb = out["cls_token"].squeeze(0).cpu().numpy()
                X[batch["cell_indices"].astype(np.int64)] = emb
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    adata.obsm[obsm_key] = X
    return adata


def main():
    parser = argparse.ArgumentParser(description="Run multi-GPU inference")
    parser.add_argument("--h5ad", required=True, help="Input h5ad file")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint")
    parser.add_argument("--out", default=None, help="Output h5ad path")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids for multi-GPU inference")
    parser.add_argument("--device", default=None, help="Single-process device, e.g. cpu or cuda:0")
    parser.add_argument("--max_cells", type=int, default=1024, help="Max cells per chunk")
    parser.add_argument("--max_gene_len", type=int, default=300, help="Max genes per cell")
    parser.add_argument("--obsm_key", default="X_nexust", help="Key for embeddings")
    args = parser.parse_args()

    adata = sc.read_h5ad(args.h5ad)
    if args.device is not None:
        from data.tokenizer import get_tokenizer
        encoder = load_encoder(args.ckpt, device=args.device).to(args.device).eval()
        adata = embed_adata(
            adata, encoder, get_tokenizer(), device=args.device,
            d_model=encoder.d_model, max_cells=args.max_cells,
            max_gene_len=args.max_gene_len, obsm_key=args.obsm_key,
        )
    else:
        adata = run_inference(
            adata=adata,
            ckpt=args.ckpt,
            gpus=args.gpus,
            max_cells=args.max_cells,
            max_gene_len=args.max_gene_len,
            obsm_key=args.obsm_key,
        )

    out_path = args.out or args.h5ad.replace(".h5ad", "_emb.h5ad")
    adata.write_h5ad(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
