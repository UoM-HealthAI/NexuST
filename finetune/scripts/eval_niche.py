"""Re-evaluate a single NexuST niche prediction checkpoint on the full val set.

Used to regenerate ``best_metrics.json`` for niche checkpoints. Scalar
``mse``/``mae``/``pcc`` were already correct under DDP (torchmetrics +
Lightning sync_dist), but the JSON only contained these three values; this
script also computes and persists ``center_mae_dict`` from
``evaluation.niche.center_grouped_mae``, which is the biologically
meaningful per-class breakdown (MAE grouped by center cell type, not by
composition column).

Inference is single-GPU by design: no DistributedSampler, no per-rank state,
no race on JSON write. Intended to be launched as a SLURM array job, one
task per checkpoint (4 datasets x 4 radii x 3 seeds = 48 tasks).

Usage:
    python finetune/scripts/eval_niche.py \
        --ckpt_path checkpoints/niche/model.ckpt \
        --data_path datasets/downstream/my_dataset \
        --radius_idx <R>
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.tokenizer import get_tokenizer
from finetune.tasks.niche_prediction import (
    NexuSTNichePrediction,
    NicheCollator,
    ValNicheDataset,
)
from finetune.utils.tools import ColumnWisePearsonCorrCoef
from evaluation.niche import center_grouped_mae


def build_val_loader(data_path: Path, niche_key: str, max_gene_len: int, num_workers: int):
    tokenizer = get_tokenizer()
    val_dataset = ValNicheDataset(
        val_dir=str(data_path / "val"),
        tokenizer=tokenizer,
        niche_key=niche_key,
        max_gene_len=max_gene_len,
    )
    collator = NicheCollator(max_gene_len=max_gene_len)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    return val_loader, val_dataset.n_cell_types


@torch.no_grad()
def run_forward(model: NexuSTNichePrediction, val_loader: DataLoader, device: str):
    """Iterate val_loader once and accumulate flat preds/targets/center_types.

    Mirrors NexuSTNichePrediction.validation_step's _flatten_and_normalize logic
    (drop padding, normalize target to a probability distribution).
    """
    all_preds, all_targets, all_center_types = [], [], []
    autocast_device = device.split(":")[0]

    for batch in val_loader:
        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            pred = model(
                gene_ids=batch["gene_ids"].to(device),
                gene_values=batch["gene_values"].to(device),
                coords=batch["coords"].to(device),
                organ_ids=batch["organ_ids"].to(device),
                cell_padding_mask=batch.get("cell_padding_mask"),
            )

        target = batch["niche_labels"].to(device)
        cell_padding_mask = batch.get("cell_padding_mask")
        center_type = batch.get("center_type")

        pred_flat = pred.view(-1, pred.size(-1)).float()
        target_flat = target.view(-1, target.size(-1)).float()

        if cell_padding_mask is not None:
            valid_mask = ~cell_padding_mask.view(-1)
            pred_flat = pred_flat[valid_mask.to(device)]
            target_flat = target_flat[valid_mask.to(device)]
            if center_type is not None:
                center_type = center_type[valid_mask.cpu().numpy()]

        # Normalize target to a proper probability distribution (matches training)
        s = target_flat + 1e-8
        target_flat = s / s.sum(dim=-1, keepdim=True)

        all_preds.append(pred_flat.cpu())
        all_targets.append(target_flat.cpu())
        if center_type is not None:
            all_center_types.append(center_type)

    return all_preds, all_targets, all_center_types


def compute_metrics(all_preds, all_targets, all_center_types, n_cell_types: int):
    """Compute metrics in the same way NexuSTNichePrediction's val path does.

    - mse / mae are F.mse_loss / F.l1_loss over the concatenated full-val tensor,
      which is mathematically equivalent to Lightning's epoch-aggregated
      val_mse / val_mae (weighted by per-batch valid cell count).
    - pcc is the same ColumnWisePearsonCorrCoef.compute() used during training.
    - center_mae_dict groups MAE by the *center* cell's type (biologically
      meaningful per-class breakdown).
    """
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse = float(F.mse_loss(preds, targets))
    mae = float(F.l1_loss(preds, targets))

    pcc_metric = ColumnWisePearsonCorrCoef(n_features=n_cell_types)
    pcc_metric.update(preds, targets)
    pcc = float(pcc_metric.compute())

    out = {
        "mse": mse,
        "mae": mae,
        "pcc": pcc,
    }

    if all_center_types:
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        center_types_np = np.concatenate(all_center_types)
        center_mae_dict = center_grouped_mae(preds_np, targets_np, center_types_np)
        # Convert numpy arrays inside the dict to lists for JSON serialization
        out["center_mae_dict"] = {
            str(ct): {
                "mae": float(info["mae"]),
                "mae_per_target": info["mae_per_target"].tolist(),
            }
            for ct, info in center_mae_dict.items()
        }

    return out


def parse_epoch(ckpt_name: str) -> int:
    m = re.search(r"epoch=(\d+)", ckpt_name)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Exact .ckpt file to evaluate")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Dataset dir containing train/ and val/")
    parser.add_argument("--radius_idx", type=int, required=True,
                        help="Niche radius index (selects X_niche_<radius_idx> obsm key)")
    parser.add_argument("--niche_key", type=str, default="X_niche",
                        help="Niche obsm key prefix (default: X_niche)")
    parser.add_argument("--max_gene_len", type=int, default=300,
                        help="Must match training (default matches niche_prediction.py)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_json", type=str, default=None,
                        help="Override output JSON path (default: <ckpt_dir>/best_metrics.json)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = Path(args.ckpt_path)
    data_path = Path(args.data_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    effective_niche_key = f"{args.niche_key}_{args.radius_idx}"

    print(f"=== {data_path.name} | r{args.radius_idx} | {ckpt_path.name} ===")
    print(f"  niche_key = {effective_niche_key}")

    val_loader, n_cell_types = build_val_loader(
        data_path, effective_niche_key, args.max_gene_len, args.num_workers,
    )
    print(f"  n_cell_types = {n_cell_types}")

    print(f"  load {ckpt_path.name}")
    model = NexuSTNichePrediction.load_from_checkpoint(
        str(ckpt_path), map_location=device, weights_only=False,
    )
    model.eval().to(device)

    all_preds, all_targets, all_center_types = run_forward(model, val_loader, device)
    metrics = compute_metrics(all_preds, all_targets, all_center_types, n_cell_types)

    out = {
        "epoch": parse_epoch(ckpt_path.name),
        "radius_idx": args.radius_idx,
        **metrics,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = ckpt_path.parent / "best_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"  mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} pcc={metrics['pcc']:.4f}")
    if "center_mae_dict" in metrics:
        print(f"  center_mae center types = {len(metrics['center_mae_dict'])}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
