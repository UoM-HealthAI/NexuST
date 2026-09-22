"""NexuST Gene Recovery Finetune (drop-input, probe-aligned).

The HVG60 gene tokens are removed from the encoder input entirely (encoder never
sees them). A single ``Linear(d_model -> n_hvg)`` head predicts their raw
expression from each cell's CLS embedding. Train and validation both target the
same fixed HVG60 set as ``probe/tasks/gene_recovery.py``. Fine-tuning updates
the encoder on sampled spatial patches; probe uses precomputed embeddings.

See the README for the downstream data requirements.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent.parent

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
from tqdm import tqdm

from data.collator import DataCollator
from data.tokenizer import Tokenizer, get_tokenizer
from finetune.utils.cli import add_shared_args, build_ckpt_dir
from finetune.utils.dataset import (
    TrainFinetuneDataset, ValFinetuneDataset, sample_or_pad_genes,
)
from finetune.utils.tools import load_encoder
from finetune.utils.training import TrainingRunner, get_global_rank
from evaluation.io import append_csv_row, needs_header
from evaluation.metrics import ColumnWiseMAE, ColumnWisePearsonCorrCoef


# ────────────────────────────────────────────────────────────────────────────
# Slide container that filters HVG cols and tracks them for label extraction
# ────────────────────────────────────────────────────────────────────────────

class ImputationSlideContainer:
    """Like ``FinetuneSlideContainer`` but with HVG removed from ``gene_ids``
    and the HVG column indices exposed for label extraction.

    Reads adata.var_names once at construction; resolves HVG / non-HVG cols.
    Raises if any HVG symbol is missing from the slide's var panel (would
    mis-align the label tensor) or from the tokenizer vocab.
    """

    def __init__(self, h5ad_path: str, tokenizer: Tokenizer, hvg_symbols: List[str]):
        self.h5ad_path = h5ad_path
        adata = sc.read_h5ad(h5ad_path, backed="r")

        if "spatial" in adata.obsm:
            self.coords = adata.obsm["spatial"][:].astype("float32")
        elif "X_spatial" in adata.obsm:
            self.coords = adata.obsm["X_spatial"][:].astype("float32")
        else:
            raise KeyError(f"No spatial coords in {h5ad_path}")

        var_names = list(adata.var_names)
        var_to_col = {g: i for i, g in enumerate(var_names)}

        missing_in_panel = [g for g in hvg_symbols if g not in var_to_col]
        if missing_in_panel:
            raise ValueError(
                f"{Path(h5ad_path).name}: HVG missing from var_names: "
                f"{missing_in_panel}"
            )
        missing_in_vocab = [g for g in hvg_symbols if g not in tokenizer.gene_vocab]
        if missing_in_vocab:
            raise ValueError(f"HVG not in tokenizer vocab: {missing_in_vocab}")

        # Column indices into the FULL adata.X
        hvg_set = set(hvg_symbols)
        kept_var_names = [g for g in var_names if g not in hvg_set]
        self.kept_cols = np.array([var_to_col[g] for g in kept_var_names], dtype=np.int64)
        # HVG cols in the FIXED hvg_symbols order — the label column order is
        # the same for every cell and every dataset.
        self.hvg_label_cols = np.array([var_to_col[g] for g in hvg_symbols], dtype=np.int64)

        # Encoder vocab tokens for the kept (non-HVG) genes only.
        self.gene_ids = [tokenizer.encode_gene(g) for g in kept_var_names]

        organ_str = adata.uns.get("organ", "unknown")
        self.organ_id = tokenizer.encode_metadata("organ", organ_str)
        platform_str = adata.uns.get("platform", "unknown")
        self.platform_id = tokenizer.encode_metadata("platform", platform_str)

        self._adata_backed = None

    def open_backed(self):
        if self._adata_backed is None:
            self._adata_backed = sc.read_h5ad(self.h5ad_path, backed="r")
        return self._adata_backed

    @property
    def n_spots(self) -> int:
        return self.coords.shape[0]

    def load_arrays(self, spot_idx: np.ndarray):
        """Return (kept_values, hvg_labels) both float32; HVG never enters
        kept_values."""
        adata_backed = self.open_backed()
        X = adata_backed.X[spot_idx, :]
        arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        arr = arr.astype(np.float32)
        return arr[:, self.kept_cols], arr[:, self.hvg_label_cols]


# ────────────────────────────────────────────────────────────────────────────
# Datasets
# ────────────────────────────────────────────────────────────────────────────

def _load_hvg_symbols(hvg_path: Path) -> List[str]:
    return np.load(hvg_path, allow_pickle=True).tolist()


class TrainImputationDataset(TrainFinetuneDataset):
    """Train dataset: drop HVG token ids from encoder input + emit HVG label."""

    def __init__(self, train_dir: str, tokenizer: Tokenizer,
                 hvg_symbols: List[str], n_spots: int = 512,
                 max_gene_len: int = 300, seed: int = 42, device: str = "cpu"):
        self._hvg_symbols = list(hvg_symbols)
        self._tokenizer = tokenizer  # _load_slides needs it
        super().__init__(train_dir=train_dir, tokenizer=tokenizer,
                         n_spots=n_spots, max_gene_len=max_gene_len,
                         seed=seed, device=device)

    def _load_slides(self):
        self.h5ad_paths = sorted(self.train_dir.glob("*.h5ad"))
        if not self.h5ad_paths:
            raise ValueError(f"No h5ad files in {self.train_dir}")
        self.slides: List[ImputationSlideContainer] = []
        total = 0
        for path in tqdm(self.h5ad_paths, desc="Loading train FOVs (imp)"):
            container = ImputationSlideContainer(
                str(path), self._tokenizer, self._hvg_symbols)
            self.slides.append(container)
            total += container.n_spots
        print(f"Loaded {len(self.slides)} train FOVs, {total:,} cells")

    def __getitem__(self, idx):
        fov_idx, n_centers, center_idx = self.index_map[idx]
        slide = self.slides[fov_idx]
        n_cells = slide.n_spots
        sample_seed = self.seed + self.epoch * 100003 + idx

        if n_cells >= self.n_spots:
            patches = self.patch_sampler.fps_and_knn(
                slide.coords, n_centers=n_centers, seed=sample_seed)
            spot_idx = patches[center_idx]
            cell_padding_mask = torch.zeros(self.n_spots, dtype=torch.bool)
        else:
            spot_idx = np.arange(n_cells)
            cell_padding_mask = torch.ones(self.n_spots, dtype=torch.bool)
            cell_padding_mask[:n_cells] = False

        spot_idx = np.asarray(spot_idx, dtype=np.int64)
        coords = torch.from_numpy(slide.coords[spot_idx]).float()

        # HVG-aware load: arr_kept = non-HVG cols, arr_hvg = HVG cols
        arr_kept, arr_hvg = slide.load_arrays(spot_idx)
        gene_values = torch.from_numpy(arr_kept)        # [k, n_kept]
        hvg_labels = torch.from_numpy(arr_hvg)          # [k, n_hvg]

        gene_ids = torch.tensor(slide.gene_ids, dtype=torch.long)
        gene_ids = gene_ids.unsqueeze(0).expand(len(spot_idx), -1)
        gene_ids, gene_values = self._sample_or_pad_genes(gene_ids, gene_values)

        k = len(spot_idx)
        n_hvg = hvg_labels.shape[1]
        coords_padded = torch.zeros(self.n_spots, 2, dtype=coords.dtype)
        gene_values_padded = torch.zeros(self.n_spots, self.max_gene_len, dtype=gene_values.dtype)
        gene_ids_padded = torch.zeros(self.n_spots, self.max_gene_len, dtype=gene_ids.dtype)
        labels_padded = torch.zeros(self.n_spots, n_hvg, dtype=hvg_labels.dtype)
        coords_padded[:k] = coords
        gene_values_padded[:k] = gene_values
        gene_ids_padded[:k] = gene_ids
        labels_padded[:k] = hvg_labels

        return {
            "coords": coords_padded,
            "gene_ids": gene_ids_padded,
            "gene_values": gene_values_padded,
            "hvg_labels": labels_padded,                # [n_spots, n_hvg]
            "cell_padding_mask": cell_padding_mask,     # [n_spots] True=pad
            "organ_id": slide.organ_id,
            "batch_label": torch.full((self.n_spots,), slide.platform_id, dtype=torch.long),
        }


class ValImputationDataset(ValFinetuneDataset):
    """Val dataset: same drop-input + label emission, Hilbert chunking inherited."""

    def __init__(self, val_dir: str, tokenizer: Tokenizer,
                 hvg_symbols: List[str], max_gene_len: int = 300,
                 max_cells: int = 512, hilbert_grid_size: int = 512):
        self._hvg_symbols = list(hvg_symbols)
        self._tokenizer = tokenizer
        super().__init__(val_dir=val_dir, tokenizer=tokenizer,
                         max_gene_len=max_gene_len, max_cells=max_cells,
                         hilbert_grid_size=hilbert_grid_size)

    def _load_slides(self):
        self.h5ad_paths = sorted(self.val_dir.glob("*.h5ad"))
        self.slides: List[ImputationSlideContainer] = []
        for path in tqdm(self.h5ad_paths, desc="Loading val FOVs (imp)"):
            self.slides.append(ImputationSlideContainer(
                str(path), self._tokenizer, self._hvg_symbols))

    def __getitem__(self, idx):
        slide_idx, cell_indices = self.index_map[idx]
        slide = self.slides[slide_idx]
        spot_idx = np.asarray(cell_indices, dtype=np.int64)

        coords = torch.from_numpy(slide.coords[spot_idx]).float()
        arr_kept, arr_hvg = slide.load_arrays(spot_idx)
        gene_values = torch.from_numpy(arr_kept)
        hvg_labels = torch.from_numpy(arr_hvg)

        gene_ids = torch.tensor(slide.gene_ids, dtype=torch.long)
        gene_ids = gene_ids.unsqueeze(0).expand(len(spot_idx), -1)
        # Pad/truncate genes to max_gene_len; no FPS sampling in val
        gene_ids, gene_values = sample_or_pad_genes(
            gene_ids, gene_values, self.max_gene_len,
            seed=0, use_expression_weights=False)

        # Val: no cell padding (each chunk is real cells)
        return {
            "coords": coords,
            "gene_ids": gene_ids,
            "gene_values": gene_values,
            "hvg_labels": hvg_labels,                   # [k, n_hvg]
            "organ_id": slide.organ_id,
            "batch_label": torch.full((len(spot_idx),), slide.platform_id, dtype=torch.long),
        }


# ────────────────────────────────────────────────────────────────────────────
# Collator: pop hvg_labels + cell_padding_mask before base DataCollator drops them
# ────────────────────────────────────────────────────────────────────────────

class ImputationCollator(DataCollator):
    def __call__(self, batch):
        hvg_labels = [s.pop("hvg_labels") for s in batch]
        cell_padding_masks = [s.pop("cell_padding_mask", None) for s in batch]
        collated = super().__call__(batch)
        collated["hvg_labels"] = torch.stack(hvg_labels, dim=0)  # [B, N, n_hvg]
        if cell_padding_masks[0] is not None:
            collated["cell_padding_mask"] = torch.stack(cell_padding_masks, dim=0)  # [B, N]
        return collated


# ────────────────────────────────────────────────────────────────────────────
# Lightning module
# ────────────────────────────────────────────────────────────────────────────

class NexuSTImputation(pl.LightningModule):
    def __init__(self, pretrain_ckpt: str, n_hvg: int,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01, min_lr: float = 1e-6):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = load_encoder(pretrain_ckpt)
        d_model = self.encoder.d_model
        self.head = nn.Linear(d_model, n_hvg)

        self.lr = lr
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.n_hvg = n_hvg


        # Probe-aligned per-gene metrics
        self.val_mae = ColumnWiseMAE(n_hvg, sync_on_compute=False)
        self.val_pcc = ColumnWisePearsonCorrCoef(n_hvg, sync_on_compute=False)
        # Scalar MSE accumulated manually to skip padded cells without inflating
        # torchmetrics state with masked zeros.
        self._val_mse_sum = 0.0
        self._val_mse_n = 0

        self.best_pcc = float("-inf")
        self.best_metrics = None
        self.last_per_gene = None

    def forward(self, batch):
        out = self.encoder(
            batch["gene_ids"], batch["gene_values"], batch["organ_ids"],
            batch["coords"],
            cell_padding_mask=batch.get("cell_padding_mask"),
            mask=None,
        )
        cls = out["cls_token"]  # [B, N, D]
        return self.head(cls)   # [B, N, n_hvg]

    def _valid_cells_view(self, pred, target, cell_padding_mask):
        """Flatten (B, N, n_hvg) → (n_real, n_hvg), excluding padded cells."""
        B, N = pred.shape[:2]
        if cell_padding_mask is None:
            return pred.view(B * N, -1), target.view(B * N, -1)
        valid = (~cell_padding_mask).view(B * N)
        return pred.view(B * N, -1)[valid], target.view(B * N, -1)[valid]

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch["hvg_labels"]
        pred_v, target_v = self._valid_cells_view(
            pred, target, batch.get("cell_padding_mask"))
        loss = F.mse_loss(pred_v, target_v)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True,
                 batch_size=pred_v.size(0))
        return loss

    def on_validation_epoch_start(self):
        self.val_mae.reset()
        self.val_pcc.reset()
        self._val_mse_sum = 0.0
        self._val_mse_n = 0

    def validation_step(self, batch, batch_idx):
        pred = self(batch).float()
        target = batch["hvg_labels"].float()
        pred_v, target_v = self._valid_cells_view(
            pred, target, batch.get("cell_padding_mask"))
        self.val_mae.update(pred_v, target_v)
        self.val_pcc.update(pred_v, target_v)
        self._val_mse_sum += float(((pred_v - target_v) ** 2).sum())
        self._val_mse_n += pred_v.numel()

    def on_validation_epoch_end(self):
        scalar_mse = self._val_mse_sum / max(self._val_mse_n, 1)
        scalar_mae = float(self.val_mae.compute())
        scalar_pcc = float(self.val_pcc.compute())

        self.log("val_mse", scalar_mse, prog_bar=True, sync_dist=True)
        self.log("val_mae", scalar_mae, prog_bar=True, sync_dist=True)
        self.log("val_pcc", scalar_pcc, prog_bar=True, sync_dist=True)

        if scalar_pcc > self.best_pcc:
            self.best_pcc = scalar_pcc
            self.best_metrics = {"mse": scalar_mse, "mae": scalar_mae,
                                 "pcc": scalar_pcc}
            self.last_per_gene = {
                "mae": self.val_mae.compute_per_column().cpu().numpy(),
                "pcc": self.val_pcc.compute_per_column().cpu().numpy(),
            }

    def configure_optimizers(self):
        # Single LR for all trainable params — same convention as the other
        # NexuST finetune tasks (classification / niche / density).
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        import math
        total_steps = self.trainer.estimated_stepping_batches
        min_lr = self.min_lr

        def _lr_lambda(step):
            progress = step / float(max(1, total_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_factor = min_lr / max(self.lr, 1e-12)
            return cosine * (1 - min_factor) + min_factor

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ────────────────────────────────────────────────────────────────────────────
# CSV writer (probe-aligned schema)
# ────────────────────────────────────────────────────────────────────────────

def write_results(output_dir: Path, dataset_name: str, mode: str, seed: int,
                  lr: float, metrics: dict, per_gene_mae: np.ndarray,
                  per_gene_pcc: np.ndarray, hvg_symbols: List[str]):
    """Append one row per CSV. Always includes ``lr`` for easy LR sweep
    aggregation (the column is harmless when lr is fixed)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_s = f"{lr:g}"

    scalar_csv = output_dir / f"{dataset_name}.csv"
    append_csv_row(
        scalar_csv,
        ["dataset", "mode", "seed", "lr", "mse", "mae", "pcc"],
        [dataset_name, mode, seed, lr_s,
         f"{metrics['mse']:.6f}", f"{metrics['mae']:.6f}", f"{metrics['pcc']:.4f}"],
        write_header=needs_header(scalar_csv),
    )

    cols = ["dataset", "mode", "seed", "lr"] + list(hvg_symbols)
    mae_csv = output_dir / f"{dataset_name}_gene_mae.csv"
    append_csv_row(
        mae_csv, cols,
        [dataset_name, mode, seed, lr_s] +
        [f"{v:.6f}" if not np.isnan(v) else "nan" for v in per_gene_mae],
        write_header=needs_header(mae_csv),
    )

    pcc_csv = output_dir / f"{dataset_name}_gene_pcc.csv"
    append_csv_row(
        pcc_csv, cols,
        [dataset_name, mode, seed, lr_s] +
        [f"{v:.4f}" if not np.isnan(v) else "nan" for v in per_gene_pcc],
        write_header=needs_header(pcc_csv),
    )
    return scalar_csv


# ────────────────────────────────────────────────────────────────────────────
# Best-epoch metric snapshot callback (mirrors per-baseline pattern)
# ────────────────────────────────────────────────────────────────────────────

class _BestEpochCb(pl.Callback):
    """Pull module.best_metrics / module.last_per_gene from module state
    (already maintained inside on_validation_epoch_end). No-op callback; kept
    for symmetry with the other tasks if we want save-to-disk later."""


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="NexuST Gene Recovery (drop-input)")
    add_shared_args(parser)
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--hvg_symbol_path", type=str, default=None,
                        help="Defaults to <data_path>/hvg60_symbol.npy")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Shared run id; defaults to current timestamp. "
                             "Set to ${SLURM_ARRAY_JOB_ID} so all array tasks "
                             "share one ckpt dir.")
    parser.set_defaults(
        output_dir=str(Path.cwd() / "output/results/gene_recovery/finetune/nexust_linear"),
        max_gene_len=300,
        lr=1e-4,  # single LR for both encoder and head
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name or Path(args.data_path).stem
    mode = "finetune"
    seed = args.seed
    pl.seed_everything(seed, workers=True)

    hvg_path = Path(args.hvg_symbol_path) if args.hvg_symbol_path \
        else Path(args.data_path) / "hvg60_symbol.npy"
    if not hvg_path.exists():
        raise FileNotFoundError(f"HVG symbol file not found: {hvg_path}")
    hvg_symbols = _load_hvg_symbols(hvg_path)
    print(f"[gene_recovery] HVG: {len(hvg_symbols)} symbols")

    tokenizer = get_tokenizer()

    train_dataset = TrainImputationDataset(
        train_dir=str(Path(args.data_path) / "train"),
        tokenizer=tokenizer, hvg_symbols=hvg_symbols,
        max_gene_len=args.max_gene_len, seed=seed,
    )
    val_dataset = ValImputationDataset(
        val_dir=str(Path(args.data_path) / "val"),
        tokenizer=tokenizer, hvg_symbols=hvg_symbols,
        max_gene_len=args.max_gene_len, max_cells=512,
    )
    collator = ImputationCollator(max_gene_len=args.max_gene_len)

    model = NexuSTImputation(
        pretrain_ckpt=args.pretrain_ckpt, n_hvg=len(hvg_symbols),
        lr=args.lr, min_lr=args.min_lr,
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = build_ckpt_dir(args.ckpt_root, "gene_recovery", run_id,
                              dataset_name, mode, f"seed{seed}")

    runner = TrainingRunner(
        seed=seed, max_epochs=args.max_epochs, devices=args.devices,
        batch_size=args.batch_size, accumulate_grad_batches=args.accumulate_grad_batches,
        num_workers=args.num_workers,
        monitor="val_pcc", monitor_mode="max", early_stop_patience=10,
        gradient_clip_val=1.0,
        ckpt_dir=ckpt_dir,
        ckpt_filename=f"seed{seed}_" + "epoch{epoch:02d}_pcc{val_pcc:.4f}",
        wandb_project=args.project, logger=args.logger,
        accelerator=args.accelerator, precision=args.precision,
        wandb_group=args.group or f"{dataset_name}-imp-{mode}",
        wandb_name=f"nexust-linear_{dataset_name}_{mode}_seed{seed}_lr{args.lr:g}",
        wandb_config={
            "task": "gene_recovery", "model": "nexust",
            "head": "linear", "pretrained_head": False,
            "phase": getattr(args, "phase", "final"),
            "dataset": dataset_name, "mode": mode, "seed": seed,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "accumulate_grad_batches": args.accumulate_grad_batches,
        },
        wandb_tags=[
            "gene_recovery",
            f"phase:{getattr(args, 'phase', 'final')}",
            "head:linear",
            "model:nexust",
            dataset_name,
        ],
    )
    runner.fit(model, train_dataset, val_dataset, collator, callbacks=[_BestEpochCb()])

    if get_global_rank() == 0:
        if model.best_metrics is None:
            raise RuntimeError("No validation epoch completed; nothing to write.")
        out_dir = Path(args.output_dir)
        scalar_csv = write_results(
            out_dir, dataset_name, mode, seed, args.lr,
            model.best_metrics, model.last_per_gene["mae"],
            model.last_per_gene["pcc"], hvg_symbols,
        )
        m = model.best_metrics
        print(f"[seed {seed}] mse={m['mse']:.6f}  mae={m['mae']:.6f}  pcc={m['pcc']:.4f}")
        print(f"  scalar CSV    : {scalar_csv}")
        print(f"  per-gene MAE  : {out_dir / f'{dataset_name}_gene_mae.csv'}")
        print(f"  per-gene PCC  : {out_dir / f'{dataset_name}_gene_pcc.csv'}")


if __name__ == "__main__":
    main()
