import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.neighbors import radius_neighbors_graph
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score

PROJECT_ROOT = Path(__file__).parent.parent.parent

from data.tokenizer import get_tokenizer, Tokenizer
from data.collator import DataCollator
from finetune.utils.dataset import TrainFinetuneDataset, ValFinetuneDataset
from finetune.utils.tools import load_encoder
from finetune.utils.callbacks import BaseMetricsRecorder
from finetune.utils.cli import add_shared_args, build_ckpt_dir, append_csv, append_per_group_csv
from finetune.utils.training import TrainingRunner, get_global_rank
from configs.datasets import DATASET_CONFIG


def compute_density(coords: np.ndarray, radius: float) -> np.ndarray:
    conn = radius_neighbors_graph(
        coords, radius=radius, mode="connectivity", include_self=False
    )
    return np.asarray(conn.sum(axis=1), dtype=np.float32).ravel()


class TrainDensityDataset(TrainFinetuneDataset):
    def __init__(
        self,
        train_dir: str,
        tokenizer: Tokenizer,
        radius: float,
        n_spots: int = 512,
        max_gene_len: int = 300,
    ):
        super().__init__(train_dir, tokenizer, n_spots=n_spots, max_gene_len=max_gene_len)
        self.radius = radius
        self.slide_density = [
            compute_density(slide.coords, radius=radius) for slide in self.slides
        ]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        fov_idx, _, _ = self.index_map[idx]
        spot_idx = batch.pop('spot_idx')  # [n_spots], -1 for padding

        density = self.slide_density[fov_idx]
        density_labels = np.zeros(self.n_spots, dtype=np.float32)
        valid = spot_idx >= 0
        valid_idx = spot_idx[valid].cpu().numpy()
        density_labels[valid.cpu().numpy()] = density[valid_idx]

        batch['density_labels'] = torch.from_numpy(density_labels)
        return batch


class ValDensityDataset(ValFinetuneDataset):
    def __init__(
        self,
        val_dir: str,
        tokenizer: Tokenizer,
        radius: float,
        label_col: str | None = None,
        max_gene_len: int = 300,
        max_cells: int = 512,
    ):
        super().__init__(val_dir, tokenizer, max_gene_len=max_gene_len, max_cells=max_cells)
        self.radius = radius
        self.label_col = label_col
        self.slide_density = [
            compute_density(slide.coords, radius=radius) for slide in self.slides
        ]
        if label_col is not None:
            self.slide_celltypes = [
                np.asarray(slide.open_backed().obs[label_col].values).astype(str)
                for slide in self.slides
            ]
        else:
            self.slide_celltypes = None

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        slide_idx, cell_indices = self.index_map[idx]
        density = self.slide_density[slide_idx][cell_indices]
        batch['density_labels'] = torch.from_numpy(np.asarray(density, dtype=np.float32))
        if self.slide_celltypes is not None:
            batch['celltype'] = self.slide_celltypes[slide_idx][cell_indices]
        return batch


class DensityCollator(DataCollator):
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        density_labels = [sample.pop('density_labels') for sample in batch]
        celltypes = [sample.pop('celltype', None) for sample in batch]
        collated = super().__call__(batch)
        collated['density_labels'] = torch.stack(density_labels, dim=0)
        if celltypes[0] is not None:
            collated['celltype'] = np.concatenate(celltypes, axis=0)
        return collated


class DensityPredictionHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=True)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class NexuSTDensityPrediction(pl.LightningModule):
    def __init__(
        self,
        pretrain_ckpt: str,
        learning_rate: float = 1e-4,
        encoder_lr: float = 1e-5,
        weight_decay: float = 1e-6,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.encoder_lr = encoder_lr
        self.weight_decay = weight_decay
        self.min_lr = min_lr

        self.nexust_encoder = load_encoder(pretrain_ckpt)

        d_model = self.nexust_encoder.d_model
        self.density_head = DensityPredictionHead(d_model)

        self.val_mse_metric = MeanSquaredError()
        self.val_mae_metric = MeanAbsoluteError()
        self.val_r2_metric = R2Score()

        # Collect val predictions for per-celltype metrics (populated each val
        # epoch, consumed by DensityMetricsRecorder in on_validation_end).
        self._val_preds: list = []
        self._val_targets: list = []
        self._val_celltypes: list = []

    def forward(self, gene_ids, gene_values, coords, organ_ids, cell_padding_mask=None):
        out = self.nexust_encoder(
            gene_ids, gene_values, organ_ids, coords,
            cell_padding_mask=cell_padding_mask)
        cls_token = out['cls_token']
        return self.density_head(cls_token)

    def _flatten_valid(self, pred, target, cell_padding_mask):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        if cell_padding_mask is not None:
            valid = ~cell_padding_mask.view(-1)
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]
        return pred_flat, target_flat

    def training_step(self, batch, batch_idx):
        pred = self(
            batch['gene_ids'], batch['gene_values'], batch['coords'],
            batch['organ_ids'], cell_padding_mask=batch.get('cell_padding_mask'))

        pred_flat, target_flat = self._flatten_valid(
            pred, batch['density_labels'], batch.get('cell_padding_mask'))

        loss = F.mse_loss(pred_flat, target_flat)
        mae = F.l1_loss(pred_flat, target_flat)

        self.log('train_mse', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_mse_metric.reset()
        self.val_mae_metric.reset()
        self.val_r2_metric.reset()
        self._val_preds.clear()
        self._val_targets.clear()
        self._val_celltypes.clear()

    def validation_step(self, batch, batch_idx):
        pred = self(
            batch['gene_ids'], batch['gene_values'], batch['coords'],
            batch['organ_ids'], cell_padding_mask=batch.get('cell_padding_mask'))

        pred_flat, target_flat = self._flatten_valid(
            pred, batch['density_labels'], batch.get('cell_padding_mask'))

        self.val_mse_metric.update(pred_flat, target_flat)
        self.val_mae_metric.update(pred_flat, target_flat)
        self.val_r2_metric.update(pred_flat, target_flat)

        # Collect for per-celltype stats. Val has no cell_padding_mask (chunks
        # carry only real cells), so celltype length matches pred_flat length.
        self._val_preds.append(pred_flat.detach().float().cpu().numpy())
        self._val_targets.append(target_flat.detach().float().cpu().numpy())
        celltype = batch.get('celltype')
        if celltype is not None:
            self._val_celltypes.append(celltype)

    def on_validation_epoch_end(self):
        self.log('val_mse', self.val_mse_metric.compute(), prog_bar=True, sync_dist=True)
        self.log('val_mae', self.val_mae_metric.compute(), prog_bar=True, sync_dist=True)
        self.log('val_r2', self.val_r2_metric.compute(), prog_bar=True, sync_dist=True)

    def get_val_predictions(self):
        """Return concatenated (preds, targets, celltypes) from last val epoch.

        celltypes is None if ValDensityDataset was built without label_col.
        """
        if not self._val_preds:
            return np.array([]), np.array([]), None
        preds = np.concatenate(self._val_preds, axis=0)
        targets = np.concatenate(self._val_targets, axis=0)
        celltypes = (np.concatenate(self._val_celltypes, axis=0)
                     if self._val_celltypes else None)
        return preds, targets, celltypes

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.nexust_encoder.parameters(), 'lr': self.encoder_lr},
            {'params': self.density_head.parameters(), 'lr': self.learning_rate},
        ], weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=self.min_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class DensityMetricsRecorder(BaseMetricsRecorder):
    def __init__(self, save_dir=None):
        super().__init__(primary_metric='mse', mode='min', save_dir=save_dir)

    def _extract_metrics(self, trainer, pl_module):
        mse = trainer.callback_metrics.get('val_mse')
        if mse is None:
            return None
        mae = trainer.callback_metrics.get('val_mae')
        r2 = trainer.callback_metrics.get('val_r2')
        metrics = {
            'mse': mse.item(),
            'mae': mae.item() if mae is not None else float('inf'),
            'r2': r2.item() if r2 is not None else float('nan'),
        }
        preds, targets, celltypes = pl_module.get_val_predictions()
        if celltypes is not None and len(preds) > 0:
            metrics['per_celltype'] = _per_celltype_metrics(preds, targets, celltypes)
        return metrics


def _per_celltype_metrics(preds: np.ndarray, targets: np.ndarray,
                          celltypes: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Group by celltype and compute mse/mae/r²/n per group.

    Groups with <2 samples or zero target variance fall back to r2=nan.
    """
    result: Dict[str, Dict[str, float]] = {}
    unique = np.unique(celltypes)
    for c in unique:
        mask = celltypes == c
        p = preds[mask]
        t = targets[mask]
        n = int(mask.sum())
        if n == 0:
            continue
        mse = float(((p - t) ** 2).mean())
        mae = float(np.abs(p - t).mean())
        if n >= 2:
            ss_res = float(((p - t) ** 2).sum())
            ss_tot = float(((t - t.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
        else:
            r2 = float('nan')
        result[str(c)] = {'n': n, 'mse': mse, 'mae': mae, 'r2': r2}
    return result


# ── CLI ──

def parse_args():
    parser = argparse.ArgumentParser(description='NexuST Density Prediction')
    add_shared_args(parser)

    parser.add_argument('--pretrain_ckpt', type=str, required=True)
    parser.add_argument('--radius_idx', type=int, required=True)
    parser.add_argument('--radii', type=float, nargs='+')
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--run_id', type=str, default=None)

    parser.set_defaults(
        output_dir=str(Path.cwd() / 'output/results/density/nexust'),
        max_gene_len=300,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset_name or Path(args.data_path).stem
    mode = "finetune"
    seed = args.seed
    pl.seed_everything(seed, workers=True)

    ds_cfg = DATASET_CONFIG.get(dataset_name)
    radii = args.radii or (ds_cfg.density_radii if ds_cfg else [])
    if not radii or not 0 <= args.radius_idx < len(radii):
        raise ValueError("Set --radii and choose a valid --radius_idx")
    actual_radius = float(radii[args.radius_idx])
    label_col = args.label_col or (ds_cfg.label_col if ds_cfg else None)

    print(f"Density prediction: radius_idx={args.radius_idx}, radius={actual_radius}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = build_ckpt_dir(
        args.ckpt_root, "density", run_id, dataset_name, mode,
        f"r{args.radius_idx}", f"seed{seed}")
    print(f"Checkpoint dir: {ckpt_dir}")

    tokenizer = get_tokenizer()
    train_dir = Path(args.data_path) / "train"
    val_dir = Path(args.data_path) / "val"

    train_dataset = TrainDensityDataset(
        train_dir=str(train_dir), tokenizer=tokenizer,
        radius=actual_radius, max_gene_len=args.max_gene_len)

    val_dataset = ValDensityDataset(
        val_dir=str(val_dir), tokenizer=tokenizer,
        radius=actual_radius, label_col=label_col,
        max_gene_len=args.max_gene_len)

    collator = DensityCollator(max_gene_len=args.max_gene_len)

    model = NexuSTDensityPrediction(
        pretrain_ckpt=args.pretrain_ckpt,
        learning_rate=args.lr, encoder_lr=args.encoder_lr, min_lr=args.min_lr)

    recorder = DensityMetricsRecorder(save_dir=ckpt_dir)

    runner = TrainingRunner(
        seed=seed, max_epochs=args.max_epochs, devices=args.devices,
        batch_size=args.batch_size, accumulate_grad_batches=args.accumulate_grad_batches,
        num_workers=args.num_workers,
        monitor='val_mse', monitor_mode='min', early_stop_patience=5,
        ckpt_dir=ckpt_dir,
        ckpt_filename=f"seed{seed}_" + "epoch{epoch:02d}_mse{val_mse:.6f}",
        wandb_project=args.project, logger=args.logger,
        accelerator=args.accelerator, precision=args.precision,
        wandb_group=args.group or f"{dataset_name}-density-{mode}",
        wandb_name=f"nexust-linear_{dataset_name}_{mode}_seed{seed}_lr{args.lr:g}_r{args.radius_idx}",
        wandb_config={
            "head": "linear", "pretrained_head": False, "phase": args.phase,
            "task": "density", "dataset": dataset_name, "mode": mode,
            "radius_idx": args.radius_idx, "radius": actual_radius,
            "model": "nexust", "lr": args.lr, "encoder_lr": args.encoder_lr,
            "batch_size": args.batch_size, "seed": seed,
        },
    )

    runner.fit(model, train_dataset, val_dataset, collator, callbacks=[recorder])

    if get_global_rank() != 0:
        return

    results = recorder.read_results()
    r2 = results.get('r2', float('nan'))
    mse = results.get('mse', float('inf'))
    mae = results.get('mae', float('inf'))

    csv_path = append_csv(
        args.output_dir, f"{dataset_name}.csv",
        header=['dataset', 'mode', 'radius_idx', 'radius', 'seed', 'mse', 'mae', 'r2'],
        row=[dataset_name, mode, args.radius_idx, f'{actual_radius:.4f}',
             seed, f'{mse:.6f}', f'{mae:.6f}', f'{r2:.4f}'])

    per_celltype = results.get('per_celltype')
    if per_celltype:
        append_per_group_csv(
            args.output_dir, f"{dataset_name}_center",
            index_cols=['dataset', 'mode', 'radius_idx', 'radius', 'seed'],
            index_values=[dataset_name, mode, args.radius_idx,
                          f'{actual_radius:.4f}', seed],
            metric_names=['mse', 'mae', 'r2'],
            per_group=per_celltype,
            float_fmt={'r2': '.4f'})

    print(f"Seed {seed}: mse={mse:.6f}, mae={mae:.6f}, r2={r2:.4f}, epoch={results.get('epoch', -1)}")
    print(f"Results: {csv_path} | Metrics JSON: {ckpt_dir / 'best_metrics.json'}")


if __name__ == '__main__':
    main()
