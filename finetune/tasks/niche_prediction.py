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
from torchmetrics.regression import MeanSquaredError

PROJECT_ROOT = Path(__file__).parent.parent.parent

from data.tokenizer import get_tokenizer, Tokenizer
from data.collator import DataCollator
from finetune.utils.dataset import TrainFinetuneDataset, ValFinetuneDataset
from finetune.utils.tools import load_encoder, ColumnWisePearsonCorrCoef, ColumnWiseMAE
from finetune.utils.callbacks import BaseMetricsRecorder
from finetune.utils.cli import add_shared_args, build_ckpt_dir, append_csv
from finetune.utils.training import TrainingRunner, get_global_rank


class TrainNicheDataset(TrainFinetuneDataset):
    def __init__(
        self,
        train_dir: str,
        tokenizer: Tokenizer,
        niche_key: str = 'X_niche',
        n_spots: int = 512,
        max_gene_len: int = 300,
    ):
        self.niche_key = niche_key
        super().__init__(train_dir, tokenizer, n_spots=n_spots, max_gene_len=max_gene_len)
        self.n_cell_types = self._get_n_cell_types()

    def _get_n_cell_types(self) -> int:
        slide = self.slides[0]
        adata = slide.open_backed()
        if self.niche_key not in adata.obsm:
            raise KeyError(f"Niche composition '{self.niche_key}' not found. "
                          f"Please precompute using compute_niche_composition().")
        return adata.obsm[self.niche_key].shape[1]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        fov_idx, _, _ = self.index_map[idx]
        slide = self.slides[fov_idx]
        adata_backed = slide.open_backed()
        spot_idx = batch.pop('spot_idx')

        niche = adata_backed.obsm[self.niche_key]
        if hasattr(niche, 'toarray'):
            niche = niche.toarray()
        niche = np.asarray(niche, dtype=np.float32)

        niche_labels = np.zeros((self.n_spots, self.n_cell_types), dtype=np.float32)
        valid = spot_idx >= 0
        valid_idx = spot_idx[valid].cpu().numpy()
        niche_labels[valid.cpu().numpy()] = niche[valid_idx]

        batch['niche_labels'] = torch.from_numpy(niche_labels)
        return batch


class NicheCollator(DataCollator):
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        niche_labels = [sample.pop('niche_labels') for sample in batch]
        center_types = [sample.pop('center_type', None) for sample in batch]
        collated = super().__call__(batch)
        collated['niche_labels'] = torch.stack(niche_labels, dim=0)
        if center_types[0] is not None:
            collated['center_type'] = np.concatenate(center_types, axis=0)
        return collated


class ValNicheDataset(ValFinetuneDataset):
    def __init__(
        self,
        val_dir: str,
        tokenizer: Tokenizer,
        niche_key: str = 'X_niche',
        max_gene_len: int = 300,
        max_cells: int = 512,
    ):
        self.niche_key = niche_key
        super().__init__(val_dir, tokenizer, max_gene_len=max_gene_len, max_cells=max_cells)
        self.n_cell_types = self._get_n_cell_types()

    def _get_n_cell_types(self) -> int:
        slide = self.slides[0]
        adata = slide.open_backed()
        if self.niche_key not in adata.obsm:
            raise KeyError(f"Niche composition '{self.niche_key}' not found.")
        return adata.obsm[self.niche_key].shape[1]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        slide_idx, cell_indices = self.index_map[idx]
        slide = self.slides[slide_idx]
        adata_backed = slide.open_backed()

        niche = adata_backed.obsm[self.niche_key][cell_indices]
        if hasattr(niche, 'toarray'):
            niche = niche.toarray()
        niche_labels = np.asarray(niche, dtype=np.float32)

        batch['niche_labels'] = torch.from_numpy(niche_labels)

        niche_info = adata_backed.uns.get('niche_info', {})
        ct_col = niche_info.get('cell_type_col')
        if ct_col and ct_col in adata_backed.obs.columns:
            batch['center_type'] = adata_backed.obs[ct_col].values[cell_indices]

        return batch


class NichePredictionHead(nn.Module):
    def __init__(self, d_model: int, n_cell_types: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_cell_types, bias=False)

    def forward(self, x):
        return self.linear(x)


class NexuSTNichePrediction(pl.LightningModule):
    def __init__(
        self,
        pretrain_ckpt: str,
        n_cell_types: int,
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
        self.n_cell_types = n_cell_types
        self.niche_head = NichePredictionHead(d_model, n_cell_types)

        self.val_mse_metric = MeanSquaredError()
        self.val_mae_metric = ColumnWiseMAE(n_features=n_cell_types)
        self.val_pearson_metric = ColumnWisePearsonCorrCoef(n_features=n_cell_types)

        # Collect val predictions for per-center metrics (populated each val
        # epoch, consumed by NicheMetricsRecorder in on_validation_end).
        # Val dataloader is un-sharded (every rank traverses full val set),
        # so rank-local concatenation already equals the global set.
        self._val_preds: list = []
        self._val_targets: list = []
        self._val_center_types: list = []

    def forward(self, gene_ids, gene_values, coords, organ_ids, cell_padding_mask=None):
        out = self.nexust_encoder(
            gene_ids, gene_values, organ_ids, coords,
            cell_padding_mask=cell_padding_mask)
        cls_token = out['cls_token']
        logits = self.niche_head(cls_token)

        s = F.softplus(logits) + 1e-8
        niche_pred = s / s.sum(dim=-1, keepdim=True)
        return niche_pred

    def _flatten_and_normalize(self, pred, target, cell_padding_mask):
        pred_flat = pred.view(-1, pred.size(-1))
        target_flat = target.view(-1, target.size(-1))
        if cell_padding_mask is not None:
            valid_mask = ~cell_padding_mask.view(-1)
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]

        s = target_flat + 1e-8
        target_flat = s / s.sum(dim=-1, keepdim=True)
        return pred_flat, target_flat

    def training_step(self, batch, batch_idx):
        pred = self(
            batch['gene_ids'], batch['gene_values'], batch['coords'],
            batch['organ_ids'], cell_padding_mask=batch.get('cell_padding_mask'))

        pred_flat, target_flat = self._flatten_and_normalize(
            pred, batch['niche_labels'], batch.get('cell_padding_mask'))

        loss = F.mse_loss(pred_flat, target_flat)
        mae = F.l1_loss(pred_flat, target_flat)

        self.log('train_mse', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_mse_metric.reset()
        self.val_mae_metric.reset()
        self.val_pearson_metric.reset()
        self._val_preds.clear()
        self._val_targets.clear()
        self._val_center_types.clear()

    def validation_step(self, batch, batch_idx):
        pred = self(
            batch['gene_ids'], batch['gene_values'], batch['coords'],
            batch['organ_ids'], cell_padding_mask=batch.get('cell_padding_mask'))

        pred_flat, target_flat = self._flatten_and_normalize(
            pred, batch['niche_labels'], batch.get('cell_padding_mask'))

        self.val_mse_metric.update(pred_flat, target_flat)
        self.val_mae_metric.update(pred_flat, target_flat)
        self.val_pearson_metric.update(pred_flat, target_flat)

        # Collect for per-center stats. Val has no cell_padding_mask (chunks
        # carry only real cells), so center_type length matches pred_flat length.
        self._val_preds.append(pred_flat.detach().float().cpu().numpy())
        self._val_targets.append(target_flat.detach().float().cpu().numpy())
        center_type = batch.get('center_type')
        if center_type is not None:
            self._val_center_types.append(center_type)

        n = pred_flat.size(0)
        self.log('val_mse', F.mse_loss(pred_flat, target_flat),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=n)
        self.log('val_mae', F.l1_loss(pred_flat, target_flat),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=n)

    def on_validation_epoch_end(self):
        pearson = self.val_pearson_metric.compute()
        self.log('val_pearson', pearson, prog_bar=True, sync_dist=True)

    def get_val_predictions(self):
        """Return concatenated (preds, targets, center_types) from last val epoch.

        center_types is None if ValNicheDataset was built without niche_info
        cell_type_col (no per-center breakdown available).
        """
        if not self._val_preds:
            return np.array([]), np.array([]), None
        preds = np.concatenate(self._val_preds, axis=0)
        targets = np.concatenate(self._val_targets, axis=0)
        center_types = (np.concatenate(self._val_center_types, axis=0)
                        if self._val_center_types else None)
        return preds, targets, center_types

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.nexust_encoder.parameters(), 'lr': self.encoder_lr},
            {'params': self.niche_head.parameters(), 'lr': self.learning_rate},
        ], weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=self.min_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class NicheMetricsRecorder(BaseMetricsRecorder):
    def __init__(self, save_dir=None):
        super().__init__(primary_metric='mse', mode='min', save_dir=save_dir)

    def _extract_metrics(self, trainer, pl_module):
        mse = trainer.callback_metrics.get('val_mse')
        mae = trainer.callback_metrics.get('val_mae')
        if mse is None:
            return None
        pcc = pl_module.val_pearson_metric.compute()
        metrics = {
            'mse': mse.item(),
            'mae': mae.item() if mae is not None else float('inf'),
            'pcc': pcc.item(),
        }
        preds, targets, center_types = pl_module.get_val_predictions()
        if center_types is not None and len(preds) > 0:
            metrics['per_center'] = _per_center_metrics(preds, targets, center_types)
        return metrics


def _per_center_metrics(preds: np.ndarray, targets: np.ndarray,
                        center_types: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Group niche predictions by the center cell's type and compute
    mse/mae/pcc/n per group.

    Preds and targets are (N, C) probability distributions (already normalized
    in _flatten_and_normalize). pcc is the average Pearson correlation over
    the C target columns.
    """
    result: Dict[str, Dict[str, float]] = {}
    unique = np.unique(center_types)
    for c in unique:
        mask = center_types == c
        p = preds[mask]
        t = targets[mask]
        n = int(mask.sum())
        if n == 0:
            continue
        mse = float(((p - t) ** 2).mean())
        mae = float(np.abs(p - t).mean())
        if n >= 2:
            p_c = p - p.mean(axis=0, keepdims=True)
            t_c = t - t.mean(axis=0, keepdims=True)
            num = (p_c * t_c).sum(axis=0)
            denom = np.sqrt((p_c ** 2).sum(axis=0) * (t_c ** 2).sum(axis=0))
            valid = denom > 1e-12
            pcc = float(np.nanmean(np.where(valid, num / np.where(valid, denom, 1.0), np.nan)))
        else:
            pcc = float('nan')
        result[str(c)] = {'n': n, 'mse': mse, 'mae': mae, 'pcc': pcc}
    return result


def _read_niche_metadata(data_path: str, niche_key: str, max_gene_len: int):
    tokenizer = get_tokenizer()
    val_dir = Path(data_path) / "val"
    temp_dataset = ValNicheDataset(
        val_dir=str(val_dir), tokenizer=tokenizer,
        niche_key=niche_key, max_gene_len=max_gene_len)
    n_cell_types = temp_dataset.n_cell_types

    reference_slide = temp_dataset.slides[0]
    reference_info = reference_slide.open_backed().uns.get('niche_info', {})
    cell_type_names = list(reference_info.get('columns', []))
    radii_list = list(reference_info.get('radii', []))
    del temp_dataset
    return n_cell_types, cell_type_names, radii_list


# ── CLI ──

def parse_args():
    parser = argparse.ArgumentParser(description='NexuST Niche Prediction')
    add_shared_args(parser)

    # Task-specific
    parser.add_argument('--pretrain_ckpt', type=str, required=True)
    parser.add_argument('--niche_key', type=str, default='X_niche')
    parser.add_argument('--radius_idx', type=int, required=True)
    parser.add_argument('--run_id', type=str, default=None)

    parser.set_defaults(
        output_dir=str(Path.cwd() / 'output/results/niche_prediction/nexust'),
        max_gene_len=300,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset_name or Path(args.data_path).stem
    mode = "finetune"
    seed = args.seed
    pl.seed_everything(seed, workers=True)

    effective_niche_key = f"{args.niche_key}_{args.radius_idx}"

    n_cell_types, cell_type_names, radii_list = _read_niche_metadata(
        args.data_path, effective_niche_key, args.max_gene_len)

    if args.radius_idx >= len(radii_list):
        raise ValueError(f"radius_idx={args.radius_idx} out of range for radii={radii_list}")
    actual_radius = radii_list[args.radius_idx]

    print(f"Niche prediction: {n_cell_types} cell types, radius_idx={args.radius_idx}, radius={actual_radius}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = build_ckpt_dir(
        args.ckpt_root, "niche", run_id, dataset_name, mode, f"r{args.radius_idx}", f"seed{seed}")
    print(f"Checkpoint dir: {ckpt_dir}")

    # Datasets
    tokenizer = get_tokenizer()
    train_dir = Path(args.data_path) / "train"
    val_dir = Path(args.data_path) / "val"

    train_dataset = TrainNicheDataset(
        train_dir=str(train_dir), tokenizer=tokenizer,
        niche_key=effective_niche_key, max_gene_len=args.max_gene_len)

    val_dataset = ValNicheDataset(
        val_dir=str(val_dir), tokenizer=tokenizer,
        niche_key=effective_niche_key, max_gene_len=args.max_gene_len)

    collator = NicheCollator(max_gene_len=args.max_gene_len)
    n_cell_types = train_dataset.n_cell_types

    # Model
    model = NexuSTNichePrediction(
        pretrain_ckpt=args.pretrain_ckpt, n_cell_types=n_cell_types,
        learning_rate=args.lr,
        encoder_lr=args.encoder_lr, min_lr=args.min_lr)

    # Train
    recorder = NicheMetricsRecorder(save_dir=ckpt_dir)

    runner = TrainingRunner(
        seed=seed, max_epochs=args.max_epochs, devices=args.devices,
        batch_size=args.batch_size, accumulate_grad_batches=args.accumulate_grad_batches,
        num_workers=args.num_workers,
        monitor='val_mse', monitor_mode='min', early_stop_patience=5,
        ckpt_dir=ckpt_dir,
        ckpt_filename=f"seed{seed}_" + "epoch{epoch:02d}_mse{val_mse:.6f}",
        wandb_project=args.project, logger=args.logger,
        accelerator=args.accelerator, precision=args.precision,
        wandb_group=args.group or f"{dataset_name}-niche-{mode}",
        wandb_name=f"nexust-linear_{dataset_name}_{mode}_seed{seed}_lr{args.lr:g}_r{args.radius_idx}",
        wandb_config={
            "head": "linear", "pretrained_head": False, "phase": args.phase,
            "task": "niche_prediction", "dataset": dataset_name, "mode": mode,
            "radius_idx": args.radius_idx, "radius": actual_radius,
            "model": "nexust", "lr": args.lr, "encoder_lr": args.encoder_lr,
            "batch_size": args.batch_size, "seed": seed,
        },
    )

    runner.fit(model, train_dataset, val_dataset, collator, callbacks=[recorder])

    # Results (rank 0 only)
    if get_global_rank() != 0:
        return

    results = recorder.read_results()
    mse = results.get('mse', float('inf'))
    mae = results.get('mae', float('inf'))
    pcc = results.get('pcc', -1.0)

    append_csv(
        args.output_dir, f"{dataset_name}.csv",
        header=['dataset', 'mode', 'radius_idx', 'radius', 'seed', 'mse', 'mae', 'pcc'],
        row=[dataset_name, mode, args.radius_idx, f'{actual_radius:.4f}',
             seed, f'{mse:.6f}', f'{mae:.6f}', f'{pcc:.4f}'])

    print(f"Seed {seed}: mse={mse:.6f}, mae={mae:.6f}, pcc={pcc:.4f}, epoch={results.get('epoch', -1)}")
    print(f"Results saved to {Path(args.output_dir) / f'{dataset_name}.csv'}")


if __name__ == '__main__':
    main()
