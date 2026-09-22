import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from data.tokenizer import get_tokenizer, Tokenizer
from data.collator import DataCollator
from finetune.utils.dataset import TrainFinetuneDataset, ValFinetuneDataset
from finetune.utils.tools import load_encoder
from finetune.utils.callbacks import BaseMetricsRecorder
from finetune.utils.cli import add_shared_args, build_ckpt_dir, append_csv
from finetune.utils.training import TrainingRunner, get_global_rank


class TrainClassificationDataset(TrainFinetuneDataset):
    def __init__(
        self,
        train_dir: str,
        tokenizer: Tokenizer,
        label_col: str,
        label_encoder: LabelEncoder = None,
        max_gene_len: int = 800,
        n_spots: int = 512,
    ):
        super().__init__(train_dir=train_dir, tokenizer=tokenizer, n_spots=n_spots, max_gene_len=max_gene_len)
        self.label_col = label_col

        if label_encoder is None:
            all_labels = []
            for slide in self.slides:
                adata_backed = slide.open_backed()
                labels = adata_backed.obs[label_col].values
                all_labels.extend(labels)
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_labels)
        else:
            self.label_encoder = label_encoder

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        fov_idx, _, _ = self.index_map[idx]
        slide = self.slides[fov_idx]
        adata_backed = slide.open_backed()
        spot_idx = batch.pop('spot_idx')  # [n_spots], -1 for padding
        cell_mask = batch.get('cell_padding_mask', None)

        labels_full = torch.full((spot_idx.shape[0],), -100, dtype=torch.long)
        if cell_mask is None:
            real_idx = spot_idx
            valid = real_idx >= 0
        else:
            valid = (~cell_mask) & (spot_idx >= 0)
            real_idx = spot_idx[valid]

        labels_raw = adata_backed.obs[self.label_col].values[real_idx.cpu().numpy()]
        labels_encoded = self.label_encoder.transform(labels_raw)
        labels_full[valid] = torch.tensor(labels_encoded, dtype=torch.long)

        batch['label'] = labels_full
        return batch


class ClassificationCollator(DataCollator):
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        labels = [sample['label'] for sample in batch]
        collated = super().__call__(batch)
        collated['labels'] = torch.stack(labels, dim=0)
        return collated


class ValClassificationDataset(ValFinetuneDataset):
    def __init__(
        self,
        val_dir: str,
        tokenizer: Tokenizer,
        label_col: str,
        label_encoder: LabelEncoder,
        max_gene_len: int = 1000,
        max_spots: int = 512,
    ):
        super().__init__(val_dir=val_dir, tokenizer=tokenizer, max_gene_len=max_gene_len, max_cells=max_spots)
        self.label_col = label_col
        self.label_encoder = label_encoder

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        slide_idx, cell_indices = self.index_map[idx]
        slide = self.slides[slide_idx]
        adata_backed = slide.open_backed()
        labels_raw = adata_backed.obs[self.label_col].values[cell_indices]
        labels_encoded = self.label_encoder.transform(labels_raw)
        batch['label'] = torch.tensor(labels_encoded, dtype=torch.long)
        return batch


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, num_classes, bias=True),
        )

    def forward(self, x):
        return self.classifier(x)


class NexuSTClassifier(pl.LightningModule):
    def __init__(
        self,
        pretrain_ckpt: str,
        num_classes: int,
        head_lr: float = 1e-3,
        encoder_lr: float = 1e-5,
        weight_decay: float = 0.01,
        min_lr: float = 1e-6,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.head_lr = head_lr
        self.encoder_lr = encoder_lr
        self.weight_decay = weight_decay
        self.min_lr = min_lr

        self.nexust_encoder = load_encoder(pretrain_ckpt, use_activation_checkpointing=gradient_checkpointing)

        d_model = self.nexust_encoder.d_model
        self.classification_head = ClassificationHead(d_model, num_classes)

        self.val_acc_metric = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_loss_metric = MeanMetric()
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        # Collect all val predictions for per-celltype metrics
        self._val_preds = []
        self._val_labels = []

    def forward(self, gene_ids, gene_values, coords, organ_ids, cell_padding_mask=None):
        out = self.nexust_encoder(gene_ids, gene_values, organ_ids, coords, cell_padding_mask=cell_padding_mask)
        cls_token = out['cls_token']
        logits = self.classification_head(cls_token)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(
            gene_ids=batch['gene_ids'], gene_values=batch['gene_values'],
            coords=batch['coords'], organ_ids=batch['organ_ids'],
            cell_padding_mask=batch.get('cell_padding_mask', None),
        )

        labels = batch['labels']
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
        preds = logits_flat.argmax(dim=-1)

        valid_mask = labels_flat != -100
        valid_preds = preds[valid_mask]
        valid_labels = labels_flat[valid_mask]

        acc = (valid_preds == valid_labels).float().mean()
        f1 = self.train_f1(valid_preds, valid_labels)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log('train_f1', f1, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)
        return loss

    def on_validation_epoch_start(self):
        self.val_acc_metric.reset()
        self.val_f1_metric.reset()
        self.val_loss_metric.reset()
        self._val_preds.clear()
        self._val_labels.clear()

    def validation_step(self, batch, batch_idx):
        logits = self(
            gene_ids=batch['gene_ids'], gene_values=batch['gene_values'],
            coords=batch['coords'], organ_ids=batch['organ_ids'],
        )

        labels = batch['labels']
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        loss = F.cross_entropy(logits_flat, labels_flat)
        preds = logits_flat.argmax(dim=-1)

        self.val_loss_metric.update(loss)
        self.val_acc_metric.update(preds, labels_flat)
        self.val_f1_metric.update(preds, labels_flat)

        self._val_preds.append(preds.cpu())
        self._val_labels.append(labels_flat.cpu())

    def on_validation_epoch_end(self):
        val_loss = self.val_loss_metric.compute()
        acc = self.val_acc_metric.compute()
        f1 = self.val_f1_metric.compute()

        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_f1', f1, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.nexust_encoder.parameters(), 'lr': self.encoder_lr},
            {'params': self.classification_head.parameters(), 'lr': self.head_lr},
        ], weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        min_lr = self.min_lr

        def lr_lambda_fn(current_step: int):
            base_lr = optimizer.param_groups[0]['lr']
            progress = current_step / float(max(1, total_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_factor = min_lr / max(base_lr, 1e-12)
            return cosine * (1 - min_factor) + min_factor

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fn),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_val_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return concatenated (preds, labels) from last validation epoch."""
        if not self._val_preds:
            return np.array([]), np.array([])
        return torch.cat(self._val_preds).numpy(), torch.cat(self._val_labels).numpy()


class ClassificationMetricsRecorder(BaseMetricsRecorder):
    def __init__(self, label_encoder: LabelEncoder, save_dir: Path | None = None):
        super().__init__(primary_metric='acc', mode='max', save_dir=save_dir)
        self.label_encoder = label_encoder

    def _extract_metrics(self, trainer, pl_module):
        acc = trainer.callback_metrics.get('val_acc')
        f1 = trainer.callback_metrics.get('val_f1')
        if acc is None:
            return None
        metrics = {
            'acc': acc.item(),
            'f1': f1.item() if f1 is not None else float('nan'),
        }
        preds, labels = pl_module.get_val_predictions()
        if len(preds) > 0:
            class_names = list(self.label_encoder.classes_)
            all_labels = list(range(len(class_names)))
            report = classification_report(
                labels, preds, labels=all_labels, target_names=class_names, output_dict=True, zero_division=0)
            metrics['per_class'] = {
                name: {k: report[name][k] for k in ('precision', 'recall', 'f1-score', 'support')}
                for name in class_names if name in report
            }
            metrics['confusion_matrix'] = confusion_matrix(labels, preds).tolist()
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='NexuST Cell Annotation')
    add_shared_args(parser)

    # Task-specific
    parser.add_argument('--pretrain_ckpt', type=str, required=True)
    parser.add_argument('--label_col', type=str, required=True)
    parser.add_argument('--head_lr', type=float, default=None)
    parser.add_argument('--n_spots', type=int, default=512)
    parser.add_argument('--gradient_checkpointing', action='store_true')

    parser.set_defaults(output_dir=str(Path.cwd() / 'output/results/cell_annotation/nexust'))
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset_name or Path(args.data_path).stem
    mode = "finetune"
    seed = args.seed
    pl.seed_everything(seed, workers=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = build_ckpt_dir(args.ckpt_root, "cell_annotation", timestamp, dataset_name, mode, f"seed{seed}")

    # Datasets
    tokenizer = get_tokenizer()
    train_dir = Path(args.data_path) / "train"
    val_dir = Path(args.data_path) / "val"

    train_dataset = TrainClassificationDataset(
        train_dir=str(train_dir), tokenizer=tokenizer,
        label_col=args.label_col, max_gene_len=args.max_gene_len, n_spots=args.n_spots)
    label_encoder = train_dataset.label_encoder
    num_classes = len(label_encoder.classes_)

    val_dataset = ValClassificationDataset(
        val_dir=str(val_dir), tokenizer=tokenizer,
        label_col=args.label_col, label_encoder=label_encoder,
        max_gene_len=args.max_gene_len, max_spots=args.n_spots)

    collator = ClassificationCollator(max_gene_len=args.max_gene_len)

    # Model
    model = NexuSTClassifier(
        pretrain_ckpt=args.pretrain_ckpt, num_classes=num_classes,
        head_lr=args.head_lr or args.lr, encoder_lr=args.encoder_lr,
        min_lr=args.min_lr,
        gradient_checkpointing=args.gradient_checkpointing)

    # Train
    recorder = ClassificationMetricsRecorder(label_encoder=label_encoder, save_dir=ckpt_dir)

    runner = TrainingRunner(
        seed=seed, max_epochs=args.max_epochs, devices=args.devices,
        batch_size=args.batch_size, accumulate_grad_batches=args.accumulate_grad_batches,
        num_workers=args.num_workers,
        monitor='val_acc', monitor_mode='max', early_stop_patience=5,
        ckpt_dir=ckpt_dir,
        ckpt_filename=f"seed{seed}_" + "epoch{epoch:02d}_acc{val_acc:.4f}",
        wandb_project=args.project, logger=args.logger,
        accelerator=args.accelerator, precision=args.precision,
        wandb_group=args.group or f"{dataset_name}-cls-{mode}",
        wandb_name=f"nexust-linear_{dataset_name}_{mode}_seed{seed}_lr{args.lr:g}",
        wandb_config={
            "mode": mode, "lr": args.head_lr or args.lr,
            "head": "linear", "pretrained_head": False, "phase": args.phase,
            "task": "cell_annotation", "dataset": dataset_name, "model": "nexust",
            "head_lr": args.head_lr or args.lr, "encoder_lr": args.encoder_lr,
            "batch_size": args.batch_size, "seed": seed,
        },
    )

    runner.fit(model, train_dataset, val_dataset, collator, callbacks=[recorder])

    # Results (rank 0 only)
    if get_global_rank() == 0:
        results = recorder.read_results()
        acc = results.get('acc', float('nan'))
        f1 = results.get('f1', float('nan'))

        csv_path = append_csv(
            args.output_dir, f"{dataset_name}.csv",
            header=['dataset', 'mode', 'seed', 'accuracy', 'f1'],
            row=[dataset_name, mode, seed, f'{acc:.4f}', f'{f1:.4f}'])

        print(f"Seed {seed}: acc={acc:.4f}, f1={f1:.4f}, epoch={results.get('epoch', -1)}")
        print(f"Results: {csv_path} | Metrics JSON: {ckpt_dir / 'best_metrics.json'}")


if __name__ == '__main__':
    main()
