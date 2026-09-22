from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from evaluation.metrics import ColumnWisePearsonCorrCoef, ColumnWiseMAE
from probe.callbacks import (
    RegressionMetricsRecorder,
    ClassificationMetricsRecorder,
    NicheMetricsRecorder,
)


class BaseProbe(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        output_dim: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(d_model, output_dim, bias=True)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.linear(x)

    @abstractmethod
    def training_step(self, batch, batch_idx):
        ...

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class RegressionProbe(BaseProbe):
    def __init__(self, d_model: int, output_dim: int = 1, **kwargs):
        super().__init__(d_model, output_dim, **kwargs)
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()

    def forward(self, x):
        out = super().forward(x)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log("train_mse", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_mse.reset()
        self.val_mae.reset()
        self.val_r2.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        self.val_mse.update(pred, y)
        self.val_mae.update(pred, y)
        self.val_r2.update(pred, y)

    def on_validation_epoch_end(self):
        self.log("val_mse", self.val_mse.compute(), prog_bar=True)
        self.log("val_mae", self.val_mae.compute(), prog_bar=True)
        self.log("val_r2", self.val_r2.compute(), prog_bar=True)


class ClassificationProbe(BaseProbe):
    def __init__(self, d_model: int, n_classes: int, **kwargs):
        super().__init__(d_model, n_classes, **kwargs)
        self.n_classes = n_classes
        self.val_acc = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.val_f1 = MulticlassF1Score(num_classes=n_classes, average="macro")

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_acc.reset()
        self.val_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=-1)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)


class ImputationProbe(RegressionProbe):
    def __init__(self, d_model: int, n_genes: int, **kwargs):
        super().__init__(d_model, output_dim=n_genes, **kwargs)
        self.n_genes = n_genes
        self.val_mae = ColumnWiseMAE(n_genes)
        self.val_pcc = ColumnWisePearsonCorrCoef(n_genes)

    def on_validation_epoch_start(self):
        self.val_mse.reset()
        self.val_mae.reset()
        self.val_pcc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        self.val_mse.update(pred, y)
        self.val_mae.update(pred, y)
        self.val_pcc.update(pred, y)

    def on_validation_epoch_end(self):
        self.log("val_mse", self.val_mse.compute(), prog_bar=True)
        self.log("val_mae", self.val_mae.compute(), prog_bar=True)
        self.log("val_pcc", self.val_pcc.compute(), prog_bar=True)


class NicheProbe(RegressionProbe):
    def __init__(self, d_model: int, n_cell_types: int, **kwargs):
        super().__init__(d_model, output_dim=n_cell_types, **kwargs)
        # bias=True (matches BaseProbe): the bias lets the head output a
        # non-uniform mean composition. Without it, mean-centered features
        # (e.g. PCA) cannot anchor the mean and the prediction undershoots
        # toward uniform — catastrophic on dominant niches (tumor ~0.8),
        # showing up as below-baseline (negative-skill) MAE for PCA only.
        self.linear = nn.Linear(d_model, n_cell_types, bias=True)
        self.n_cell_types = n_cell_types
        self.val_mae = ColumnWiseMAE(n_cell_types)
        self.val_pcc = ColumnWisePearsonCorrCoef(n_cell_types)

    def forward(self, x):
        s = F.softplus(self.linear(x)) + 1e-8
        return s / s.sum(dim=-1, keepdim=True)

    def on_validation_epoch_start(self):
        self.val_mse.reset()
        self.val_mae.reset()
        self.val_pcc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        self.val_mse.update(pred, y)
        self.val_mae.update(pred, y)
        self.val_pcc.update(pred, y)

    def on_validation_epoch_end(self):
        self.log("val_mse", self.val_mse.compute(), prog_bar=True)
        self.log("val_mae", self.val_mae.compute(), prog_bar=True)
        self.log("val_pcc", self.val_pcc.compute(), prog_bar=True)


def _build_trainer(seed, max_epochs, patience, monitor, mode, num_workers,
                   devices, recorder, batch_size, train_emb, train_lab,
                   val_emb, val_lab, accelerator, precision):
    """Shared DataLoader + Trainer construction."""
    pl.seed_everything(seed, workers=True)

    train_dataset = TensorDataset(torch.from_numpy(train_emb), torch.from_numpy(train_lab))
    val_dataset = TensorDataset(torch.from_numpy(val_emb), torch.from_numpy(val_lab))

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False, generator=g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    early_stop = EarlyStopping(monitor=monitor, patience=patience, mode=mode, verbose=True)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[recorder, early_stop],
        logger=False,
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        log_every_n_steps=10,
        enable_checkpointing=False,
    )

    return trainer, train_loader, val_loader


def run_regression_probe(
    train_emb: np.ndarray,
    train_lab: np.ndarray,
    val_emb: np.ndarray,
    val_lab: np.ndarray,
    d_model: int,
    seed: int,
    output_dim: int = 1,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 0.001,
    max_epochs: int = 50,
    num_workers: int = 4,
    devices: int = 1,
    patience: int = 10,
    accelerator: str = "auto",
    precision: str = "bf16-true",
) -> Tuple[Dict[str, float], int]:
    """Train a RegressionProbe. Returns (metrics_dict, best_epoch)."""
    pl.seed_everything(seed, workers=True)
    model = RegressionProbe(d_model=d_model, output_dim=output_dim,
                            learning_rate=lr, weight_decay=weight_decay)
    recorder = RegressionMetricsRecorder()

    trainer, train_loader, val_loader = _build_trainer(
        seed, max_epochs, patience, "val_mse", "min",
        num_workers, devices, recorder, batch_size,
        train_emb, train_lab, val_emb, val_lab, accelerator, precision,
    )
    trainer.fit(model, train_loader, val_loader)
    return recorder.results, recorder.best_epoch


def run_classification_probe(
    train_emb: np.ndarray,
    train_lab: np.ndarray,
    val_emb: np.ndarray,
    val_lab: np.ndarray,
    d_model: int,
    n_classes: int,
    seed: int,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 0.001,
    max_epochs: int = 50,
    num_workers: int = 4,
    devices: int = 1,
    patience: int = 10,
    accelerator: str = "auto",
    precision: str = "bf16-true",
) -> Tuple[Dict[str, float], int, np.ndarray]:
    """Train a ClassificationProbe. Returns (metrics_dict, best_epoch, val_preds)."""
    pl.seed_everything(seed, workers=True)
    model = ClassificationProbe(d_model=d_model, n_classes=n_classes,
                                learning_rate=lr, weight_decay=weight_decay)
    recorder = ClassificationMetricsRecorder()

    trainer, train_loader, val_loader = _build_trainer(
        seed, max_epochs, patience, "val_f1", "max",
        num_workers, devices, recorder, batch_size,
        train_emb, train_lab, val_emb, val_lab, accelerator, precision,
    )
    trainer.fit(model, train_loader, val_loader)

    # Restore best-epoch weights so post-fit forward produces preds consistent
    # with recorder.results (which are captured at best_f1 epoch). autocast
    # mirrors Trainer(precision='bf16'); without it fp32 logits shift borderline
    # argmax and CM / per_class drift from CSV accuracy by up to ~3%.
    model.load_state_dict(recorder.best_state)
    model.eval()
    val_preds = _predict(model, val_loader, precision).argmax(axis=-1)

    return recorder.results, recorder.best_epoch, val_preds


def run_niche_probe(
    train_emb: np.ndarray,
    train_lab: np.ndarray,
    val_emb: np.ndarray,
    val_lab: np.ndarray,
    d_model: int,
    n_cell_types: int,
    seed: int,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    max_epochs: int = 50,
    num_workers: int = 4,
    devices: int = 1,
    patience: int = 5,
    accelerator: str = "auto",
    precision: str = "bf16-true",
) -> Tuple[Dict[str, float], int]:
    pl.seed_everything(seed, workers=True)
    model = NicheProbe(d_model=d_model, n_cell_types=n_cell_types,
                       learning_rate=lr, weight_decay=weight_decay)
    recorder = NicheMetricsRecorder()

    trainer, train_loader, val_loader = _build_trainer(
        seed, max_epochs, patience, "val_mse", "min",
        num_workers, devices, recorder, batch_size,
        train_emb, train_lab, val_emb, val_lab, accelerator, precision,
    )
    trainer.fit(model, train_loader, val_loader)

    model.load_state_dict(recorder.best_state)
    model.eval()
    val_preds = _predict(model, val_loader, precision)

    return (recorder.results,
            recorder.best_per_col_mae,
            recorder.best_per_col_pcc,
            val_preds)


def run_imputation_probe(
    train_emb: np.ndarray,
    train_lab: np.ndarray,
    val_emb: np.ndarray,
    val_lab: np.ndarray,
    d_model: int,
    n_genes: int,
    seed: int,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-6,
    max_epochs: int = 100,
    num_workers: int = 4,
    devices: int = 1,
    patience: int = 5,
    accelerator: str = "auto",
    precision: str = "bf16-true",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Train an ImputationProbe. Returns (metrics, per_col_mae, per_col_pcc, val_preds)."""
    pl.seed_everything(seed, workers=True)
    model = ImputationProbe(d_model=d_model, n_genes=n_genes,
                            learning_rate=lr, weight_decay=weight_decay)
    recorder = NicheMetricsRecorder()

    trainer, train_loader, val_loader = _build_trainer(
        seed, max_epochs, patience, "val_mse", "min",
        num_workers, devices, recorder, batch_size,
        train_emb, train_lab, val_emb, val_lab, accelerator, precision,
    )
    trainer.fit(model, train_loader, val_loader)

    model.load_state_dict(recorder.best_state)
    model.eval()
    val_preds = _predict(model, val_loader, precision)

    return (recorder.results,
            recorder.best_per_col_mae,
            recorder.best_per_col_pcc,
            val_preds)



def _predict(model, loader, precision):
    """Return predictions with the fitted model's device and precision."""
    from contextlib import nullcontext
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    context = (torch.autocast(device.type, dtype=torch.bfloat16 if precision == "bf16-mixed" else torch.float16)
               if precision in ("bf16-mixed", "16-mixed") else nullcontext())
    outputs = []
    with torch.no_grad(), context:
        for x, _ in loader:
            outputs.append(model(x.to(device=device, dtype=dtype)).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)
