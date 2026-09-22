"""Shared training infrastructure for finetune tasks."""

import os
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb

from finetune.utils.tools import TrainEpochSetter, seed_worker


def get_local_rank() -> int:
    return int(os.environ.get('SLURM_LOCALID', os.environ.get('LOCAL_RANK', 0)))


def get_global_rank() -> int:
    return int(os.environ.get('SLURM_PROCID', os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0))))


class TrainingRunner:
    """Manages the shared training loop for all finetune tasks.

    Usage:
        runner = TrainingRunner(
            seed=42, max_epochs=50, devices=4, batch_size=16,
            accumulate_grad_batches=12, monitor="val_acc", monitor_mode="max",
            ckpt_dir=Path("..."), ckpt_filename="seed42_epoch{epoch:02d}",
            wandb_project="HiGeST-finetune", wandb_group="cls-finetune",
            wandb_name="cls_seed42", wandb_config={...},
        )
        recorder = MetricsRecorder(save_dir=ckpt_dir)
        ckpt_cb = runner.fit(model, train_dataset, val_dataset, collator, callbacks=[recorder])
        # recorder.best_xxx is now populated
    """

    def __init__(
        self,
        *,
        seed: int,
        max_epochs: int,
        devices: int,
        batch_size: int,
        accumulate_grad_batches: int,
        num_workers: int = 4,
        monitor: str,
        monitor_mode: str,
        early_stop_patience: int = 5,
        gradient_clip_val: Optional[float] = None,
        ckpt_dir: Path,
        ckpt_filename: str,
        wandb_project: str,
        wandb_group: str,
        wandb_name: str,
        wandb_config: dict,
        wandb_tags: Optional[list[str]] = None,
        logger: str = "csv",
        accelerator: str = "auto",
        precision: str = "bf16-mixed",
    ):
        self.logger = logger
        self.accelerator = accelerator
        self.precision = precision
        self.seed = seed
        self.max_epochs = max_epochs
        self.devices = devices
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.num_workers = num_workers
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.early_stop_patience = early_stop_patience
        self.gradient_clip_val = gradient_clip_val
        self.ckpt_dir = ckpt_dir
        self.ckpt_filename = ckpt_filename
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        self.wandb_name = wandb_name
        self.wandb_config = wandb_config
        self.wandb_tags = wandb_tags

    def _build_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        collator,
    ) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
        g = torch.Generator()
        g.manual_seed(self.seed)

        # Train: manual DistributedSampler so we can keep val un-sharded.
        # Val dataset is designed for single-card traversal (variable-size chunks
        # per FOV), so we disable Lightning's auto distributed sampler and let
        # every rank run the full val set — torchmetrics still aggregate correctly
        # since each rank's local state equals the global state.
        if self.devices > 1:
            # Pass num_replicas/rank explicitly so this works before Lightning
            # calls init_process_group inside trainer.fit().
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.devices,
                rank=get_global_rank(),
                shuffle=True, seed=self.seed, drop_last=True,
            )
        else:
            train_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True,
            generator=g,
            worker_init_fn=seed_worker,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

        return train_loader, val_loader, train_sampler

    def fit(
        self,
        model: pl.LightningModule,
        train_dataset: Dataset,
        val_dataset: Dataset,
        collator,
        callbacks: Optional[list[Callback]] = None,
    ) -> ModelCheckpoint:
        """Run the full training loop.

        Args:
            model: The LightningModule to train.
            train_dataset: Must have a .set_epoch() method (TrainFinetuneDataset).
            val_dataset: Validation dataset.
            collator: DataCollator instance.
            callbacks: Task-specific callbacks (e.g. MetricsRecorder).
                       The caller holds references — best metrics are readable after fit().

        Returns:
            The ModelCheckpoint callback (use .best_model_path to get the best checkpoint).
        """
        pl.seed_everything(self.seed, workers=True)

        train_loader, val_loader, train_sampler = self._build_dataloaders(
            train_dataset, val_dataset, collator)

        epoch_setter = TrainEpochSetter(train_dataset, train_sampler=train_sampler)
        early_stop = EarlyStopping(
            monitor=self.monitor, patience=self.early_stop_patience,
            mode=self.monitor_mode, verbose=True)
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(self.ckpt_dir), filename=self.ckpt_filename,
            monitor=self.monitor, mode=self.monitor_mode,
            save_top_k=1, save_weights_only=True)

        if self.logger == "wandb":
            logger = WandbLogger(
                project=self.wandb_project, group=self.wandb_group,
                name=self.wandb_name, config=self.wandb_config, tags=self.wandb_tags)
        elif self.logger == "csv":
            logger = CSVLogger(str(self.ckpt_dir), name="logs")
            logger.log_hyperparams(self.wandb_config)
        else:
            raise ValueError(f"Unknown logger: {self.logger}")

        strategy = DDPStrategy(find_unused_parameters=True) if self.devices > 1 else "auto"

        all_callbacks = [epoch_setter, early_stop, checkpoint_cb]
        if callbacks:
            all_callbacks.extend(callbacks)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=all_callbacks,
            logger=logger,
            devices=self.devices,
            accelerator=self.accelerator,
            strategy=strategy,
            precision=self.precision,
            accumulate_grad_batches=self.accumulate_grad_batches,
            gradient_clip_val=self.gradient_clip_val,
            log_every_n_steps=10,
            enable_checkpointing=True,
            # Disable Lightning's auto DistributedSampler wrapping.
            # Train loader is already manually wrapped above; val loader must
            # stay un-sharded so each rank traverses the full val set.
            use_distributed_sampler=False,
        )

        trainer.fit(model, train_loader, val_loader)
        if self.logger == "wandb":
            wandb.finish()

        return checkpoint_cb
