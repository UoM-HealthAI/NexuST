import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.utils.data import DataLoader
from pathlib import Path
import torch.distributed as dist
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy
from functools import partial


class GlooObjectBroadcastFSDPStrategy(FSDPStrategy):
    """Use a CPU (gloo) process group to broadcast Python objects (e.g., log_dir strings).
    This avoids ROCm/NCCL broadcast_object_list issues at large world sizes.
    """
    def setup_environment(self) -> None:
        super().setup_environment()
        # Create a dedicated gloo group for object collectives
        if dist.is_available() and dist.is_initialized():
            self._gloo_group = dist.new_group(backend="gloo")
        else:
            self._gloo_group = None

    def broadcast(self, obj, src: int = 0):
        if getattr(self, "_gloo_group", None) is None:
            return obj
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=src, group=self._gloo_group)
        return obj_list[0]

from configs.pretrain_config import Config
from data.dataset import NexuSTDataset
from data.collator import DataCollator
from data.tokenizer import get_tokenizer
from engines.train_module import NexuSTPreTrain
from model.nexust import NexuSTEncoderLayer
            

class PretrainTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.run_name = self.config.logging.run_name
        self.checkpoint_dir = Path(self.config.logging.save_dir) / self.run_name

        pl.seed_everything(self.config.seed, workers=True)
        self.setup_datasets()
        self.setup_model()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.setup_trainer()

        self.config.to_yaml(str(self.checkpoint_dir / "pretrain_config.yaml"))

    def setup_datasets(self):
        tokenizer = get_tokenizer(self.config.tokenizer.vocab_file, self.config.tokenizer.metadata_vocab_file)
        self.config.model.gene_vocab_size = tokenizer.gene_vocab_size
        self.config.model.organ_vocab_size = tokenizer.organ_vocab_size
        self.config.model.batch_vocab_size = tokenizer.platform_vocab_size
        self.config.model.padding_idx = tokenizer.pad_token_id

        ds_cfg = self.config.dataset
        train_root = Path(ds_cfg.train_dir)
        val_root = Path(ds_cfg.val_dir)

        # Load from precomputed patch_index.pt
        train_index = train_root / 'patch_index.pt'
        val_index = val_root / 'patch_index.pt'

        self.train_dataset = NexuSTDataset(
            patch_index_path=str(train_index),
            tokenizer=tokenizer,
        )
        self.val_dataset = NexuSTDataset(
            patch_index_path=str(val_index),
            tokenizer=tokenizer,
        )
        train_collator = DataCollator(
            max_gene_len=self.config.dataset.max_gene_len,
            pad_token_id=0,
            pad_value=0.0,
            gene_sampling=self.config.dataset.gene_sampling,
        )
        val_collator = DataCollator(
            max_gene_len=self.config.dataset.max_gene_len,
            pad_token_id=0,
            pad_value=0.0,
            gene_sampling=True,
            gene_sampling_seed=42,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,  
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
            collate_fn=train_collator
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
            collate_fn=val_collator
        )
        
    def setup_model(self):
        self.model = NexuSTPreTrain(self.config)
        n_all = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model Params] total={n_all:,} trainable={n_train:,}")

    def setup_trainer(self):
        callbacks = [
            ModelCheckpoint(
                dirpath=self.checkpoint_dir,
                filename='best-{step:06d}-{val_loss:.4f}',
                monitor='val_loss',
                mode='min',
                save_top_k=self.config.logging.save_top_k,
                save_last=True
            ),
            ModelCheckpoint(
                dirpath=self.checkpoint_dir,
                filename='step_{step:06d}',
                every_n_train_steps=self.config.logging.save_every_n_steps,
                save_top_k=-1,
            )
        ]

        if self.config.logging.logger == "wandb":
            logger = WandbLogger(project=self.config.logging.project_name, name=self.run_name)
        elif self.config.logging.logger == "csv":
            logger = CSVLogger(str(self.checkpoint_dir), name="logs")
        else:
            raise ValueError(f"Unknown logger: {self.config.logging.logger}")

        strategy = self.config.compute.strategy
        if strategy == 'fsdp':
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={NexuSTEncoderLayer},
            )
            strategy = GlooObjectBroadcastFSDPStrategy(
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=getattr(ShardingStrategy, self.config.compute.fsdp_sharding_strategy),
                cpu_offload=False,
                activation_checkpointing_policy={NexuSTEncoderLayer} if self.config.compute.fsdp_activation_checkpointing else None,
                use_orig_params=True,
            )

        self.trainer = pl.Trainer(
            max_steps=self.config.training.max_steps,
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=None,  # Manual clipping in LightningModule for FSDP compatibility
            accumulate_grad_batches=self.config.training.accumulate_grad_batches,
            val_check_interval=self.config.logging.val_check_interval,
            log_every_n_steps=self.config.logging.log_every_n_steps,
            precision=self.config.compute.precision,
            devices=self.config.compute.devices,
            num_nodes=self.config.compute.num_nodes,
            accelerator=self.config.compute.accelerator,
            strategy=strategy,
            default_root_dir=str(self.checkpoint_dir),
            profiler=None,
        )

    def train(self):
        ckpt_path = self.config.resume_from_checkpoint
        self.trainer.fit(self.model, self.train_loader, self.val_loader, ckpt_path=ckpt_path)
        print(f"Training completed. Best checkpoint: {self.trainer.checkpoint_callback.best_model_path}")
