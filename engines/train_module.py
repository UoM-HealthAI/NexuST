import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import math
from typing import Dict

from model.nexust import NexuST
from configs.pretrain_config import Config


class NexuSTPreTrain(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters({"config": config.__dict__})

        self.model = NexuST(
            gene_vocab_size=config.model.gene_vocab_size,
            organ_vocab_size=config.model.organ_vocab_size,
            n_genes=config.model.n_genes,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            dropout=config.model.dropout,
            num_experts=config.model.num_experts,
            batch_vocab_size=config.model.batch_vocab_size,
            use_library=config.model.use_library,
        )

        self.total_params_raw = sum(p.numel() for p in self.model.parameters())
        self.trainable_params_raw = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.gene_mask_ratio = config.model.gene_mask_ratio
        self.grad_clip_norm = config.training.grad_clip_norm

    @torch.no_grad()
    def _compute_library_r2(self, library: torch.Tensor, gene_values: torch.Tensor,
                            valid_gene: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute R² between predicted library and true depth with affine calibration.

        Args:
            library: [B, N] predicted library size
            gene_values: [B, N, G] log1p gene expression values
            valid_gene: [B, N, G] valid gene mask (gene_ids != 0)

        Returns:
            R² score
        """
        # target subset depth in linear space
        target_L = (torch.expm1(gene_values).clamp_min(0.0) * valid_gene).sum(dim=-1).clamp_min(0.0)  # [B,N]

        x = torch.log1p(library.clamp_min(0.0)).flatten()
        y = torch.log1p(target_L).flatten()

        # affine calibration y ≈ a x + b
        x_mean, y_mean = x.mean(), y.mean()
        x_c, y_c = x - x_mean, y - y_mean
        a = (x_c * y_c).sum() / (x_c.pow(2).sum() + eps)
        b = y_mean - a * x_mean
        y_hat = a * x + b

        ss_res = (y - y_hat).pow(2).sum()
        ss_tot = y_c.pow(2).sum().clamp_min(eps)
        return 1.0 - ss_res / ss_tot
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        gene_ids = batch['gene_ids']
        gene_values = batch['gene_values']

        out = self.model(
            gene_ids,
            gene_values,
            batch['coords'],
            organ_ids=batch.get('organ_ids', None),
            gene_mask_ratio=self.gene_mask_ratio,
            deterministic=False,
            batch_labels=batch.get('batch_labels', None),
        )

        target = gene_values
        mask = out['mask']

        loss = F.mse_loss(out['cell_expr_pred'][mask], target[mask])

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        # Monitor library vs true depth R² (only when use_library=True)
        if self.config.model.use_library:
            valid = (gene_ids != 0)
            library_r2 = self._compute_library_r2(out['library'], target, valid)
            self.log('train_library_r2', library_r2, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    @rank_zero_only
    def on_fit_start(self) -> None:
        print(
            f"[NexuST] params(raw): total={self.total_params_raw:,} "
            f"trainable={self.trainable_params_raw:,}"
        )

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        gene_ids = batch['gene_ids']
        gene_values = batch['gene_values']

        out = self.model(
            gene_ids,
            gene_values,
            batch['coords'],
            organ_ids=batch.get('organ_ids', None),
            gene_mask_ratio=self.gene_mask_ratio,
            deterministic=True,
            batch_labels=batch.get('batch_labels', None),
        )

        target = gene_values
        mask = out['mask']

        valid = (gene_ids != 0)
        m = mask & valid

        loss = F.mse_loss(out['cell_expr_pred'][m], target[m])

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Monitor library vs true depth R² (only when use_library=True)
        if self.config.model.use_library:
            library_r2 = self._compute_library_r2(out['library'], target, valid)
            self.log('val_library_r2', library_r2, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_eps,
            weight_decay=self.config.training.weight_decay
        )

        total_steps = self.config.training.max_steps
        warmup_steps = self.config.training.warmup_steps
        min_lr_ratio = self.config.training.min_lr / self.config.training.learning_rate
        print(f"LR Schedule: warmup {warmup_steps} steps, total {total_steps} steps")

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def on_before_optimizer_step(self, optimizer):
        """Manual gradient clipping for FSDP compatibility and grad norm logging."""
        if self.grad_clip_norm is None or self.grad_clip_norm <= 0:
            return

        model = self.trainer.strategy.model
        if isinstance(model, FSDP):
            grad_norm = model.clip_grad_norm_(max_norm=self.grad_clip_norm, norm_type=2.0)
        else:
            params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)

        self.log('grad_norm', grad_norm, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

    def on_train_epoch_start(self):
        train_loader = self.trainer.train_dataloader
        dataset = getattr(train_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    def on_train_epoch_end(self):
        """Log MoE metrics at the end of training epoch."""
        if hasattr(self.model.cell_decoder, 'get_moe_metrics'):
            metrics = self.model.cell_decoder.get_moe_metrics()
            if metrics is not None:
                for i, usage in enumerate(metrics['expert_usage']):
                    self.log(f'train_moe/expert_{i}', usage, on_epoch=True, sync_dist=True)
                self.log('train_moe/load_balance', metrics['load_balance'], on_epoch=True, sync_dist=True)
                self.log('train_moe/entropy', metrics['routing_entropy'], on_epoch=True, sync_dist=True)
                self.log('train_moe/dead_experts', metrics['dead_expert_count'], on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """Log MoE metrics at the end of validation epoch."""
        if hasattr(self.model.cell_decoder, 'get_moe_metrics'):
            metrics = self.model.cell_decoder.get_moe_metrics()
            if metrics is not None:
                for i, usage in enumerate(metrics['expert_usage']):
                    self.log(f'moe/expert_{i}', usage, on_epoch=True, sync_dist=True)
                self.log('moe/load_balance', metrics['load_balance'], on_epoch=True, sync_dist=True)
                self.log('moe/entropy', metrics['routing_entropy'], on_epoch=True, sync_dist=True)
                self.log('moe/dead_experts', metrics['dead_expert_count'], on_epoch=True, sync_dist=True)
