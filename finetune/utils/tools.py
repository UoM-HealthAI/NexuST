import torch
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr
from typing import List
from pathlib import Path
import random
from torchmetrics import Metric
from pytorch_lightning.callbacks import Callback
from model.nexust import NexuSTEncoder, NexuST
from model.layers import BatchEncoder


class GeneWisePearsonCorrCoef(Metric):
    def __init__(self, hvg_gene_ids: List[int], **kwargs):
        super().__init__(**kwargs)
        self.hvg_gene_ids = hvg_gene_ids
        n = len(hvg_gene_ids)

        max_id = max(hvg_gene_ids) + 1
        lookup = torch.full((max_id,), -1, dtype=torch.long)
        for i, g in enumerate(hvg_gene_ids):
            lookup[g] = i
        self.register_buffer('lookup', lookup)

        # Accumulated statistics (all support DDP reduce)
        self.add_state("sp", default=torch.zeros(n), dist_reduce_fx="sum")    # Σp
        self.add_state("st", default=torch.zeros(n), dist_reduce_fx="sum")    # Σt
        self.add_state("sp2", default=torch.zeros(n), dist_reduce_fx="sum")   # Σp²
        self.add_state("st2", default=torch.zeros(n), dist_reduce_fx="sum")   # Σt²
        self.add_state("spt", default=torch.zeros(n), dist_reduce_fx="sum")   # Σpt
        self.add_state("cnt", default=torch.zeros(n), dist_reduce_fx="sum")   # n

    def update(self, gene_ids: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            gene_ids: [N] gene IDs for masked positions
            preds: [N] predicted values
            targets: [N] target values
        """
        idx = self.lookup[gene_ids]
        # Convert to float32 for accumulation (bf16-mixed training may pass bf16 tensors)
        preds = preds.float()
        targets = targets.float()

        self.sp.scatter_add_(0, idx, preds)
        self.st.scatter_add_(0, idx, targets)
        self.sp2.scatter_add_(0, idx, preds ** 2)
        self.st2.scatter_add_(0, idx, targets ** 2)
        self.spt.scatter_add_(0, idx, preds * targets)
        self.cnt.scatter_add_(0, idx, torch.ones_like(preds))

    def compute(self):
        valid = self.cnt >= 2
        if not torch.any(valid):
            return self.sp.new_tensor(0.0)

        n = self.cnt[valid]
        sp = self.sp[valid]; st = self.st[valid]
        sp2 = self.sp2[valid]; st2 = self.st2[valid]; spt = self.spt[valid]

        mp = sp / n
        mt = st / n
        cov = spt / n - mp * mt
        vp = sp2 / n - mp * mp
        vt = st2 / n - mt * mt

        eps = 1e-8
        good = (vp > eps) & (vt > eps)
        if not torch.any(good):
            return self.sp.new_tensor(0.0)

        pcc = cov[good] / (torch.sqrt(vp[good] * vt[good]) + eps)
        return pcc.mean().to(torch.float32)


class GeneWiseSpearmanCorrCoef(Metric):
    """Gene-wise Spearman correlation, DDP-compatible.

    Unlike Pearson, Spearman requires ranking which cannot be computed incrementally.
    We collect all (gene_id, pred, target) tuples and compute at the end.
    """
    def __init__(self, hvg_gene_ids: List[int], **kwargs):
        super().__init__(**kwargs)
        self.hvg_gene_ids = set(hvg_gene_ids)

        # Use list states for collecting data (DDP will concatenate across ranks)
        self.add_state("gene_ids_list", default=[], dist_reduce_fx="cat")
        self.add_state("preds_list", default=[], dist_reduce_fx="cat")
        self.add_state("targets_list", default=[], dist_reduce_fx="cat")

    def update(self, gene_ids: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            gene_ids: [N] gene IDs for masked positions
            preds: [N] predicted values
            targets: [N] target values
        """
        self.gene_ids_list.append(gene_ids.detach())
        self.preds_list.append(preds.detach().float())
        self.targets_list.append(targets.detach().float())

    def compute(self):
        if len(self.gene_ids_list) == 0:
            return torch.tensor(0.0)

        # After DDP sync, dist_reduce_fx="cat" converts list -> tensor
        # Use try/except for robustness
        try:
            gene_ids = torch.cat(self.gene_ids_list)
            preds = torch.cat(self.preds_list)
            targets = torch.cat(self.targets_list)
        except TypeError:
            # Already concatenated by DDP sync
            gene_ids = self.gene_ids_list
            preds = self.preds_list
            targets = self.targets_list

        # Move to CPU for scipy
        gene_ids_np = gene_ids.cpu().numpy()
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Compute per-gene Spearman
        spearman_scores = []
        for gid in self.hvg_gene_ids:
            mask = gene_ids_np == gid
            if mask.sum() >= 2:
                p = preds_np[mask]
                t = targets_np[mask]
                rho, _ = spearmanr(p, t)
                if not np.isnan(rho):
                    spearman_scores.append(rho)

        # Return tensor on same device as input for DDP compatibility
        device = gene_ids.device if isinstance(gene_ids, torch.Tensor) else 'cpu'
        if spearman_scores:
            return torch.tensor(np.mean(spearman_scores), dtype=torch.float32, device=device)
        return torch.tensor(0.0, dtype=torch.float32, device=device)


class ColumnWisePearsonCorrCoef(Metric):
    """Column-wise (per-feature) Pearson correlation, DDP-compatible.

    For niche prediction: computes PCC for each cell type across all samples.
    """
    def __init__(self, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.add_state("sum_pred", default=torch.zeros(n_features), dist_reduce_fx="sum")
        self.add_state("sum_target", default=torch.zeros(n_features), dist_reduce_fx="sum")
        self.add_state("sum_pred_sq", default=torch.zeros(n_features), dist_reduce_fx="sum")
        self.add_state("sum_target_sq", default=torch.zeros(n_features), dist_reduce_fx="sum")
        self.add_state("sum_cross", default=torch.zeros(n_features), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(n_features), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target: [N, n_features]
        self.sum_pred += preds.sum(dim=0)
        self.sum_target += target.sum(dim=0)
        self.sum_pred_sq += (preds ** 2).sum(dim=0)
        self.sum_target_sq += (target ** 2).sum(dim=0)
        self.sum_cross += (preds * target).sum(dim=0)
        self.count += preds.size(0)

    def compute(self):
        n = self.count.clamp_min(1)
        mean_pred = self.sum_pred / n
        mean_target = self.sum_target / n

        var_pred = (self.sum_pred_sq / n) - (mean_pred ** 2)
        var_target = (self.sum_target_sq / n) - (mean_target ** 2)
        cov = (self.sum_cross / n) - (mean_pred * mean_target)

        std_pred = var_pred.clamp_min(1e-8).sqrt()
        std_target = var_target.clamp_min(1e-8).sqrt()
        pearson = cov / (std_pred * std_target + 1e-8)

        valid = (var_pred > 1e-8) & (var_target > 1e-8)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pearson.device)

        return pearson[valid].mean()


class ColumnWiseMAE(Metric):
    """Column-wise (per-feature) MAE, DDP-compatible.

    For niche prediction: computes MAE for each cell type across all samples.
    """
    def __init__(self, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.add_state("sum_abs_error", default=torch.zeros(n_features), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(n_features), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target: [N, n_features]
        abs_error = (preds.float() - target.float()).abs()
        self.sum_abs_error += abs_error.sum(dim=0)
        self.count += preds.size(0)

    def compute(self):
        valid = self.count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=self.sum_abs_error.device)
        mae = self.sum_abs_error[valid] / self.count[valid]
        return mae.mean()


class TrainEpochSetter(Callback):
    """Keep dataset sampling and DistributedSampler in sync with the current epoch."""

    def __init__(self, train_dataset, train_sampler=None):
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_dataset.set_epoch(trainer.current_epoch)
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(trainer.current_epoch)


def get_model_config(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["hyper_parameters"]["config"]["model"]


def load_nexust(ckpt_path: str, device: str = "cpu") -> NexuST:
    """Load complete NexuST model (encoder + MoE decoder + library decoder)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt["hyper_parameters"]["config"]["model"]

    model = NexuST(
        gene_vocab_size=model_cfg.gene_vocab_size,
        organ_vocab_size=model_cfg.organ_vocab_size,
        n_genes=model_cfg.n_genes,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        dropout=model_cfg.dropout,
        num_experts=getattr(model_cfg, 'num_experts', 8),
        batch_vocab_size=getattr(model_cfg, 'batch_vocab_size', 16),
        use_library=getattr(model_cfg, 'use_library', True),
    )

    state_dict = {
        k.replace("model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    return model


def load_encoder(ckpt_path: str, device: str = "cpu", use_activation_checkpointing: bool = False) -> NexuSTEncoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt["hyper_parameters"]["config"]["model"]

    encoder = NexuSTEncoder(
        gene_vocab_size=model_cfg.gene_vocab_size,
        organ_vocab_size=model_cfg.organ_vocab_size,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        dropout=model_cfg.dropout,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    encoder_state = {
        k.replace("model.NexuSTEncoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if "NexuSTEncoder" in k
    }
    encoder.load_state_dict(encoder_state)

    return encoder

def load_batch_encoder(ckpt_path: str, device: str = "cpu") -> BatchEncoder:
    """Load BatchEncoder from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt["hyper_parameters"]["config"]["model"]

    batch_vocab_size = getattr(model_cfg, 'batch_vocab_size', 4)
    d_model = model_cfg.d_model

    batch_encoder = BatchEncoder(batch_vocab_size, d_model, padding_idx=0)

    # state_dict key: "model.batch_encoder.weight" -> "weight"
    batch_encoder_state = {
        k.replace("model.batch_encoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if "batch_encoder" in k
    }
    batch_encoder.load_state_dict(batch_encoder_state)

    return batch_encoder


def seed_worker(worker_id: int):
    """Ensure each DataLoader worker has a unique but deterministic seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
