"""Shared torchmetrics classes for NexuST downstream tasks.

All metrics are DDP-compatible (use dist_reduce_fx="sum").
"""

import torch
from torchmetrics import Metric


class GeneWisePearsonCorrCoef(Metric):
    """Gene-wise Pearson correlation for imputation, DDP-compatible."""

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.add_state("sp", default=torch.zeros(vocab_size), dist_reduce_fx="sum")
        self.add_state("st", default=torch.zeros(vocab_size), dist_reduce_fx="sum")
        self.add_state("sp2", default=torch.zeros(vocab_size), dist_reduce_fx="sum")
        self.add_state("st2", default=torch.zeros(vocab_size), dist_reduce_fx="sum")
        self.add_state("spt", default=torch.zeros(vocab_size), dist_reduce_fx="sum")
        self.add_state("cnt", default=torch.zeros(vocab_size), dist_reduce_fx="sum")

    def update(self, gene_ids: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            gene_ids: [N] gene token IDs for masked positions
            preds: [N] predicted values
            targets: [N] target values
        """
        preds = preds.float()
        targets = targets.float()

        self.sp.scatter_add_(0, gene_ids, preds)
        self.st.scatter_add_(0, gene_ids, targets)
        self.sp2.scatter_add_(0, gene_ids, preds ** 2)
        self.st2.scatter_add_(0, gene_ids, targets ** 2)
        self.spt.scatter_add_(0, gene_ids, preds * targets)
        self.cnt.scatter_add_(0, gene_ids, torch.ones_like(preds))

    def compute(self):
        valid = self.cnt >= 2
        if not torch.any(valid):
            return self.sp.new_tensor(0.0)

        n = self.cnt[valid]
        sp, st = self.sp[valid], self.st[valid]
        sp2, st2, spt = self.sp2[valid], self.st2[valid], self.spt[valid]

        mp, mt = sp / n, st / n
        vp = sp2 / n - mp * mp
        vt = st2 / n - mt * mt
        cov = spt / n - mp * mt

        eps = 1e-8
        good = (vp > eps) & (vt > eps)
        if not torch.any(good):
            return self.sp.new_tensor(0.0)

        pcc = cov[good] / (torch.sqrt(vp[good] * vt[good]) + eps)
        return pcc.mean().to(torch.float32)


class ColumnWisePearsonCorrCoef(Metric):
    """Column-wise (per-feature) Pearson correlation, DDP-compatible."""

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

    def compute_per_column(self) -> torch.Tensor:
        """Return per-column PCC as a 1D tensor [n_features]. Invalid columns get NaN."""
        n = self.count.clamp_min(1)
        mean_pred = self.sum_pred / n
        mean_target = self.sum_target / n

        var_pred = (self.sum_pred_sq / n) - (mean_pred ** 2)
        var_target = (self.sum_target_sq / n) - (mean_target ** 2)
        cov = (self.sum_cross / n) - (mean_pred * mean_target)

        std_pred = var_pred.clamp_min(1e-8).sqrt()
        std_target = var_target.clamp_min(1e-8).sqrt()
        pearson = cov / (std_pred * std_target + 1e-8)

        invalid = (var_pred <= 1e-8) | (var_target <= 1e-8)
        pearson[invalid] = float('nan')
        return pearson


class ColumnWiseMAE(Metric):
    """Column-wise (per-feature) MAE, DDP-compatible."""

    def __init__(self, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.add_state("sum_abs_error", default=torch.zeros(n_features), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(n_features), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        abs_error = (preds.float() - target.float()).abs()
        self.sum_abs_error += abs_error.sum(dim=0)
        self.count += preds.size(0)

    def compute(self):
        valid = self.count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=self.sum_abs_error.device)
        mae = self.sum_abs_error[valid] / self.count[valid]
        return mae.mean()

    def compute_per_column(self) -> torch.Tensor:
        """Return per-column MAE as a 1D tensor [n_features]. Empty columns get NaN."""
        mae = torch.full_like(self.sum_abs_error, float('nan'))
        valid = self.count > 0
        mae[valid] = self.sum_abs_error[valid] / self.count[valid]
        return mae
