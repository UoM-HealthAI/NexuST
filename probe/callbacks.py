"""Metrics recorder callbacks for linear probing tasks."""

from typing import Dict

from pytorch_lightning.callbacks import Callback


class RegressionMetricsRecorder(Callback):
    def __init__(self):
        self.best_mse = float("inf")
        self.best_mae = float("inf")
        self.best_r2 = -float("inf")
        self.best_epoch = -1

    def on_validation_end(self, trainer, pl_module):
        mse = trainer.callback_metrics.get("val_mse")
        if mse is not None and mse.item() < self.best_mse:
            self.best_mse = mse.item()
            mae = trainer.callback_metrics.get("val_mae")
            r2 = trainer.callback_metrics.get("val_r2")
            self.best_mae = mae.item() if mae is not None else float("inf")
            self.best_r2 = r2.item() if r2 is not None else -float("inf")
            self.best_epoch = trainer.current_epoch

    @property
    def results(self) -> Dict[str, float]:
        return {"mse": self.best_mse, "mae": self.best_mae, "r2": self.best_r2}


class ClassificationMetricsRecorder(Callback):
    def __init__(self):
        self.best_acc = -float("inf")
        self.best_f1 = -float("inf")
        self.best_epoch = -1
        self.best_state = None  # state_dict snapshot at best_epoch (in-memory)

    def on_validation_end(self, trainer, pl_module):
        f1 = trainer.callback_metrics.get("val_f1")
        if f1 is not None and f1.item() > self.best_f1:
            self.best_f1 = f1.item()
            acc = trainer.callback_metrics.get("val_acc")
            self.best_acc = acc.item() if acc is not None else -float("inf")
            self.best_epoch = trainer.current_epoch
            self.best_state = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}

    @property
    def results(self) -> Dict[str, float]:
        return {"accuracy": self.best_acc, "f1": self.best_f1}


class NicheMetricsRecorder(Callback):
    def __init__(self):
        self.best_mse = float("inf")
        self.best_mae = float("inf")
        self.best_pcc = -float("inf")
        self.best_epoch = -1
        self.best_state = None
        self.best_per_col_mae = None  # np.ndarray [n_cell_types]
        self.best_per_col_pcc = None  # np.ndarray [n_cell_types]

    def on_validation_end(self, trainer, pl_module):
        mse = trainer.callback_metrics.get("val_mse")
        if mse is not None and mse.item() < self.best_mse:
            self.best_mse = mse.item()
            mae = trainer.callback_metrics.get("val_mae")
            pcc = trainer.callback_metrics.get("val_pcc")
            self.best_mae = mae.item() if mae is not None else float("inf")
            self.best_pcc = pcc.item() if pcc is not None else -float("inf")
            self.best_epoch = trainer.current_epoch
            self.best_state = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}
            self.best_per_col_mae = pl_module.val_mae.compute_per_column().cpu().numpy()
            self.best_per_col_pcc = pl_module.val_pcc.compute_per_column().cpu().numpy()

    @property
    def results(self) -> Dict[str, float]:
        return {"mse": self.best_mse, "mae": self.best_mae, "pcc": self.best_pcc}
