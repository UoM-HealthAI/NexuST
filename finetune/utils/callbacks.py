"""Base metrics recorder for finetune tasks."""

import json
from pathlib import Path

from pytorch_lightning.callbacks import Callback


class BaseMetricsRecorder(Callback):
    """Track best metrics during training and persist to JSON (DDP-spawn safe).

    Subclasses override ``_extract_metrics`` to pull task-specific values from
    ``trainer.callback_metrics`` or ``pl_module``.

    Usage:
        class MyRecorder(BaseMetricsRecorder):
            def __init__(self, save_dir=None):
                super().__init__(primary_metric='acc', mode='max', save_dir=save_dir)

            def _extract_metrics(self, trainer, pl_module):
                acc = trainer.callback_metrics.get('val_acc')
                if acc is None:
                    return None
                return {'acc': acc.item()}
    """

    def __init__(self, primary_metric: str, mode: str = "max",
                 save_dir: Path | None = None):
        self.primary_metric = primary_metric
        self.mode = mode
        self.save_dir = save_dir
        self.best_primary = float('-inf') if mode == "max" else float('inf')
        self.best_epoch = -1
        self.best_metrics: dict = {}

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_primary
        return value < self.best_primary

    def _extract_metrics(self, trainer, pl_module) -> dict | None:
        """Return dict of metrics to save, or None to skip this epoch.

        Must include the ``primary_metric`` key.
        """
        raise NotImplementedError

    def on_validation_end(self, trainer, pl_module):
        # Must be ``on_validation_end`` (not ``..._epoch_end``): the logger
        # connector only merges the current epoch's ``self.log(...)`` values
        # into ``trainer.callback_metrics`` *after* the module's epoch-end
        # hook returns, so ``val_acc`` read in ``on_validation_epoch_end`` is
        # the previous epoch's value — it would get paired with the current
        # epoch's preds in ``_extract_metrics`` and mismatch.
        metrics = self._extract_metrics(trainer, pl_module)
        if metrics is None:
            return
        primary = metrics.get(self.primary_metric)
        if primary is not None and self._is_better(primary):
            self.best_primary = primary
            self.best_epoch = trainer.current_epoch
            self.best_metrics = {**metrics, 'epoch': self.best_epoch}
            # All ranks update in-memory state (so read_results() works on any
            # rank after fit), but only rank 0 writes to disk to avoid a race.
            if trainer.is_global_zero:
                self._save_to_disk()

    def _save_to_disk(self):
        if self.save_dir is None:
            return
        payload = {}
        for k, v in self.best_metrics.items():
            if hasattr(v, 'tolist'):
                payload[k] = v.tolist()
            else:
                payload[k] = v
        path = self.save_dir / 'best_metrics.json'
        path.write_text(json.dumps(payload, indent=2))

    def read_results(self) -> dict:
        """DDP-spawn safe: falls back to JSON on disk if callback state is lost."""
        if self.best_epoch >= 0:
            return self.best_metrics
        if self.save_dir is not None:
            json_path = self.save_dir / 'best_metrics.json'
            if json_path.exists():
                return json.loads(json_path.read_text())
        return {}
