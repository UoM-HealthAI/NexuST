import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


from probe.config import TaskConfig
from probe.heads import run_classification_probe
from probe.tasks.base import BaseProbeTask


class ClassificationTask(BaseProbeTask):
    task_name = "classification"
    csv_columns = ["dataset", "mode", "seed", "accuracy", "f1"]
    metric_keys = ["accuracy", "f1"]

    def get_train_cfg(self, cfg):
        return cfg.classification

    def load_labels(self, adatas, metadata):
        dataset_name = metadata["dataset_name"]
        label_col = metadata["dataset_spec"].label_col
        if not label_col:
            raise ValueError("Set dataset.label_col in the probe config")

        labels = np.concatenate([ad.obs[label_col].values for ad in adatas])

        if not hasattr(self, "_label_encoder"):
            self._label_encoder = LabelEncoder()
            self._label_encoder.fit(labels)
            metadata["class_names"] = list(self._label_encoder.classes_)

        encoded = self._label_encoder.transform(labels).astype(np.int64)
        return encoded, metadata

    def run_probe(self, train_emb, train_lab, val_emb, val_lab,
                  d_model, seed, train_cfg, cfg):
        dataset_name = cfg.dataset_name or Path(cfg.data_path).stem
        n_classes = len(self._label_encoder.classes_)

        result, best_epoch, val_preds = run_classification_probe(
            train_emb, train_lab, val_emb, val_lab,
            d_model=d_model,
            n_classes=n_classes,
            seed=seed,
            batch_size=train_cfg.batch_size,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            max_epochs=train_cfg.max_epochs,
            num_workers=train_cfg.num_workers,
            devices=cfg.compute.devices,
            accelerator=cfg.compute.accelerator, precision=cfg.compute.precision,
            patience=train_cfg.patience,
        )
        return result, {"val_preds": val_preds, "best_epoch": best_epoch}

    def format_row(self, metadata, seed, result):
        return [metadata["dataset_name"], "probe", seed,
                f"{result['accuracy']:.6f}", f"{result['f1']:.6f}"]

    def on_seed_done(self, seed, extra, output_dir, metadata):
        dataset_name = metadata["dataset_name"]
        val_preds = extra["val_preds"]
        val_lab = metadata["_val_lab"]
        class_names = metadata.get("class_names", [])

        all_labels = list(range(len(class_names)))
        report = classification_report(
            val_lab, val_preds, labels=all_labels,
            target_names=class_names, output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(val_lab, val_preds).tolist()

        per_class = {
            name: {k: report[name][k] for k in ("precision", "recall", "f1-score", "support")}
            for name in class_names if name in report
        }

        payload = {
            "dataset": dataset_name,
            "seed": seed,
            "best_epoch": extra["best_epoch"],
            "per_class": per_class,
            "confusion_matrix": cm,
        }

        json_path = output_dir / f"{dataset_name}_seed{seed}_metrics.json"
        json_path.write_text(json.dumps(payload, indent=2))
