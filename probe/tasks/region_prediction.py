"""Region / niche classification linear probe.

Classifies each cell into its tissue region / niche (e.g. tumor vs interface
for liver cancer, hepatic zones for normal liver, CNiche for lung). Mirrors
``cell_annotation.py`` but reads ``obs[region_col]`` instead of the cell-type
label, and infers ``n_classes`` from the fitted LabelEncoder so per-dataset
class counts do not need to be hardcoded.

Datasets without a region label (e.g. adult_umb5958) are not supported —
the task raises a clear error if ``region_col`` is unset for the dataset.

Seed 42 predictions are saved under output_dir/predictions/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


from probe.config import TaskConfig
from probe.heads import run_classification_probe
from probe.tasks.base import BaseProbeTask


class RegionClassificationTask(BaseProbeTask):
    task_name = "region_prediction"
    csv_columns = ["dataset", "mode", "seed", "accuracy", "f1"]
    metric_keys = ["accuracy", "f1"]

    def get_train_cfg(self, cfg):
        return cfg.region_prediction

    def load_labels(self, adatas, metadata):
        dataset_name = metadata["dataset_name"]
        ds = metadata["dataset_spec"]
        region_col = ds.region_col if ds else None
        if not region_col:
            raise KeyError(
                f"Dataset {dataset_name!r} has no region_col configured. "
                "Set dataset.region_col in the probe config."
            )

        labels = np.concatenate([ad.obs[region_col].astype(str).values for ad in adatas])

        if not hasattr(self, "_label_encoder"):
            self._label_encoder = LabelEncoder()
            self._label_encoder.fit(labels)
            metadata["class_names"] = list(self._label_encoder.classes_)
            metadata["n_classes"] = len(self._label_encoder.classes_)

        encoded = self._label_encoder.transform(labels).astype(np.int64)
        return encoded, metadata

    def run_probe(self, train_emb, train_lab, val_emb, val_lab,
                  d_model, seed, train_cfg, cfg):
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

        ds = metadata["dataset_spec"]
        payload = {
            "dataset": dataset_name,
            "region_col": ds.region_col if ds else None,
            "seed": seed,
            "best_epoch": extra["best_epoch"],
            "per_class": per_class,
            "confusion_matrix": cm,
        }

        json_path = output_dir / f"{dataset_name}_seed{seed}_metrics.json"
        json_path.write_text(json.dumps(payload, indent=2))

        # Also dump val_preds into the per_celltype analysis directory.
        # Only seed 42 is dumped — analysis uses a single representative seed.
        model_name = metadata.get("model_name")
        if model_name and seed == 42:
            preds_dir = output_dir / "predictions"
            preds_dir.mkdir(parents=True, exist_ok=True)
            np.save(preds_dir / f"{dataset_name}.npy",
                    val_preds.astype(np.int64))
