import argparse
from pathlib import Path

import numpy as np


from probe.config import TaskConfig
from probe.utils.density import compute_density
from probe.heads import run_regression_probe
from probe.tasks.base import BaseProbeTask


class DensityPredictionTask(BaseProbeTask):
    task_name = "density"
    csv_columns = ["dataset", "mode", "radius_idx", "radius", "seed", "mse", "mae", "r2"]
    metric_keys = ["mse", "mae", "r2"]

    def get_radii(self, cfg):
        radii = cfg.dataset_spec.density_radii
        if not radii:
            raise ValueError("Set dataset.density_radii in the probe config")
        return radii

    def get_train_cfg(self, cfg):
        return cfg.density

    def load_labels(self, adatas, metadata):
        actual_radius = metadata["actual_radius"]
        labels_list = []
        for adata in adatas:
            coords = adata.obsm[metadata["spatial_key"]]
            density = compute_density(coords, radius=actual_radius)
            labels_list.append(density)
        return np.concatenate(labels_list, axis=0), {}

    def run_probe(self, train_emb, train_lab, val_emb, val_lab,
                  d_model, seed, train_cfg, cfg):
        result, best_epoch = run_regression_probe(
            train_emb, train_lab, val_emb, val_lab,
            d_model=d_model,
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
        return result, None

    def format_row(self, metadata, seed, result):
        return [metadata["dataset_name"], "probe",
                metadata["radius_idx"], f"{metadata['actual_radius']:.4f}",
                seed, f"{result['mse']:.6f}", f"{result['mae']:.6f}",
                f"{result['r2']:.4f}"]
