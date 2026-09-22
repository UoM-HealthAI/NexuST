import argparse
import os
from pathlib import Path

import numpy as np


from probe.config import TaskConfig
from evaluation.niche import extract_niche_labels, center_grouped_mae
from probe.heads import run_niche_probe
from evaluation.io import append_csv_row, needs_header
from probe.tasks.base import BaseProbeTask


class NichePredictionTask(BaseProbeTask):
    task_name = "niche"
    csv_columns = ["dataset", "mode", "radius_idx", "radius", "seed", "mse", "mae", "pcc"]
    metric_keys = ["mse", "mae", "pcc"]

    def get_radii(self, cfg):
        radii = cfg.dataset_spec.niche_radii
        if not radii:
            raise ValueError("Set dataset.niche_radii in the probe config")
        return radii

    def get_train_cfg(self, cfg):
        return cfg.niche

    def load_labels(self, adatas, metadata):
        niche_key = f"X_niche_{metadata['radius_idx']}"
        return extract_niche_labels(adatas, niche_key)

    def run_probe(self, train_emb, train_lab, val_emb, val_lab,
                  d_model, seed, train_cfg, cfg):
        result, per_col_mae, per_col_pcc, val_preds = run_niche_probe(
            train_emb, train_lab, val_emb, val_lab,
            d_model=d_model,
            n_cell_types=train_lab.shape[1],
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
        return result, {
            "per_col_mae": per_col_mae,
            "per_col_pcc": per_col_pcc,
            "val_preds": val_preds,
        }

    def format_row(self, metadata, seed, result):
        return [metadata["dataset_name"], "probe",
                metadata["radius_idx"], f"{metadata['actual_radius']:.4f}",
                seed, f"{result['mse']:.6f}", f"{result['mae']:.6f}",
                f"{result['pcc']:.4f}"]

    def collect_val_aux(self, val_adatas, metadata, cfg):
        """Stash per-val-cell coords (val order) for the per-cell prediction dump.

        Runs only under NICHE_DUMP_PREDS. Always captures obsm[cfg.spatial_key]
        (cfg.spatial_key defaults to "spatial", configs/__init__.py) — every
        platform has it. Absolute slide coords x_slide_mm / y_slide_mm are
        CosMx-only (a stitchable slide frame feeding the whole-slide advantage
        map); Xenium/MERFISH have no slide frame, so they are captured only when
        present and the dump otherwise proceeds without them.
        """
        if not os.environ.get("NICHE_DUMP_PREDS"):
            return
        metadata["_val_spatial"] = np.concatenate(
            [np.asarray(a.obsm[cfg.spatial_key], dtype=float) for a in val_adatas], axis=0)
        if all("x_slide_mm" in a.obs and "y_slide_mm" in a.obs for a in val_adatas):
            metadata["_val_slide_xy"] = np.concatenate(
                [np.column_stack([a.obs["x_slide_mm"].to_numpy(float),
                                  a.obs["y_slide_mm"].to_numpy(float)])
                 for a in val_adatas], axis=0)

    def _dump_val_preds(self, seed, val_preds, metadata):
        """Save per-cell val predictions + alignment metadata as one .npz.

        Opt-in via NICHE_DUMP_PREDS; writes under
        output/niche_prediction/predictions/<model>/<dataset>_r<idx>_seed<seed>.npz so the
        spatial advantage map (plot_niche_advantage_slide) can load NexuST and
        PCA dumps and align them cell-for-cell (identical sorted val order).
        """
        if not os.environ.get("NICHE_DUMP_PREDS"):
            return
        out = Path(metadata["output_dir"]) / "predictions"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{metadata['dataset_name']}_r{metadata['radius_idx']}_seed{seed}.npz"
        arrays = dict(
            val_preds=np.asarray(val_preds, np.float32),
            val_lab=np.asarray(metadata["_val_lab"], np.float32),
            center_types=np.asarray(metadata["center_cell_types"]).astype(str),
            cell_type_names=np.asarray(list(metadata.get("cell_type_names", []))).astype(str),
            spatial=np.asarray(metadata["_val_spatial"], np.float32),
        )
        # slide_xy present only for CosMx (see collect_val_aux); omitted otherwise.
        if "_val_slide_xy" in metadata:
            arrays["slide_xy"] = np.asarray(metadata["_val_slide_xy"], np.float32)
        np.savez_compressed(path, **arrays)
        print(f"  dumped per-cell preds -> {path}")

    def on_seed_done(self, seed, extra, output_dir, metadata):
        dataset_name = metadata["dataset_name"]
        radius_idx = metadata["radius_idx"]
        actual_radius = metadata["actual_radius"]
        cell_type_names = metadata.get("cell_type_names", [])
        ct_columns = ["dataset", "mode", "radius_idx", "radius", "seed"] + cell_type_names
        base_row = [dataset_name, "probe", radius_idx, f"{actual_radius:.4f}", seed]

        # Target-dimension per-column MAE & PCC
        csv_mae = output_dir / f"{dataset_name}_celltype_mae.csv"
        row = base_row + [f"{v:.6f}" if not np.isnan(v) else "nan" for v in extra["per_col_mae"]]
        append_csv_row(csv_mae, ct_columns, row, write_header=needs_header(csv_mae))

        csv_pcc = output_dir / f"{dataset_name}_celltype_pcc.csv"
        row = base_row + [f"{v:.4f}" if not np.isnan(v) else "nan" for v in extra["per_col_pcc"]]
        append_csv_row(csv_pcc, ct_columns, row, write_header=needs_header(csv_pcc))

        # Optional: dump per-cell val predictions for the spatial advantage map
        self._dump_val_preds(seed, extra["val_preds"], metadata)

        # Center-celltype-grouped MAE — compute once, store for on_all_seeds_done
        center_types = metadata["center_cell_types"]
        val_lab = metadata["_val_lab"]
        grouped = center_grouped_mae(extra["val_preds"], val_lab, center_types)
        center_names = sorted(grouped.keys())
        extra["center_mae"] = np.array([grouped[ct]["mae"] for ct in center_names])
        extra["center_target_mae"] = {ct: grouped[ct]["mae_per_target"] for ct in center_names}
        extra["center_names"] = center_names

        center_columns = ["dataset", "mode", "radius_idx", "radius", "seed"] + center_names
        csv_center_mae = output_dir / f"{dataset_name}_center_mae.csv"
        row = base_row + [f"{v:.6f}" for v in extra["center_mae"]]
        append_csv_row(csv_center_mae, center_columns, row,
                       write_header=needs_header(csv_center_mae))

    def on_all_seeds_done(self, extras, output_dir, metadata):
        dataset_name = metadata["dataset_name"]
        radius_idx = metadata["radius_idx"]
        actual_radius = metadata["actual_radius"]
        cell_type_names = metadata.get("cell_type_names", [])
        ct_columns = ["dataset", "mode", "radius_idx", "radius", "seed"] + cell_type_names

        center_names = extras[0]["center_names"]
        center_columns = ["dataset", "mode", "radius_idx", "radius", "seed"] + center_names

        for key, suffix, columns, fmt in [
            ("per_col_mae", "celltype_mae", ct_columns, ".6f"),
            ("per_col_pcc", "celltype_pcc", ct_columns, ".4f"),
            ("center_mae", "center_mae", center_columns, ".6f"),
        ]:
            csv_path = output_dir / f"{dataset_name}_{suffix}.csv"
            stacked = np.stack([e[key] for e in extras], axis=0)
            for agg_name, agg_fn in [("mean", np.nanmean), ("std", np.nanstd)]:
                agg_vals = agg_fn(stacked, axis=0)
                row = [dataset_name, "probe", radius_idx, f"{actual_radius:.4f}", agg_name]
                row += [f"{v:{fmt}}" if not np.isnan(v) else "nan" for v in agg_vals]
                append_csv_row(csv_path, columns, row)
