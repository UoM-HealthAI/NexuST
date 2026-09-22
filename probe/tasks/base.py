"""Base class for linear probe tasks."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


from probe.config import TaskConfig
from probe.utils.data import embed_h5ad_dir
from evaluation.io import append_csv_row, needs_header


class BaseProbeTask(ABC):
    """Template for: embed once -> load labels -> probe over seeds -> CSV.

    Subclasses must implement:
        task_name, csv_columns, metric_keys,
        get_train_cfg, load_labels, run_probe, format_row

    Optional overrides:
        get_radii, on_seed_done, on_all_seeds_done
    """

    @property
    @abstractmethod
    def task_name(self) -> str:
        ...

    @property
    @abstractmethod
    def csv_columns(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def metric_keys(self) -> List[str]:
        ...

    @abstractmethod
    def get_train_cfg(self, cfg: TaskConfig):
        ...

    @abstractmethod
    def load_labels(self, adatas: list, metadata: dict) -> Tuple[np.ndarray, dict]:
        """Extract labels from pre-loaded adatas.

        Args:
            adatas: List of AnnData objects.
            metadata: dict with dataset_name, radius_idx, actual_radius, etc.

        Returns:
            labels:   np.ndarray (N, ...) aligned with embeddings
            metadata: dict merged back into the shared metadata
        """
        ...

    @abstractmethod
    def run_probe(self, train_emb, train_lab, val_emb, val_lab,
                  d_model: int, seed: int, train_cfg, cfg) -> Tuple[Dict[str, float], Any]:
        """Run one seed. Return (result_dict, extra)."""
        ...

    @abstractmethod
    def format_row(self, metadata: dict, seed, result: Dict[str, float]) -> list:
        ...

    def get_radii(self, dataset_name: str) -> Optional[List[float]]:
        """Override for radius-based tasks. Return None if not applicable."""
        return None

    def on_seed_done(self, seed: int, extra: Any,
                     output_dir: Path, metadata: dict) -> None:
        pass

    def on_all_seeds_done(self, extras: list, output_dir: Path,
                          metadata: dict) -> None:
        pass

    def collect_val_aux(self, val_adatas: list, metadata: dict,
                        cfg: TaskConfig) -> None:
        """Optional: stash per-val-cell aux (coords, ids) into metadata.

        Called once per run() after labels are loaded, with the val adatas in
        the same order their cells appear in val_lab / val_preds. Tasks that
        dump per-cell predictions override this to capture alignment metadata.
        """
        pass

    def _embed(self, cfg: TaskConfig):
        """Build adapter, embed train/val, cleanup."""
        adapter = cfg.build_embedder()
        adapter.load_model()

        train_dir = Path(cfg.data_path) / "train"
        val_dir = Path(cfg.data_path) / "val"

        train_emb, train_adatas = embed_h5ad_dir(adapter, train_dir, cfg.spatial_key)
        val_emb, val_adatas = embed_h5ad_dir(adapter, val_dir, cfg.spatial_key)

        adapter.cleanup()
        torch.cuda.empty_cache()

        return train_emb, val_emb, train_adatas, val_adatas

    def run(self, train_emb, val_emb, train_adatas, val_adatas,
            metadata, train_cfg, cfg, output_dir, local_rank):
        """Load labels, probe over seeds, save CSV."""
        metadata = {**metadata, "dataset_spec": cfg.dataset_spec, "spatial_key": cfg.spatial_key, "output_dir": str(output_dir)}
        train_lab, meta_train = self.load_labels(train_adatas, metadata)
        metadata.update(meta_train)
        val_lab, meta_val = self.load_labels(val_adatas, metadata)
        metadata.update(meta_val)
        metadata["_val_lab"] = val_lab
        self.collect_val_aux(val_adatas, metadata, cfg)

        dataset_name = metadata["dataset_name"]
        d_model = train_emb.shape[1]
        csv_path = output_dir / f"{dataset_name}.csv"
        write_header = needs_header(csv_path) if local_rank == 0 else False

        results = []
        extras = []
        for seed in train_cfg.seeds:
            result, extra = self.run_probe(
                train_emb, train_lab, val_emb, val_lab,
                d_model, seed, train_cfg, cfg,
            )
            results.append(result)
            extras.append(extra)

            if local_rank == 0:
                row = self.format_row(metadata, seed, result)
                append_csv_row(csv_path, self.csv_columns, row, write_header=write_header)
                write_header = False
                self.on_seed_done(seed, extra, output_dir, metadata)

        if local_rank == 0:
            for agg_name, agg_fn in [("mean", np.mean), ("std", np.std)]:
                agg_result = {k: agg_fn([r[k] for r in results]) for k in self.metric_keys}
                row = self.format_row(metadata, agg_name, agg_result)
                append_csv_row(csv_path, self.csv_columns, row)

            self.on_all_seeds_done(extras, output_dir, metadata)

            parts = []
            for k in self.metric_keys:
                vals = [r[k] for r in results]
                parts.append(f"{k} {np.mean(vals):.6f}\u00b1{np.std(vals):.6f}")
            print(f"{dataset_name} {cfg.model}: "
                  f"{', '.join(parts)} -> {csv_path}")

        return results

    def run_single(self, cfg: TaskConfig, radius_idx: int = 0):
        """Embed once, run one radius (or no radius for non-radius tasks)."""
        train_cfg = self.get_train_cfg(cfg)
        dataset_name = cfg.dataset_name or Path(cfg.data_path).stem
        radii = self.get_radii(cfg)
        if radii is not None and not 0 <= radius_idx < len(radii):
            raise ValueError(f"Invalid radius_idx={radius_idx} for radii={radii}")

        train_emb, val_emb, train_adatas, val_adatas = self._embed(cfg)

        output_dir = Path(cfg.output_dir) if cfg.output_dir else (
            Path("output/results") / self.task_name / "probe" / "nexust"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))

        metadata = {"dataset_name": dataset_name, "mode": "probe",
                    "model_name": cfg.model}
        if radii:
            metadata["radius_idx"] = radius_idx
            metadata["actual_radius"] = radii[radius_idx]

        results = self.run(
            train_emb, val_emb, train_adatas, val_adatas,
            metadata, train_cfg, cfg, output_dir, local_rank,
        )

        del train_adatas, val_adatas
        return results

    def run_all_radii(self, cfg: TaskConfig):
        """Embed once, loop over all radii."""
        train_cfg = self.get_train_cfg(cfg)
        dataset_name = cfg.dataset_name or Path(cfg.data_path).stem
        radii = self.get_radii(cfg)

        train_emb, val_emb, train_adatas, val_adatas = self._embed(cfg)

        output_dir = Path(cfg.output_dir) if cfg.output_dir else (
            Path("output/results") / self.task_name / "probe" / "nexust"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))

        all_results = {}
        for ridx, actual_radius in enumerate(radii):
            print(f"\n--- radius {ridx}/{len(radii)}: {actual_radius} ---")
            metadata = {
                "dataset_name": dataset_name,
                "mode": "probe",
                "model_name": cfg.model,
                "radius_idx": ridx,
                "actual_radius": actual_radius,
            }
            all_results[ridx] = self.run(
                train_emb, val_emb, train_adatas, val_adatas,
                metadata, train_cfg, cfg, output_dir, local_rank,
            )

        del train_adatas, val_adatas
        return all_results
