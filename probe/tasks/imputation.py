"""Imputation linear probe (drop-token variant).

Drop target genes from the input panel entirely (not zero, not masked),
re-embed, then linearly predict their expression from the resulting cell
embedding. This corresponds to the "unmeasured gene prediction" setup:
the encoder never sees the target genes in this forward pass.

Flow:
    1. Load target gene list from {data_path}/hvg60_symbol.npy
    2. For each h5ad file:
         a. Read adata
         b. Save original expression at target gene columns -> labels
         c. Drop those columns from adata (var_names + X)
         d. Embed the reduced-panel adata via adapter
    3. Train Linear(d_model, n_target_genes) with MSE loss
    4. Evaluate: gene-wise Pearson correlation + per-gene MAE/PCC
"""

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from tqdm import tqdm


from probe.config import TaskConfig
from probe.heads import run_imputation_probe
from evaluation.io import append_csv_row, needs_header
from probe.tasks.base import BaseProbeTask


def _drop_and_embed_dir(adapter, data_dir: Path, spatial_key: str,
                        target_genes: list):
    """Load h5ads, drop target gene columns, embed, return labels + embeds.

    Returns:
        embeddings: (N, d_model) float32
        adatas:     list[AnnData] (target columns removed)
        labels:     (N, n_target_genes) float32 — original values before dropping
    """
    h5ad_files = sorted(Path(data_dir).rglob("*.h5ad"))
    if not h5ad_files:
        raise FileNotFoundError(f"No h5ad files found in {data_dir}")

    emb_list, adata_list, label_list = [], [], []
    target_set = set(target_genes)
    target_order = {g: i for i, g in enumerate(target_genes)}

    for path in tqdm(h5ad_files, desc=f"Drop+embed {Path(data_dir).name}"):
        adata = sc.read_h5ad(path)

        # Find target gene column indices in this adata
        var_names = list(adata.var_names)
        present_idx = [i for i, g in enumerate(var_names) if g in target_set]
        present_genes = [var_names[i] for i in present_idx]
        if len(present_idx) != len(target_genes):
            missing = target_set - set(present_genes)
            raise ValueError(
                f"{path.name}: {len(missing)} target genes missing from var_names: "
                f"{sorted(missing)[:5]}..."
            )

        # Reorder target columns to match target_genes ordering for consistent labels
        order = np.array([target_order[g] for g in present_genes])
        sorted_idx = np.argsort(order)
        col_idx = np.array(present_idx)[sorted_idx]

        # Extract original target expression (labels) before dropping
        X = adata.X
        if sp.issparse(X):
            target_expr = np.asarray(X[:, col_idx].toarray(), dtype=np.float32)
        else:
            target_expr = np.asarray(X[:, col_idx], dtype=np.float32)
        label_list.append(target_expr)

        # Drop target gene columns from adata entirely
        keep_mask = ~adata.var_names.isin(target_genes)
        adata_dropped = adata[:, keep_mask].copy()

        # Embed the reduced-panel adata
        emb = adapter.embed(adata_dropped, spatial_key=spatial_key)
        emb_list.append(emb)
        adata_list.append(adata_dropped)

    return (np.concatenate(emb_list, axis=0),
            adata_list,
            np.concatenate(label_list, axis=0))


class ImputationTask(BaseProbeTask):
    task_name = "imputation"
    csv_columns = ["dataset", "mode", "seed", "mse", "mae", "pcc"]
    metric_keys = ["mse", "mae", "pcc"]

    def __init__(self):
        super().__init__()
        self._label_cache = {}
        self._target_genes = None

    def get_train_cfg(self, cfg):
        return cfg.imputation

    def _embed(self, cfg: TaskConfig):
        """Override base _embed: drop target genes from input, then embed."""
        # Load target gene list
        hvg_path = Path(cfg.data_path) / "hvg60_symbol.npy"
        if not hvg_path.exists():
            raise FileNotFoundError(f"HVG file not found: {hvg_path}")
        target_genes = np.load(hvg_path, allow_pickle=True).tolist()
        self._target_genes = target_genes
        print(f"Loaded {len(target_genes)} target genes from {hvg_path}")

        adapter = cfg.build_embedder()
        adapter.load_model()

        train_dir = Path(cfg.data_path) / "train"
        val_dir = Path(cfg.data_path) / "val"

        train_emb, train_adatas, train_labels = _drop_and_embed_dir(
            adapter, train_dir, cfg.spatial_key, target_genes)
        val_emb, val_adatas, val_labels = _drop_and_embed_dir(
            adapter, val_dir, cfg.spatial_key, target_genes)

        adapter.cleanup()
        torch.cuda.empty_cache()

        # Cache labels keyed by the adata list identity so load_labels can find them
        self._label_cache = {
            id(train_adatas): train_labels,
            id(val_adatas): val_labels,
        }

        return train_emb, val_emb, train_adatas, val_adatas

    def load_labels(self, adatas, metadata):
        """Return the pre-extracted target expression cached during _embed."""
        labels = self._label_cache[id(adatas)]
        return labels, {
            "n_genes": len(self._target_genes),
            "gene_names": list(self._target_genes),
        }

    def run_probe(self, train_emb, train_lab, val_emb, val_lab,
                  d_model, seed, train_cfg, cfg):
        result, per_col_mae, per_col_pcc, val_preds = run_imputation_probe(
            train_emb, train_lab, val_emb, val_lab,
            d_model=d_model,
            n_genes=train_lab.shape[1],
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
        return [metadata["dataset_name"], "probe", seed,
                f"{result['mse']:.6f}", f"{result['mae']:.6f}",
                f"{result['pcc']:.4f}"]

    def on_seed_done(self, seed, extra, output_dir, metadata):
        dataset_name = metadata["dataset_name"]
        gene_names = metadata.get("gene_names", [])
        columns = ["dataset", "mode", "seed"] + gene_names
        base_row = [dataset_name, "probe", seed]

        csv_mae = output_dir / f"{dataset_name}_gene_mae.csv"
        row = base_row + [f"{v:.6f}" if not np.isnan(v) else "nan"
                          for v in extra["per_col_mae"]]
        append_csv_row(csv_mae, columns, row,
                       write_header=needs_header(csv_mae))

        csv_pcc = output_dir / f"{dataset_name}_gene_pcc.csv"
        row = base_row + [f"{v:.4f}" if not np.isnan(v) else "nan"
                          for v in extra["per_col_pcc"]]
        append_csv_row(csv_pcc, columns, row,
                       write_header=needs_header(csv_pcc))

        # Per-cell prediction dump (off by default; CLI --dump_pred_dir turns
        # it on). Written for the first seed only, since downstream
        # visualization uses a single seed and predictions across seeds are
        # near-identical for a frozen-encoder probe. Output order matches the
        # val H5AD obs order (sorted FOV name, then within-FOV obs).
        dump_dir = getattr(self, "_dump_pred_dir", None)
        if dump_dir is not None and not getattr(self, "_dumped_once", False):
            dump_dir = Path(dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            np.save(dump_dir / f"{dataset_name}.npy",
                    np.asarray(extra["val_preds"], dtype=np.float32))
            np.save(dump_dir / f"{dataset_name}_hvg.npy",
                    np.asarray(gene_names, dtype=object))
            print(f"[probe dump] wrote {extra['val_preds'].shape} "
                  f"→ {dump_dir / f'{dataset_name}.npy'}")
            self._dumped_once = True

    def on_all_seeds_done(self, extras, output_dir, metadata):
        dataset_name = metadata["dataset_name"]
        gene_names = metadata.get("gene_names", [])
        columns = ["dataset", "mode", "seed"] + gene_names

        for key, suffix, fmt in [
            ("per_col_mae", "gene_mae", ".6f"),
            ("per_col_pcc", "gene_pcc", ".4f"),
        ]:
            csv_path = output_dir / f"{dataset_name}_{suffix}.csv"
            stacked = np.stack([e[key] for e in extras], axis=0)
            for agg_name, agg_fn in [("mean", np.nanmean), ("std", np.nanstd)]:
                agg_vals = agg_fn(stacked, axis=0)
                row = [dataset_name, "probe", agg_name]
                row += [f"{v:{fmt}}" if not np.isnan(v) else "nan"
                        for v in agg_vals]
                append_csv_row(csv_path, columns, row)
