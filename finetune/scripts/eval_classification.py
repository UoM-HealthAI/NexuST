"""Re-evaluate a single NexuST classification checkpoint on the full val set.

Used to regenerate ``best_metrics.json`` files that were polluted by the DDP
val-sharding bug (``_val_preds`` / ``_val_labels`` were computed on 1/N of
the data, leading to wrong per_class and confusion_matrix entries). Scalar
``acc``/``f1`` were mostly correct because torchmetrics auto-syncs on
``compute()``, but DistributedSampler padding could introduce a small bias
on datasets where ``len(val) % world_size != 0``. This script overwrites
the stale JSON with full-data values.

Metrics are computed with torchmetrics, matching the NexuST classification
training module. This script evaluates the complete validation set.

Inference is single-GPU by design: no DistributedSampler, no per-rank state,
no race on JSON write. Intended to be launched as a SLURM array job, one
task per checkpoint.

Usage:
    python finetune/scripts/eval_classification.py \
        --ckpt_path checkpoints/classification/model.ckpt \
        --data_path datasets/downstream/adult_umb5958 \
        --label_col H1_annotation
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.tokenizer import get_tokenizer
from finetune.tasks.classification import (
    ClassificationCollator,
    NexuSTClassifier,
    TrainClassificationDataset,
    ValClassificationDataset,
)


def build_label_encoder(data_path: Path, label_col: str, max_gene_len: int, n_spots: int):
    """Rebuild LabelEncoder by scanning the train split (matches training setup)."""
    tokenizer = get_tokenizer()
    train_dataset = TrainClassificationDataset(
        train_dir=str(data_path / "train"),
        tokenizer=tokenizer,
        label_col=label_col,
        max_gene_len=max_gene_len,
        n_spots=n_spots,
    )
    return train_dataset.label_encoder


def build_val_loader(data_path: Path, label_col: str, label_encoder,
                     max_gene_len: int, n_spots: int, num_workers: int):
    tokenizer = get_tokenizer()
    val_dataset = ValClassificationDataset(
        val_dir=str(data_path / "val"),
        tokenizer=tokenizer,
        label_col=label_col,
        label_encoder=label_encoder,
        max_gene_len=max_gene_len,
        max_spots=n_spots,
    )
    collator = ClassificationCollator(max_gene_len=max_gene_len)
    return DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


@torch.no_grad()
def run_forward(model: NexuSTClassifier, val_loader: DataLoader, device: str):
    all_preds, all_labels = [], []
    autocast_device = device.split(":")[0]
    for batch in val_loader:
        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            logits = model(
                gene_ids=batch["gene_ids"].to(device),
                gene_values=batch["gene_values"].to(device),
                coords=batch["coords"].to(device),
                organ_ids=batch["organ_ids"].to(device),
            )
        preds = logits.view(-1, logits.size(-1)).argmax(-1).cpu().numpy()
        labels = batch["labels"].view(-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(preds: np.ndarray, labels: np.ndarray, label_encoder):
    """Compute metrics with torchmetrics to match training conventions exactly.

    Uses the same metric classes as ``NexuSTClassifier`` (MulticlassAccuracy
    with average='micro', MulticlassF1Score with average='macro'). This also
    matches ``baseline/nicheformer``'s ``F1Score(task='multiclass',
    average='macro')`` so NexuST and nicheformer results stay directly
    comparable.

    In particular, macro F1 follows torchmetrics' default behavior of excluding
    classes with zero support in val from the macro average denominator — this
    is the same convention used during training, and the same convention as
    nicheformer's f1_macro_val.
    """
    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)

    preds_t = torch.from_numpy(preds).long()
    labels_t = torch.from_numpy(labels).long()

    # Scalar metrics (same metric classes as NexuSTClassifier in classification.py)
    acc = float(MulticlassAccuracy(num_classes=n_classes, average="micro")(preds_t, labels_t))
    f1 = float(MulticlassF1Score(num_classes=n_classes, average="macro")(preds_t, labels_t))

    # Per-class precision / recall / f1 (average=None → one value per class)
    prec_pc = MulticlassPrecision(num_classes=n_classes, average=None)(preds_t, labels_t)
    rec_pc = MulticlassRecall(num_classes=n_classes, average=None)(preds_t, labels_t)
    f1_pc = MulticlassF1Score(num_classes=n_classes, average=None)(preds_t, labels_t)

    # Per-class support = number of samples whose true label is that class
    support = torch.bincount(labels_t, minlength=n_classes)

    per_class = {
        name: {
            "precision": float(prec_pc[i]),
            "recall": float(rec_pc[i]),
            "f1-score": float(f1_pc[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }

    # Confusion matrix: rows = true label, columns = predicted label
    cm_t = MulticlassConfusionMatrix(num_classes=n_classes)(preds_t, labels_t)
    cm = cm_t.tolist()

    return acc, f1, per_class, cm


def parse_epoch(ckpt_name: str) -> int:
    m = re.search(r"epoch=(\d+)", ckpt_name)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Exact .ckpt file to evaluate")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Dataset dir containing train/ and val/")
    parser.add_argument("--label_col", type=str, required=True)
    parser.add_argument("--max_gene_len", type=int, default=300,
                        help="Must match training (default matches add_shared_args)")
    parser.add_argument("--n_spots", type=int, default=512,
                        help="Must match training (default matches classification.py)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_json", type=str, default=None,
                        help="Override output JSON path (default: <ckpt_dir>/best_metrics.json)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = Path(args.ckpt_path)
    data_path = Path(args.data_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    print(f"=== {data_path.name} | {ckpt_path.name} ===")

    label_encoder = build_label_encoder(
        data_path, args.label_col, args.max_gene_len, args.n_spots,
    )
    print(f"  label_encoder classes = {list(label_encoder.classes_)}")

    val_loader = build_val_loader(
        data_path, args.label_col, label_encoder,
        args.max_gene_len, args.n_spots, args.num_workers,
    )

    print(f"  load {ckpt_path.name}")
    model = NexuSTClassifier.load_from_checkpoint(
        str(ckpt_path), map_location=device, weights_only=False,
    )
    model.eval().to(device)

    preds, labels = run_forward(model, val_loader, device)
    acc, f1, per_class, cm = compute_metrics(preds, labels, label_encoder)

    out = {
        "acc": acc,
        "f1": f1,
        "epoch": parse_epoch(ckpt_path.name),
        "per_class": per_class,
        "confusion_matrix": cm,
    }
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = ckpt_path.parent / "best_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  acc={acc:.4f} f1={f1:.4f}  -> {out_path}")


if __name__ == "__main__":
    main()
