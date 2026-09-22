"""NexuST region classification fine-tuning.

Same training recipe as ``classification.py`` (same NexuSTClassifier, same
datasets, same recorder) but reads ``obs[region_col]`` instead of the cell-type
label. ``label_col`` defaults to ``DATASET_CONFIG[dataset].region_col``; passing
``--label_col`` overrides.

``n_classes`` is inferred from the fitted LabelEncoder — no per-dataset
hardcoding.
"""

import argparse
import pytorch_lightning as pl
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent

from data.tokenizer import get_tokenizer  # noqa: E402
from finetune.tasks.classification import (  # noqa: E402
    TrainClassificationDataset,
    ValClassificationDataset,
    ClassificationCollator,
    NexuSTClassifier,
    ClassificationMetricsRecorder,
)
from finetune.utils.cli import add_shared_args, build_ckpt_dir, append_csv  # noqa: E402
from finetune.utils.training import TrainingRunner, get_global_rank  # noqa: E402

from configs.datasets import DATASET_CONFIG  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='NexuST Region Classification')
    add_shared_args(parser)

    # Task-specific
    parser.add_argument('--pretrain_ckpt', type=str, required=True)
    parser.add_argument('--label_col', type=str, default=None,
                        help='obs column for region labels. '
                             'Defaults to DATASET_CONFIG[dataset].region_col.')
    parser.add_argument('--head_lr', type=float, default=None)
    parser.add_argument('--n_spots', type=int, default=512)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Shared run identifier; defaults to current timestamp. '
                             'Set via sbatch (e.g. SLURM_ARRAY_JOB_ID) to group '
                             'all array tasks under one ckpt dir.')

    parser.set_defaults(output_dir=str(Path.cwd() / 'output/results/region_prediction'))
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset_name or Path(args.data_path).stem
    mode = "finetune"
    seed = args.seed
    pl.seed_everything(seed, workers=True)

    if args.label_col is None:
        ds = DATASET_CONFIG.get(dataset_name)
        if ds is None or not ds.region_col:
            raise ValueError(
                f"Dataset {dataset_name!r} has no region_col configured in "
                f"configs/datasets.yaml. Either add it or pass "
                f"--label_col explicitly."
            )
        args.label_col = ds.region_col
    print(f"Region label column: {args.label_col}", flush=True)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = build_ckpt_dir(args.ckpt_root, "region_classification", run_id,
                              dataset_name, mode, f"seed{seed}")

    tokenizer = get_tokenizer()
    train_dir = Path(args.data_path) / "train"
    val_dir = Path(args.data_path) / "val"

    train_dataset = TrainClassificationDataset(
        train_dir=str(train_dir), tokenizer=tokenizer,
        label_col=args.label_col, max_gene_len=args.max_gene_len, n_spots=args.n_spots)
    label_encoder = train_dataset.label_encoder
    num_classes = len(label_encoder.classes_)

    val_dataset = ValClassificationDataset(
        val_dir=str(val_dir), tokenizer=tokenizer,
        label_col=args.label_col, label_encoder=label_encoder,
        max_gene_len=args.max_gene_len, max_spots=args.n_spots)

    collator = ClassificationCollator(max_gene_len=args.max_gene_len)

    model = NexuSTClassifier(
        pretrain_ckpt=args.pretrain_ckpt, num_classes=num_classes,
        head_lr=args.head_lr or args.lr, encoder_lr=args.encoder_lr,
        min_lr=args.min_lr,
        gradient_checkpointing=args.gradient_checkpointing)

    recorder = ClassificationMetricsRecorder(label_encoder=label_encoder, save_dir=ckpt_dir)

    runner = TrainingRunner(
        seed=seed, max_epochs=args.max_epochs, devices=args.devices,
        batch_size=args.batch_size, accumulate_grad_batches=args.accumulate_grad_batches,
        num_workers=args.num_workers,
        monitor='val_acc', monitor_mode='max', early_stop_patience=5,
        ckpt_dir=ckpt_dir,
        ckpt_filename=f"seed{seed}_" + "epoch{epoch:02d}_acc{val_acc:.4f}",
        wandb_project=args.project, logger=args.logger,
        accelerator=args.accelerator, precision=args.precision,
        wandb_group=args.group or f"{dataset_name}-region-{mode}",
        wandb_name=f"{dataset_name}_region_{mode}_seed{seed}",
        wandb_config={
            "mode": mode, "lr": args.head_lr or args.lr,
            "head": "linear", "pretrained_head": False, "phase": args.phase,
            "task": "region_classification", "dataset": dataset_name, "model": "nexust",
            "head_lr": args.head_lr or args.lr, "encoder_lr": args.encoder_lr,
            "batch_size": args.batch_size, "seed": seed,
            "label_col": args.label_col, "num_classes": num_classes,
        },
    )

    runner.fit(model, train_dataset, val_dataset, collator, callbacks=[recorder])

    if get_global_rank() == 0:
        results = recorder.read_results()
        acc = results.get('acc', float('nan'))
        f1 = results.get('f1', float('nan'))

        # Layout: output_dir/{mode_dir}/nexust/{dataset}.csv
        # Preserve the existing full-finetune results layout.
        mode_dir = 'finetune'
        csv_path = append_csv(
            Path(args.output_dir) / mode_dir / 'nexust', f"{dataset_name}.csv",
            header=['dataset', 'mode', 'seed', 'accuracy', 'f1'],
            row=[dataset_name, mode, seed, f'{acc:.4f}', f'{f1:.4f}'])

        print(f"Seed {seed}: acc={acc:.4f}, f1={f1:.4f}, epoch={results.get('epoch', -1)}")
        print(f"Results: {csv_path} | Metrics JSON: {ckpt_dir / 'best_metrics.json'}")


if __name__ == '__main__':
    main()
