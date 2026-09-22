import argparse
import sys
from pathlib import Path
from datetime import datetime


from engines.trainer import PretrainTrainer
from configs.pretrain_config import Config


def main():
    parser = argparse.ArgumentParser(description='NexuST Pretraining Script')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    # Overrides
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--accumulate_grad_batches', type=int, default=None)
    parser.add_argument('--gene_mask_ratio', type=float, default=None)
    parser.add_argument('--num_nodes', type=int, default=None)
    parser.add_argument('--devices', type=int, default=None)
    args = parser.parse_args()
    config = Config.from_yaml(args.config)

    # Apply overrides
    overrides = [
        (args.learning_rate, config.training, "learning_rate"),
        (args.max_steps, config.training, "max_steps"),
        (args.warmup_steps, config.training, "warmup_steps"),
        (args.accumulate_grad_batches, config.training, "accumulate_grad_batches"),
        (args.gene_mask_ratio, config.model, "gene_mask_ratio"),
        (args.num_nodes, config.compute, "num_nodes"),
        (args.devices, config.compute, "devices"),
    ]
    for val, obj, attr in overrides:
        if val is not None:
            setattr(obj, attr, val)
            print(f"Override: {attr} = {val}")

    if args.resume:
        config.resume_from_checkpoint = args.resume
        print(f"Resume from: {args.resume}")

    # Generate run_name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    if args.run_name:
        config.logging.run_name = f"{args.run_name}_{timestamp}"
    else:
        lr = config.training.learning_rate
        eff_bs = config.dataset.batch_size * config.training.accumulate_grad_batches * config.compute.devices * config.compute.num_nodes
        print(f"effbs is {eff_bs}")
        config.logging.run_name = f"pretrain_{timestamp}"

    PretrainTrainer(config).train()


if __name__ == '__main__':
    main()
