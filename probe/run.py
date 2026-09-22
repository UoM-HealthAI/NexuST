"""Run a NexuST linear probe: python -m probe --task ... --config ..."""
import argparse
from importlib import import_module
from probe.config import TaskConfig

TASKS = {
    "classification": ("classification", "ClassificationTask"),
    "region_classification": ("region_classification", "RegionClassificationTask"),
    "imputation": ("imputation", "ImputationTask"),
    "niche": ("niche_prediction", "NichePredictionTask"),
    "density": ("density_prediction", "DensityPredictionTask"),
}

def run_task(task_name, cfg, *, radius_idx=0, all_radii=False, dump_pred_dir=None):
    module, class_name = TASKS[task_name]
    task = getattr(import_module(f"probe.tasks.{module}"), class_name)()
    if all_radii and task_name not in ("niche", "density"):
        raise ValueError("--all_radii applies only to niche and density")
    if dump_pred_dir is not None and task_name != "imputation":
        raise ValueError("--dump_pred_dir applies only to imputation")
    if task_name == "imputation":
        task._dump_pred_dir = dump_pred_dir
    if all_radii:
        return task.run_all_radii(cfg)
    return task.run_single(cfg, radius_idx=radius_idx)

def main():
    parser = argparse.ArgumentParser(description="NexuST frozen linear probes")
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--config", required=True)
    for name in ("data_path", "dataset_name", "spatial_key", "output_dir"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--radius_idx", type=int, default=0)
    parser.add_argument("--all_radii", action="store_true")
    parser.add_argument("--dump_pred_dir")
    args = parser.parse_args()
    overrides = {k: getattr(args, k) for k in
                 ("data_path", "dataset_name", "spatial_key", "output_dir")
                 if getattr(args, k) is not None}
    cfg = TaskConfig.from_yaml(args.config, overrides)
    training = getattr(cfg, args.task)
    if args.seeds is not None:
        training.seeds = args.seeds
    if args.batch_size is not None:
        training.batch_size = args.batch_size
    run_task(args.task, cfg, radius_idx=args.radius_idx,
             all_radii=args.all_radii, dump_pred_dir=args.dump_pred_dir)

if __name__ == "__main__":
    main()
