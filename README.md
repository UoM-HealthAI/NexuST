# NexuST

Hierarchical Gene-level Spatial Transcriptomics Foundation Model.

NexuST supports pretraining, cell embedding inference, frozen linear probing,
and end-to-end fine-tuning. Fine-tuning updates the encoder and a fresh linear
head; niche predictions additionally use softplus and normalization. The full
pretraining MoE decoder is retained. Other-model baselines and manuscript
reproduction are outside this release.

## Installation

Use Python 3.10 or newer. Install PyTorch >=2.7 for your CPU, CUDA or ROCm system,
then run from the repository root:

```bash
python -m pip install -e .
# Optional: Seurat-v3 highly variable gene selection
python -m pip install -e '.[preprocess]'
```

Run the commands below from the repository root and replace data/checkpoint
paths with your own. Installed console commands `nexust-pretrain`,
`nexust-embed`, and `nexust-probe` are also available. File paths in commands and
YAML files are relative to the working directory; bundled vocabularies are
loaded relative to their installed modules.

Local validation used Python 3.12, PyTorch 2.7.1 + ROCm 6.2.4 and Lightning 2.5.3.
CPU checks covered all five probe and fine-tuning tasks, embedding inference,
pretraining save/resume, and loading existing research checkpoints. Fresh
dependency resolution and production multi-node GPU runs have not been validated.

## Data and pretrained weights

- [HumanST-46M dataset](https://huggingface.co/datasets/Haiping-UoM/HumanST-46M):
  72 training and 4 validation H5AD files, approximately 99.63 GiB in total.
- [NexuST model](https://huggingface.co/Haiping-UoM/NexuST):
  `NexuST-step10000.ckpt`, the checkpoint saved at training step 10,000.

Both repositories are currently private and are planned for public release.
Until then, downloads require a Hugging Face account with access:

```bash
python -m pip install huggingface_hub
hf auth login
```

Download the checkpoint and validation data:

```python
from huggingface_hub import hf_hub_download, snapshot_download

checkpoint_path = hf_hub_download(
    repo_id="Haiping-UoM/NexuST",
    filename="NexuST-step10000.ckpt",
    local_dir="checkpoints",
)
snapshot_download(
    repo_id="Haiping-UoM/HumanST-46M",
    repo_type="dataset",
    allow_patterns=["val/**/*.h5ad"],
    local_dir="datasets/HumanST-46M",
)
```

For the full pretraining dataset, use
`allow_patterns=["train/**/*.h5ad", "val/**/*.h5ad"]`. Set the pretraining YAML's
`dataset.train_dir` and `dataset.val_dir` to the downloaded `train/` and `val/`
directories, then generate local patch indices as described below.

The dataset preserves platform subdirectories. It does not include sampling
indices or prepared downstream train/val splits; prepare those separately using
the data requirements below. Downloaded H5AD files are already preprocessed;
do not apply normalization or log1p again.

## Data preparation

NexuST reads AnnData `.h5ad` files:

| Field | Required content |
| --- | --- |
| `X` | Nonnegative log1p expression, sparse or dense; not scaled/z-scored |
| `var_names` | Gene symbols matching the checkpoint vocabulary |
| `obsm['spatial']` | Finite spatial coordinates of shape `(n_cells, 2)` |
| `uns['organ']` | Organ name from the metadata vocabulary; missing means unknown |
| `uns['platform']` | Platform name from the metadata vocabulary; missing means unknown |

Keep `data/gene_vocab.json` and `data/metadata_vocab.json` aligned with the
checkpoint. Unknown genes map to padding, so filter unsupported genes during
preparation. Do not rebuild the vocabulary for an existing checkpoint.
Keep raw counts in `layers['counts']` for HVG selection. Use
`python -m data.process.preprocess --help` for the expression preprocessing
options, and apply the same convention to training and validation.

Prepare downstream data as:

```text
datasets/downstream/my_dataset/
  train/*.h5ad
  val/*.h5ad
  hvg60_symbol.npy             # imputation only
```

Split by FOV/sample before evaluation. The validation split is used for model
selection and reported metrics; it is not an independent test split.

- **Classification / region classification:** supply `obs[label_col]`. Validation
  labels must occur in the training label vocabulary.
- **Imputation:** supply an ordered string array `hvg60_symbol.npy`. All target
  genes must occur in every input file and the model vocabulary. Select targets
  using training data. Probe and fine-tuning both remove target genes from the
  encoder input and predict their original values in `X`. The filename is
  historical; the number of targets can differ from 60.
- **Niche:** supply `obsm['X_niche_0']`, etc., and `uns['niche_info']` containing
  `columns`, `cell_type_col`, and `radii`. Keep the cell-type column order
  consistent across splits. The tasks normalize counts into proportions.
- **Density:** counts neighbors within each configured radius, excluding the
  center cell. All radii use the same units as the spatial coordinates; choose
  values appropriate to your data.

Generate niche labels with:

```bash
python -m finetune.utils.niche_composition \
  --data_dir datasets/downstream/my_dataset \
  --label_col cell_type --radii 0.25 0.5
```

This helper collects category names across the supplied directory, including
validation. For a strictly training-derived category vocabulary, call its
`niche_dir` Python API with `global_cell_types` set explicitly and
`collect_global_cell_types=False`.

### Pretraining patch indices

After preparing pretraining H5AD files, generate an index locally in each split:

```bash
python data/process/compute_patch_index.py --data_dir /absolute/path/to/pretrain/train
python data/process/compute_patch_index.py --data_dir /absolute/path/to/pretrain/val
```

Each command creates `patch_index.pt`. The script searches recursively and
selects CPU/GPU automatically. Defaults are 512 cells per patch, 15 FPS runs,
12 centers, and approximately 5,000 cells per subslide. Indices store slide
paths and cell indices: use absolute paths and regenerate after moving data
or changing cell content/order. Probe and fine-tuning do not need these indices.

## Pretraining

Edit `dataset.train_dir`, `dataset.val_dir`, and compute/logging settings in
`configs/yaml/small.yaml`, then run:

```bash
python train.py --config configs/yaml/small.yaml --run_name small
# Resume with a full training checkpoint, including optimizer/scheduler state:
python train.py --config configs/yaml/small.yaml --resume checkpoints/pretrain/RUN/last.ckpt
```

`small.yaml` is a single-device starter. `large.yaml` contains the large model
architecture and 8-node / 4-device FSDP settings; adjust these to your allocation.
YAML files inherit `base.yaml` when it is present beside them. Explicit CLI
options override YAML values. Checkpoints and resolved configuration are saved
under `logging.save_dir / run_name`, with a timestamp appended to the run name.
Local CSV logging is the default; set `logging.logger: wandb` to use the
`HiGeST-pretrain` project. For CPU, set `compute.accelerator: cpu` and
`compute.precision: 32-true`.

## Embedding inference

```bash
python inference.py --ckpt checkpoints/NexuST-step10000.ckpt \
  --h5ad datasets/HumanST-46M/val/cosmx/cosmx_liver_normal.h5ad \
  --out output/embedded.h5ad --device cuda:0
```

Use `--device cpu` for CPU execution, or `--gpus 0,1` for the existing multi-GPU
path. `--max_gene_len` and `--max_cells` control gene truncation and spatial
chunk size. The Python API is `inference.embed_adata`.

## Frozen linear probing

Probe loads the encoder in evaluation mode, disables gradients, embeds train
and validation data once, then trains a linear task head over fixed features.
Copy `configs/probe/nexust.yaml` and set your paths. A custom dataset can use:

```yaml
data_path: datasets/downstream/my_dataset
dataset_name: my_dataset
output_dir: output/probe
embedding:
  pretrain_ckpt: checkpoints/NexuST-step10000.ckpt
  max_gene_len: 300
  max_cells: 1024
  device: cpu
compute:
  devices: 1
  accelerator: cpu
  precision: 32-true
dataset:
  label_col: cell_type
  region_col: region
  density_radii: [0.25, 0.5]
  niche_radii: [0.25, 0.5]
```

Save this as `my_probe.yaml` and run the desired task:

```bash
python -m probe --task classification --config my_probe.yaml
python -m probe --task region_classification --config my_probe.yaml
python -m probe --task imputation --config my_probe.yaml
python -m probe --task niche --config my_probe.yaml --all_radii
python -m probe --task density --config my_probe.yaml --all_radii
```

Built-in dataset defaults come from `configs/datasets.yaml`; explicit `dataset`
fields override them. Task sections such as `classification` accept `seeds`,
`lr`, `weight_decay`, `batch_size`, `max_epochs`, `num_workers`, and `patience`.
CLI `--seeds`, `--batch_size`, and `--output_dir` override those values.
Use `--radius_idx 0` to evaluate one radius. Embedding device and head-training
compute are configured separately; the supplied template defaults to automatic
device selection and `bf16-true` head training.

Classification and region classification select the best validation macro-F1;
regression tasks select the best MSE. Results include CSV seed rows and
mean/std summaries plus task-specific JSON/per-target metrics. Imputation
`--dump_pred_dir PATH` saves predictions; niche supports `NICHE_DUMP_PREDS=1`;
region classification saves seed-42 predictions. Niche and region predictions
are written under `output_dir/predictions`. Prediction dumps use best-epoch
weights. Probe does not currently export a persistent trained-head checkpoint.

## Fine-tuning

Fine-tuning trains the encoder and a fresh linear head. Frozen evaluation uses
the separate probe workflow; there is no freeze switch or downstream decoder
head option.

```bash
python -m finetune.tasks.classification \
  --data_path datasets/downstream/my_dataset \
  --pretrain_ckpt checkpoints/NexuST-step10000.ckpt --label_col cell_type
```

All tasks require `--data_path` and `--pretrain_ckpt`:

| Module after `python -m finetune.tasks.` | Task arguments |
| --- | --- |
| `classification` | `--label_col cell_type` |
| `region_classification` | `--label_col region` or a built-in dataset name |
| `imputation` | Optional `--hvg_symbol_path`; defaults to `data_path/hvg60_symbol.npy` |
| `niche_prediction` | `--radius_idx 0` |
| `density_prediction` | `--radius_idx 0 --radii 0.25 0.5`; optional `--label_col` |

Use each module's `--help` for all options. Common controls include `--seed`,
`--lr`, `--encoder_lr`, `--batch_size`, `--accumulate_grad_batches`,
`--max_epochs`, `--max_gene_len`, `--num_workers`, `--devices`, `--accelerator`,
`--precision`, `--output_dir`, and `--ckpt_root`. Classification/region also
accept `--head_lr`; imputation uses its single-learning-rate recipe.
For CPU, add `--accelerator cpu --precision 32-true`. Logging defaults to local
CSV; `--logger wandb` uses `HiGeST-finetune`.

Classification/region select best validation accuracy, niche/density select
best MSE, and imputation selects best PCC. Fine-tuning uses sampled spatial
patches, whereas probe uses fixed embeddings. Niche applies linear -> softplus
-> normalized proportions; its fine-tuning head has no bias and its probe head
has bias, preserving the task implementations.

Downstream checkpoints are weights-only; the task CLIs do not expose optimizer
resume. Classification and niche evaluation scripts remain in
`finetune/scripts/`. When relocating the original pretraining checkpoint,
override `pretrain_ckpt=...` in the downstream Lightning `load_from_checkpoint`
call because that path is saved in the hyperparameters.

## Checkpoints and availability

Use a NexuST Lightning pretraining `.ckpt` with matching gene and metadata
vocabularies. `finetune.utils.tools.load_encoder` loads the encoder;
`load_nexust` loads the full pretraining model. Existing checkpoint module paths
and strict state-dict matching are retained; no format conversion is needed.

The Hugging Face file `NexuST-step10000.ckpt` is 815,205,507 bytes. Its SHA256 is:

```text
178de8d6ad027c2bb699de9e25ebf0dc76915d63701590752fbd8ed85e647d7e
```

It is byte-identical to the step-10,000 research checkpoint used in the local
GPU inference and short classification fine-tuning checks; the filename was
changed for distribution. This is not the separate best-validation-loss
checkpoint. Use it with the gene and metadata vocabularies bundled in this repo.

Project license and citation metadata are still pending.
