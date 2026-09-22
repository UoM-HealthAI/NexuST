"""Probe configuration with optional built-in dataset defaults."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from omegaconf import OmegaConf
from configs.datasets import DATASET_CONFIG, DatasetSpec
from probe.embed import EmbeddingConfig, NexuSTEmbedder

@dataclass
class TrainingConfig:
    max_epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 64
    patience: int = 5
    num_workers: int = 4
    weight_decay: float = 1e-6
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])

@dataclass
class ComputeConfig:
    devices: int = 1
    accelerator: str = "auto"
    precision: str = "bf16-true"

@dataclass
class TaskConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    classification: TrainingConfig = field(default_factory=lambda: TrainingConfig(lr=0.1))
    region_classification: TrainingConfig = field(default_factory=lambda: TrainingConfig(lr=0.01))
    imputation: TrainingConfig = field(default_factory=TrainingConfig)
    niche: TrainingConfig = field(default_factory=TrainingConfig)
    density: TrainingConfig = field(default_factory=TrainingConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    data_path: str = ""
    dataset_name: str | None = None
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    spatial_key: str = "spatial"
    output_dir: str | None = None

    @property
    def model(self):
        return "nexust"

    @property
    def dataset_spec(self):
        return self.dataset

    @classmethod
    def from_yaml(cls, yaml_path, overrides=None):
        supplied = OmegaConf.load(yaml_path)
        if overrides:
            supplied = OmegaConf.merge(supplied, overrides)
        name = supplied.get("dataset_name") or Path(supplied.get("data_path", "")).name
        defaults = {"dataset": asdict(DATASET_CONFIG[name])} if name in DATASET_CONFIG else {}
        config = OmegaConf.merge(OmegaConf.structured(cls), defaults, supplied)
        result = OmegaConf.to_object(config)
        if not result.data_path or not result.embedding.pretrain_ckpt:
            raise ValueError("Set data_path and embedding.pretrain_ckpt in the probe config")
        return result

    def build_embedder(self):
        return NexuSTEmbedder(self.embedding)
