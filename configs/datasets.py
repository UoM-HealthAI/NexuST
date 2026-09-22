"""Built-in dataset descriptions; paths belong in the user's run config."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class DatasetSpec:
    label_col: str | None = None
    region_col: str | None = None
    n_classes: int | None = None
    platform: str = "unknown"
    specie: str = "human"
    density_radii: list[float] = field(default_factory=list)
    niche_radii: list[float] = field(default_factory=list)

with Path(__file__).with_name("datasets.yaml").open() as f:
    DATASET_CONFIG = {name: DatasetSpec(**values)
                      for name, values in yaml.safe_load(f)["datasets"].items()}
