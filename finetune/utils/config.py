from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
import os


# Data preparation root; override with NEXUST_DATA_ROOT.
CORPUS_ROOT = Path(os.environ.get("NEXUST_DATA_ROOT", "datasets"))
VAL_PROCESSED_ROOT = CORPUS_ROOT / "val"
DOWNSTREAM_ROOT = CORPUS_ROOT / "downstream"


@dataclass
class ValProcessedConfig:
    """Config for the single validation h5ad under val_processed/."""
    # Relative path under val_processed/ WITHOUT the .h5ad suffix (e.g., "merfish/adult_umb5958")
    rel_path: str
    # Label column in adata.obs
    label_col: str
    # FOV column (None = use pseudo FOV)
    library_key: str = None
    # Grid size for pseudo FOV (only used if library_key is None)
    pseudo_fov_bins: int = 10
    # Spatial neighborhood radii for niche prediction
    niche_radius: float | list[float] = 0.5
    # Spatial neighborhood radii for density prediction
    density_radius: float | list[float] = 0.5

    @property
    def val_h5ad_path(self) -> Path:
        """Path to original h5ad in val_processed."""
        return VAL_PROCESSED_ROOT / f"{self.rel_path}.h5ad"


@dataclass
class DownstreamDirConfig:
    """Config for train/val split dirs under downstream/ (for linear probing)."""
    # Relative path under downstream/ (e.g., "adult_umb5958")
    rel_path: str

    @property
    def downstream_dir(self) -> Path:
        """Path to downstream directory (contains train/val subdirs)."""
        return DOWNSTREAM_ROOT / self.rel_path

    @property
    def train_dir(self) -> Path:
        return self.downstream_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.downstream_dir / "val"


@dataclass
class DatasetConfig:
    """Dataset config composed of val_processed + (optional) downstream split."""
    val: ValProcessedConfig
    downstream: Optional[DownstreamDirConfig] = None

    # ---- Compatibility aliases (minimize callsite changes) ----
    @property
    def label_col(self) -> str:
        return self.val.label_col

    @property
    def library_key(self) -> str:
        return self.val.library_key

    @property
    def pseudo_fov_bins(self) -> int:
        return self.val.pseudo_fov_bins

    @property
    def niche_radius(self) -> float | list[float]:
        return self.val.niche_radius

    @property
    def density_radius(self) -> float | list[float]:
        return self.val.density_radius

    @property
    def val_h5ad_path(self) -> Path:
        return self.val.val_h5ad_path

    @property
    def train_dir(self) -> Optional[Path]:
        return None if self.downstream is None else self.downstream.train_dir

    @property
    def val_dir(self) -> Optional[Path]:
        return None if self.downstream is None else self.downstream.val_dir

    @property
    def downstream_dir(self) -> Optional[Path]:
        return None if self.downstream is None else self.downstream.downstream_dir


# All downstream datasets
DATASETS: Dict[str, DatasetConfig] = {
    "cosmx_liver_cancer": DatasetConfig(
        val=ValProcessedConfig(
            rel_path="cosmx/cosmx_liver_cancer",
            label_col="cellType",
            library_key="fov",
            niche_radius=[0.21, 0.30, 0.49, 0.71],
            density_radius=[0.25, 0.50, 0.75],
        ),
        downstream=DownstreamDirConfig(rel_path="cosmx_liver_cancer"),
    ),
    "cosmx_liver_normal": DatasetConfig(
        val=ValProcessedConfig(
            rel_path="cosmx/cosmx_liver_normal",
            label_col="cellType",
            library_key="fov",
            niche_radius=[0.25, 0.35, 0.56, 0.81],
            density_radius=[0.25, 0.50, 0.75],
        ),
        downstream=DownstreamDirConfig(rel_path="cosmx_liver_normal"),
    ),
    "adult_umb5958": DatasetConfig(
        val=ValProcessedConfig(
            rel_path="merfish/adult_umb5958",
            label_col="H1_annotation",
            library_key=None,
            pseudo_fov_bins=10,
            niche_radius=[0.29, 0.42, 0.68, 0.99],
            density_radius=[0.25, 0.50, 1.00],
        ),
        downstream=DownstreamDirConfig(rel_path="adult_umb5958"),
    ),
    "pulmonary_fibrosis_xenium": DatasetConfig(
        val=ValProcessedConfig(
            rel_path="xenium/Pulmonary_Fibrosis/pulmonary_fibrosis_xenium",
            label_col="final_CT",
            library_key="sample",
            niche_radius=[0.12, 0.17, 0.28, 0.40],
            density_radius=[0.15, 0.25, 0.40],
        ),
        downstream=DownstreamDirConfig(rel_path="pulmonary_fibrosis_xenium"),
    ),
}
