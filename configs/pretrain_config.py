"""
Configuration management using dataclasses
"""
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from pathlib import Path
import yaml
from omegaconf import OmegaConf, DictConfig


class AnalysisConfig:
    """Backward-compat placeholder to load old checkpoints.

    Some older Lightning checkpoints may have pickled this class into
    hyper_parameters/callback state. Keeping this symbol allows torch.load
    (weights_only=False) to succeed even if the analysis callback is removed.
    """

    pass


@dataclass
class ModelConfig:
    """Model architecture configuration for NexuST"""
    gene_vocab_size: int = 19228  # Vocabulary size for gene embeddings
    organ_vocab_size: int = 16    # Vocabulary size for organ embeddings
    n_genes: int = 300       # Number of genes per sample
    d_model: int = 512        # Model dimension
    n_layers: int = 8        # Number of transformer layers
    n_heads: int = 8         # Number of attention heads
    dropout: float = 0.1     # Dropout rate

    # Gene-level masking parameters
    gene_mask_ratio: float = 0.20  # Mask ratio for genes per spot

    # Decoder parameters (DenseMoE)
    num_experts: int = 8  # Number of experts in MoE decoder
    batch_vocab_size: int = 4  # Vocabulary size for batch/platform embeddings (3 platforms + padding)
    use_library: bool = True  # Use library head for expression prediction (False for ablation)

    # Token parameters
    max_value: int = 512
    padding_idx: int = 0

@dataclass
class DatasetConfig:
    """Dataset configuration"""
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    slides_list: Optional[List[dict]] = None
    val_slides: Optional[List[str]] = None  # List of slide names (stems) to use for validation

    # Sampling parameters
    n_spots: int = 512
    max_gene_len: int = 300  # Maximum number of genes (for padding)
    gene_sampling: bool = True  # Whether to sample (True) or truncate (False) when n_genes > max_gene_len

    # DataLoader parameters
    batch_size: int = 16
    num_workers: int = 4

    # Sub-slide sampling parameters (FPS-based sampling)
    target_cells_per_subslide: int = 5000
    patches_per_subslide: int = 12  # Number of FPS centers per sub-slide


@dataclass
class TokenizerConfig:
    vocab_file: Optional[str] = None
    metadata_vocab_file: Optional[str] = None
    default_vocab_type: str = 'census'  # 'census' or 'standard'
    special_tokens: List[str] = field(default_factory=lambda: ["<pad>"])
    default_token: str = "<pad>"


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    # Training duration - use steps for large-scale pretraining
    max_steps: int = 20000               # Total training steps
    warmup_steps: int = 2000             # Linear warmup steps

    grad_clip_norm: float = 1.0
    accumulate_grad_batches: int = 1

    # Optimizer
    optimizer: str = 'adamw'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # Scheduler
    scheduler: str = 'cosine'
    min_lr: float = 1e-6

    # Early stopping
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4
    

@dataclass
class LoggingConfig:
    """Logging configuration"""
    project_name: str = 'HiGeST-pretrain'
    logger: str = 'csv'
    run_name: Optional[str] = None
    save_dir: str = './checkpoints'
    log_every_n_steps: int = 50
    val_check_interval: int = 500        # Validate every N steps
    save_every_n_steps: int = 5000       # Save checkpoint every N steps
    save_top_k: int = 1
    monitor_metric: str = 'val_loss'
    monitor_mode: str = 'min'


@dataclass
class ComputeConfig:
    """Compute configuration"""
    accelerator: str = 'gpu'
    devices: Any = 1  # Can be int or list
    num_nodes: int = 1
    precision: str = 'bf16-mixed'  # '32', '16-mixed', 'bf16-mixed'
    strategy: str = 'auto'  # 'auto', 'ddp', 'fsdp'
    compile_model: bool = False
    detect_anomaly: bool = False

    # FSDP Configuration
    fsdp_sharding_strategy: str = 'HYBRID_SHARD'  # 'FULL_SHARD', 'SHARD_GRAD_OP', 'HYBRID_SHARD', 'NO_SHARD' (recommended for multi-node)
    fsdp_activation_checkpointing: bool = False  # Whether to use activation checkpointing
    

@dataclass
class Config:
    """Master configuration for the model and training pipeline."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)

    seed: int = 42
    debug: bool = False
    resume_from_checkpoint: Optional[str] = None
    test_only: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file with optional base config"""
        from pathlib import Path

        config_path = Path(yaml_path)
        config_dir = config_path.parent

        # Check if base.yaml exists
        base_path = config_dir / 'base.yaml'

        if base_path.exists() and config_path.name != 'base.yaml':
            # Load base configuration first
            with open(base_path, 'r') as f:
                base_dict = yaml.safe_load(f)

            # Load specific configuration
            with open(yaml_path, 'r') as f:
                specific_dict = yaml.safe_load(f)

            # Use OmegaConf to merge configurations
            schema = OmegaConf.structured(cls)
            base_config = OmegaConf.merge(schema, OmegaConf.create(base_dict))
            final_config = OmegaConf.merge(base_config, OmegaConf.create(specific_dict))
        else:
            # Load single configuration file
            with open(yaml_path, 'r') as f:
                yaml_dict = yaml.safe_load(f)

            schema = OmegaConf.structured(cls)
            final_config = OmegaConf.merge(schema, OmegaConf.create(yaml_dict))

        return OmegaConf.to_object(final_config)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = OmegaConf.create(self)
        with open(yaml_path, 'w') as f:
            yaml.dump(OmegaConf.to_container(config_dict), f, default_flow_style=False)
    
    def update_from_dict(self, update_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        config_omega = OmegaConf.struct(self)
        updated = OmegaConf.merge(config_omega, OmegaConf.create(update_dict))
        return OmegaConf.to_object(updated)
    
    def print_config(self):
        """Pretty print configuration"""
        print(OmegaConf.to_yaml(self))


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
