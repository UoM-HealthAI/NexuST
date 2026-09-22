"""NexuST frozen embedding extraction."""
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class EmbeddingConfig:
    pretrain_ckpt: str = ""
    max_gene_len: int = 300
    max_cells: int = 1024
    device: str = "auto"

class NexuSTEmbedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.encoder = None
        self.tokenizer = None

    def load_model(self):
        from finetune.utils.tools import load_encoder
        from data.tokenizer import get_tokenizer
        self.device = self.config.device
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.encoder = load_encoder(self.config.pretrain_ckpt, device=self.device)
        self.encoder.to(self.device).eval().requires_grad_(False)
        self.d_model = self.encoder.d_model
        self.tokenizer = get_tokenizer()

    def embed(self, adata, spatial_key="spatial"):
        from inference import embed_adata
        if spatial_key != "spatial":
            adata = adata.copy()
            adata.obsm["spatial"] = adata.obsm[spatial_key]
        result = embed_adata(
            adata, self.encoder, self.tokenizer, device=self.device,
            d_model=self.d_model, max_cells=self.config.max_cells,
            max_gene_len=self.config.max_gene_len,
        )
        return result.obsm["X_nexust"].astype(np.float32)

    def cleanup(self):
        self.encoder = None
        self.tokenizer = None
