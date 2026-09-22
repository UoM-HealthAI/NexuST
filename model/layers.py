import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, freqs_scale: float = 10000.0):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D (x/y) sinusoidal encoding"
        self.d_model = d_model
        self.freqs_scale = float(freqs_scale)

        d_axis_half = d_model // 4

        i = torch.arange(d_axis_half, dtype=torch.float32)
        inv_freq = self.freqs_scale ** (-i / d_axis_half)  # [d_axis_half]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[..., 0:1]
        y = coords[..., 1:2]

        x_ang = 2 * math.pi * (x * self.inv_freq)
        y_ang = 2 * math.pi * (y * self.inv_freq)

        x_pe = torch.cat([torch.sin(x_ang), torch.cos(x_ang)], dim=-1)
        y_pe = torch.cat([torch.sin(y_ang), torch.cos(y_ang)], dim=-1)

        pe = torch.cat([x_pe, y_pe], dim=-1)
        return pe.to(coords.dtype)


class GeneEncoder(nn.Module):
    """Encode gene IDs"""
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        x = self.embedding(x)  
        x = self.enc_norm(x)
        return x


class ValueEncoder(nn.Module):
    """Encode gene expression values"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

        self.mask_embedding = nn.Parameter(torch.randn(d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B*N, G] expression values
            mask: [B*N, G] bool tensor, True for masked positions
        """
        x = x.unsqueeze(-1)  # [B*N, G, 1]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)  # [B*N, G, d_model]

        if mask is not None:
            mask_ = mask.unsqueeze(-1)            # [B*N, G, 1]
            mask_vec = self.mask_embedding.view(1, 1, -1)  # [1,1,D]
            x = torch.where(mask_, mask_vec, x)  
        return self.dropout(x)


class OrganEncoder(nn.Module):
    """Encode organ IDs"""
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class BatchEncoder(nn.Embedding):
    """Encode batch/platform IDs. Simple wrapper around nn.Embedding for explicit naming."""
    def __init__(self, batch_vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__(num_embeddings=batch_vocab_size, embedding_dim=d_model, padding_idx=padding_idx)


class GeneAttentionLayer(nn.Module):
    """Gene-level attention with CLS token"""
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._last_attn_weights = None

    def forward(
        self,
        gene_tokens: torch.Tensor,
        cls_token: torch.Tensor,
        gene_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gene_tokens: [bs*n_spots, n_genes, d_model]
            cls_token: [bs*n_spots, 1, d_model]
            gene_padding_mask: [bs*n_spots, n_genes] - True for padding positions
            return_attention: if True, compute and store attention weights
        Returns:
            gene_tokens: [bs*n_spots, n_genes, d_model]
            cls_token: [bs*n_spots, d_model]
        """
        B, N_genes, C = gene_tokens.shape
        tokens = torch.cat([cls_token, gene_tokens], dim=1)  # [B, N+1, C]
        N = tokens.shape[1]

        normed_tokens = self.norm1(tokens)        # prenorm
        qkv = self.qkv(normed_tokens).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = None
        if gene_padding_mask is not None:
            key_padding_mask = torch.zeros(
                B, N, dtype=torch.bool, device=tokens.device
            )
            key_padding_mask[:, 1:] = gene_padding_mask  # Skip CLS (position 0)
            attn_mask = torch.zeros(B, N, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(key_padding_mask, float('-inf'))   # -inf for masked positions
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, N, N)

        if return_attention:
            # Manual attention computation to get weights
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, n_heads, N, N]
            if attn_mask is not None:
                scores = scores + attn_mask
            self._last_q = q.detach()  # [B, n_heads, N, head_dim]
            self._last_k = k.detach()  # [B, n_heads, N, head_dim]
            attn_weights = F.softmax(scores, dim=-1)  # [B, n_heads, N, N]
            attn_out = torch.matmul(attn_weights, v)  # [B, n_heads, N, head_dim]
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0
            )  # [B, n_heads, N, head_dim]

        # residual
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        attn_out = self.proj(attn_out)
        tokens = tokens + self.dropout(attn_out)

        normed_tokens = self.norm2(tokens)
        ffn_out = self.ffn(normed_tokens)
        tokens = tokens + self.dropout(ffn_out)

        cls_token = tokens[:, 0, :]  # [bs*n_spots, d_model]
        gene_tokens = tokens[:, 1:, :]  # [bs*n_spots, n_genes, d_model]

        return gene_tokens, cls_token


class CellAttentionLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._last_attn_weights = None
        self._last_attn_scores = None  # Pre-softmax logits for contrast analysis

    def forward(
        self,
        cls_token: torch.Tensor,
        cell_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            cls_token: [bs, n_spots, d_model]
            cell_padding_mask: [bs, n_spots] bool, True = padding, False = real cell
            return_attention: if True, compute and store attention weights
        Returns:
            cls_token: [bs, n_spots, d_model]
        """
        B, N, C = cls_token.shape

        normed_tokens = self.norm1(cls_token)        # prenorm
        qkv = self.qkv(normed_tokens).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = None
        if cell_padding_mask is not None:
            attn_mask = torch.zeros(B, N, dtype=q.dtype, device=q.device)  # True = padding, False = real cell
            attn_mask.masked_fill_(cell_padding_mask, float('-inf'))
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]

        if return_attention:
            # Manual attention computation to get weights
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, n_heads, N, N]
            if attn_mask is not None:
                scores = scores + attn_mask
            self._last_q = q.detach()  # [B, n_heads, N, head_dim]
            self._last_k = k.detach()  # [B, n_heads, N, head_dim]
            attn_weights = F.softmax(scores, dim=-1)  # [B, n_heads, N, N]
            attn_out = torch.matmul(attn_weights, v)  # [B, n_heads, N, head_dim]
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0
            )  # [B, n_heads, N, head_dim]

        # residual
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        attn_out = self.proj(attn_out)
        cls_token = cls_token + self.dropout(attn_out)

        normed_tokens = self.norm2(cls_token)        # prenorm
        ffn_out = self.ffn(normed_tokens)
        cls_token = cls_token + self.dropout(ffn_out)

        return cls_token


class MVCExpert(nn.Module):
    def __init__(self, d_model: int, use_batch_labels: bool = False):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.gene2query = nn.Linear(d_model, d_model)
        self.query_activation = nn.GELU()
        self.W = nn.Linear(d_model, d_in, bias=False)

    def forward(self,  gene_embs: torch.Tensor, cell_emb: torch.Tensor) -> torch.Tensor:
        query_vecs = self.query_activation(self.gene2query(gene_embs))
        cell_emb = cell_emb.unsqueeze(2)
        pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
        return pred_value


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)


class TopKMoE(nn.Module):
    def __init__(self, d_model: int, use_batch_labels: bool = False, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self._last_gate_probs = None
        self.experts = nn.ModuleList([MVCExpert(d_model, use_batch_labels) for _ in range(num_experts)])
        self.gate = GatingNetwork(d_model, num_experts)
        self.top_k = top_k
    
    def forward(self, cell_emb: torch.Tensor, gene_embs: torch.Tensor) -> torch.Tensor:
        gate_probs = self.gate(gene_embs)  # [B, G, E]
        self._last_gate_probs = gate_probs.detach()

        topk_probs, topk_idx = gate_probs.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(gate_probs).scatter_(-1, topk_idx, 1)
        gate_probs = gate_probs * mask
        gate_probs = gate_probs / (gate_probs.sum(dim=-1, keepdim=True) + 1e-8)  # renormalize

        expert_outputs = []
        for expert in self.experts:
            output = expert(cell_emb, gene_embs)
            expert_outputs.append(output)
        expert_outputs = torch.stack(expert_outputs, dim=-1)

        mixed_output = (expert_outputs * gate_probs).sum(dim=-1)
        return mixed_output

    @staticmethod
    def compute_moe_metrics(gate_probs: torch.Tensor, dead_threshold: float = 0.01) -> Dict[str, torch.Tensor]:
        probs = gate_probs.view(-1, gate_probs.shape[-1])  # [B*G, E]
        num_experts = probs.shape[-1]
        expert_usage = probs.mean(dim=0)
        ideal = 1.0 / num_experts
        load_balance = 1.0 - (expert_usage - ideal).abs().mean() / ideal
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        dead_expert_count = (expert_usage < dead_threshold).sum()
        return {
            'expert_usage': expert_usage,
            'load_balance': load_balance,
            'routing_entropy': entropy,
            'dead_expert_count': dead_expert_count,
        }

    def get_moe_metrics(self, dead_threshold: float = 0.01) -> Optional[Dict[str, torch.Tensor]]:
        if self._last_gate_probs is None:
            return None
        return self.compute_moe_metrics(self._last_gate_probs, dead_threshold)


class DenseMoE(nn.Module):
    def __init__(self, d_model: int, use_batch_labels: bool = False, num_experts: int = 8):
        super().__init__()
        self._last_gate_probs = None
        self.experts = nn.ModuleList([MVCExpert(d_model, use_batch_labels) for _ in range(num_experts)])
        self.gate = GatingNetwork(d_model, num_experts)

    def forward(self, gene_embs: torch.Tensor, cell_emb: torch.Tensor) -> torch.Tensor:
        gate_probs = self.gate(gene_embs)
        self._last_gate_probs = gate_probs.detach()

        expert_outputs = torch.stack([expert(gene_embs, cell_emb) for expert in self.experts], dim=-1)
        mixed_output = (expert_outputs * gate_probs).sum(dim=-1)
        return mixed_output

    def get_moe_metrics(self, dead_threshold: float = 0.01) -> Optional[Dict[str, torch.Tensor]]:
        if self._last_gate_probs is None:
            return None
        return TopKMoE.compute_moe_metrics(self._last_gate_probs, dead_threshold)


class library_decoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, gene_tokens: torch.Tensor) -> torch.Tensor:
        return self.fc(gene_tokens).squeeze(-1)


class GeneDecoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, gene_tokens: torch.Tensor) -> torch.Tensor:
        return self.fc(gene_tokens).squeeze(-1)