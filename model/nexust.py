import torch
import torch.nn as nn
from typing import Dict, Tuple
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.layers import (
    SinusoidalPE,
    OrganEncoder,
    ValueEncoder,
    GeneEncoder,
    GeneAttentionLayer,
    CellAttentionLayer,
    DenseMoE,
    library_decoder,
    BatchEncoder,
)


class NexuSTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.gene_layer = GeneAttentionLayer(d_model, n_heads, dropout)
        self.cell_layer = CellAttentionLayer(d_model, n_heads, dropout)

    def forward(
        self,
        gene_tokens: torch.Tensor,
        cls_token: torch.Tensor,
        gene_padding_mask: torch.Tensor,
        batch_size: int,
        n_spots: int,
        cell_padding_mask: torch.Tensor = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        gene_tokens, cls_token = self.gene_layer(
            gene_tokens, cls_token, gene_padding_mask, return_attention=return_attention
        )  # cls_token: [B*N, D]
        cls_token = cls_token.view(batch_size, n_spots, -1)  # [B, N, D]
        cls_token = self.cell_layer(cls_token, cell_padding_mask, return_attention=return_attention)
        cls_token = cls_token.view(batch_size * n_spots, 1, -1)  # [B*N, 1, D]

        return gene_tokens, cls_token


class NexuSTEncoder(nn.Module):
    def __init__(
        self,
        gene_vocab_size: int,
        organ_vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        use_activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_activation_checkpointing = use_activation_checkpointing

        self.gene_encoder = GeneEncoder(gene_vocab_size, d_model, padding_idx=0)
        self.value_encoder = ValueEncoder(d_model, dropout)
        self.organ_encoder = OrganEncoder(organ_vocab_size, d_model, padding_idx=0)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        self.layers = nn.ModuleList([
            NexuSTEncoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ln_output = nn.LayerNorm(self.d_model)
        self.pos_encoder = SinusoidalPE(d_model)

    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        organ_ids: torch.Tensor,
        coords: torch.Tensor,
        cell_padding_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        return_attention: bool = False,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward through encoder only

        Args:
            gene_ids: [B, N, G]
            gene_values: [B, N, G]
            organ_ids: [B] organ IDs per sample
            coords: [B, N, 2]
            cell_padding_mask: [B, N] bool, True = padding, False = real cell (optional)
            mask: [B, N, G] bool tensor for masked positions (optional)
            return_attention: if True, collect attention weights from all layers
            return_hidden_states: return the output-normalized CLS state after
                every encoder layer as ``[B, L, N, D]``.

        Returns:
            Dict with cls_token [B, N, D] and gene_tokens [B*N, 1+G, D] (organ + genes)
            If return_attention=True, also includes:
                - gene_qk: list of (q, k), each tensor shaped
                  [B*N, n_heads, 2+G, head_dim] (CLS + organ + genes)
                - cell_qk: list of (q, k), each tensor shaped
                  [B, n_heads, N, head_dim]
        """
        B, N, G = gene_ids.shape

        gene_ids_flat = gene_ids.view(B * N, G)
        gene_values_flat = gene_values.view(B * N, G)
        mask_flat = mask.view(B * N, G) if mask is not None else None

        gene_padding_mask = (gene_ids_flat == 0)  # [B*N, G]
        gene_embeds = self.gene_encoder(gene_ids_flat)
        value_embeds = self.value_encoder(gene_values_flat, mask=mask_flat)
        gene_tokens = gene_embeds + value_embeds  # [B*N, G, D]

        if organ_ids is not None:
            organ_ids_expanded = organ_ids.unsqueeze(1).expand(B, N).reshape(B * N)  # [B] -> [B*N]
            organ_emb = self.organ_encoder(organ_ids_expanded)  # [B*N, D]
            organ_token = organ_emb.unsqueeze(1)  # [B*N, 1, D]
            gene_tokens = torch.cat([organ_token, gene_tokens], dim=1)  # [B*N, 1+G, D]
            organ_mask = torch.zeros(B * N, 1, dtype=torch.bool, device=gene_ids.device)
            gene_padding_mask = torch.cat([organ_mask, gene_padding_mask], dim=1)  # [B*N, 1+G]

        cls_token = self.cls_token.expand(B * N, 1, self.d_model)
        pos_embedding = self.pos_encoder(coords)  # [B, N, D]
        pos_embedding_flat = pos_embedding.view(B * N, 1, self.d_model)
        cls_token = cls_token + pos_embedding_flat

        gene_qk = []  # list of (q, k) for each layer
        cell_qk = []  # list of (q, k) for each layer
        cls_hidden_states = []

        for layer in self.layers:
            if self.use_activation_checkpointing and not return_attention:
                gene_tokens, cls_token = checkpoint(
                    layer,
                    gene_tokens, cls_token, gene_padding_mask, B, N,
                    cell_padding_mask=cell_padding_mask,
                    return_attention=False,
                    use_reentrant=False,
                )
            else:
                gene_tokens, cls_token = layer(
                    gene_tokens, cls_token, gene_padding_mask,
                    B, N, cell_padding_mask=cell_padding_mask,
                    return_attention=return_attention
                )
            if return_attention:
                gene_qk.append((layer.gene_layer._last_q, layer.gene_layer._last_k))
                cell_qk.append((layer.cell_layer._last_q, layer.cell_layer._last_k))
            if return_hidden_states:
                layer_cls = cls_token.squeeze(1).view(B, N, self.d_model)
                cls_hidden_states.append(self.ln_output(layer_cls))

        cls_token = cls_token.squeeze(1)  # [B*N, D]
        cls_token = cls_token.view(B, N, self.d_model)  # [B, N, D]
        cls_token = self.ln_output(cls_token)

        result = {
            'cls_token': cls_token,  # [B, N, D]
            'gene_tokens': gene_tokens,  # [B*N, G, D]
            'gene_embeds': gene_embeds,  # [B*N, G, D]
        }
        if return_attention:
            result['gene_qk'] = gene_qk  # each q/k: [B*N, n_heads, 2+G, head_dim]
            result['cell_qk'] = cell_qk  # list of (q, k), each [B, n_heads, N, head_dim]
        if return_hidden_states:
            result['cls_hidden_states'] = torch.stack(cls_hidden_states, dim=1)

        return result



class NexuST(nn.Module):
    def __init__(
        self,
        gene_vocab_size: int,
        organ_vocab_size: int,
        n_genes: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        num_experts: int = 8,
        batch_vocab_size: int = 16,
        use_library: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_genes = n_genes
        self.use_library = use_library
        self.NexuSTEncoder = NexuSTEncoder(
            gene_vocab_size=gene_vocab_size,
            organ_vocab_size=organ_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.cell_decoder = DenseMoE(
            d_model=d_model,
            num_experts=num_experts,
            use_batch_labels=True,
        )
        self.library_decoder = library_decoder(d_model=d_model * 2)
        self.batch_encoder = BatchEncoder(batch_vocab_size, d_model, padding_idx=0)

    @torch.no_grad()
    def build_mask(
        self,
        gene_ids: torch.Tensor,
        gene_mask_ratio: float = 0.20,
        deterministic: bool = False,
    ) -> torch.Tensor:
        device = gene_ids.device
        B, N, G = gene_ids.shape

        generator = torch.Generator(device=device).manual_seed(42) if deterministic else None
        valid = (gene_ids != 0)
        rnd = torch.rand(B, N, G, device=device, generator=generator)
        mask = (rnd < gene_mask_ratio) & valid

        return mask

    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        coords: torch.Tensor,
        organ_ids: torch.Tensor,
        gene_mask_ratio: float,
        deterministic: bool = False,
        cell_padding_mask: torch.Tensor = None,
        batch_labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        B, N, G = gene_ids.shape
        mask = self.build_mask(gene_ids, gene_mask_ratio, deterministic=deterministic)
        latent = self.NexuSTEncoder(gene_ids, gene_values, organ_ids, coords, cell_padding_mask=cell_padding_mask, mask=mask)

        cls_token = latent['cls_token']  # [B, N, D]
        gene_embeds = latent['gene_embeds']  # [B*N, G, D]

        cls_token_flat = cls_token.view(B * N, self.d_model)  # [B*N, D]

        batch_emb = self.batch_encoder(batch_labels)  # [B, N, D]
        batch_emb_flat = batch_emb.view(B * N, self.d_model)  # [B*N, D]
        cls_token_flat = torch.cat([cls_token_flat, batch_emb_flat], dim=-1)  # [B*N, 2D]

        # DenseMoE expects (gene_embs, cell_emb): gene_embs [B*N, G, D], cell_emb [B*N, 2D]
        gene_logits = self.cell_decoder(gene_embeds, cls_token_flat).view(B, N, G)  # -> [B, N, G]

        if self.use_library:
            valid_gene = (gene_ids != 0)  # [B, N, G]
            rho = torch.softmax(gene_logits.masked_fill(~valid_gene, float("-inf")), dim=-1)
            library = F.softplus(self.library_decoder(cls_token_flat)).view(B, N) + 1e-8
            cell_expr_pred = library.unsqueeze(-1) * rho  # [B, N, G]
            cell_expr_pred = torch.log1p(cell_expr_pred)
            return {
                'cell_expr_pred': cell_expr_pred,  # [B, N, G]
                'mask': mask,  # [B, N, G]
                'rho': rho,  # [B, N, G]
                'library': library,  # [B, N]
            }
        else:
            # Ablation: directly use MoE output as prediction
            return {
                'cell_expr_pred': gene_logits,  # [B, N, G]
                'mask': mask,  # [B, N, G]
            }
