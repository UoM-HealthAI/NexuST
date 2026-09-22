from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch


@dataclass
class DataCollator:
    """
    Data collator for NexuST with gene padding support.

    Pads gene_values and gene_ids to max_gene_len when samples in a batch
    have different numbers of genes (due to local HVG per slide).

    Args:
        max_gene_len: Maximum number of genes (padding target)
        pad_token_id: Token ID for padding gene_ids (default: 0)
        pad_value: Value for padding gene_values (default: 0.0)
        gene_sampling: If True, randomly sample genes when n_genes > max_gene_len.
                  If False, truncate to first max_gene_len genes (default: True)
        gene_sampling_seed: If set, use fixed seed for deterministic gene sampling.
                  Useful for validation/downstream tasks to ensure reproducibility
                  while maintaining same distribution as random training. (default: None)
    """
    max_gene_len: int = 300
    pad_token_id: int = 0
    pad_value: float = 0.0
    gene_sampling: bool = True  # Whether to sample (True) or truncate (False) when n_genes > max_gene_len
    gene_sampling_seed: int = None  # Fixed seed for deterministic sampling (None = random)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Pad and collate batch samples.

        Args:
            batch: List of samples from dataset, each containing:
                - gene_values: [n_spots, n_genes] tensor
                - gene_ids: [n_spots, n_genes] tensor
                - coords: [n_spots, 2] tensor
                - batch_label: [n_spots] tensor (optional)

        Returns:
            Dict with batched and padded tensors:
                - gene_values: [batch_size, n_spots, max_gene_len]
                - gene_ids: [batch_size, n_spots, max_gene_len]
                - coords: [batch_size, n_spots, 2]
                - batch_labels: [batch_size, n_spots] (if provided)
                - organ_ids: [batch_size] (if provided)
        """
        padded_gene_values = []
        padded_gene_ids = []
        coords_list = []
        batch_labels_list = []
        organ_ids_list = []
        log1p_total_counts_list = []

        for sample in batch:
            gene_values = sample['gene_values']  # [n_spots, n_genes]
            gene_ids = sample['gene_ids']        # [n_spots, n_genes]
            coords = sample['coords']            # [n_spots, 2]

            if 'batch_label' in sample and sample['batch_label'] is not None:
                batch_labels_list.append(sample['batch_label'])

            if 'organ_id' in sample:
                organ_ids_list.append(sample['organ_id'])

            if 'log1p_total_counts' in sample:
                log1p_total_counts_list.append(sample['log1p_total_counts'])

            # Use _sample_or_truncate_plus_pad to handle padding, sampling, and truncation
            gene_ids, gene_values = self._sample_or_truncate_plus_pad(gene_ids, gene_values, self.max_gene_len)

            padded_gene_values.append(gene_values)
            padded_gene_ids.append(gene_ids)
            coords_list.append(coords)

        # Stack into batch
        output = {
            'gene_values': torch.stack(padded_gene_values, dim=0),  # [B, N, G]
            'gene_ids': torch.stack(padded_gene_ids, dim=0),        # [B, N, G]
            'coords': torch.stack(coords_list, dim=0),              # [B, N, 2]
        }
        if batch_labels_list:
            output['batch_labels'] = torch.stack(batch_labels_list, dim=0).long()  # [B, N]

        if organ_ids_list:
            output['organ_ids'] = torch.tensor(organ_ids_list, dtype=torch.long)  # [B]

        if log1p_total_counts_list:
            output['log1p_total_counts'] = torch.stack(log1p_total_counts_list, dim=0)  # [B, N]

        return output

    def _pad(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad gene_ids/gene_values to max_gene_len.
        """
        gene_ids = torch.cat([gene_ids, 
                              torch.full((gene_ids.shape[0], 
                                        self.max_gene_len - gene_ids.shape[1]), 
                                        self.pad_token_id, 
                                        dtype=gene_ids.dtype, 
                                        device=gene_ids.device)], dim=1)

        gene_values = torch.cat([gene_values, 
                                 torch.full((gene_values.shape[0],
                                  self.max_gene_len - gene_values.shape[1]), 
                                  self.pad_value, 
                                  dtype=gene_values.dtype, 
                                  device=gene_values.device)], dim=1)

        return gene_ids, gene_values

    def _sample_or_truncate_plus_pad(self, gene_ids: torch.Tensor, gene_values: torch.Tensor, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample or truncate and pad gene_ids and gene_values to max_length.

        Args:
            gene_ids: [n_spots, n_genes] tensor
            gene_values: [n_spots, n_genes] tensor
            max_length: Target number of genes

        Returns:
            Tuple of (gene_ids, gene_values) with shape [n_spots, max_length]
        """
        n_spots, n_genes = gene_ids.shape

        if n_genes == max_length:
            return gene_ids, gene_values
        if n_genes < max_length:
            return self._pad(gene_ids, gene_values)

        if self.gene_sampling:
            epsilon = 1e-6
            weights = torch.where(gene_values > 0, 1.0, epsilon)  # [n_spots, n_genes]

            # Batch sampling with multinomial
            if self.gene_sampling_seed is not None:
                g = torch.Generator(device=gene_ids.device)
                g.manual_seed(self.gene_sampling_seed)
                indices = torch.multinomial(weights, max_length, replacement=False, generator=g)
            else:
                indices = torch.multinomial(weights, max_length, replacement=False)  # [n_spots, max_length]

            # Batch gather
            new_gene_ids = torch.gather(gene_ids, 1, indices)
            new_gene_values = torch.gather(gene_values, 1, indices)

            return new_gene_ids, new_gene_values
        else:
            return gene_ids[:, :max_length], gene_values[:, :max_length]
