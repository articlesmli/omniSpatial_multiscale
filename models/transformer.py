"""
models/transformer.py
Masked Gene Modeling (MGM) Foundation Backbone for Single-Cell RNA-seq
"""

import torch
import torch.nn as nn

class MaskedGeneTransformer(nn.Module):
    def __init__(self, num_genes: int = 2000, d_model: int = 256, n_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.gene_embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Gene expression tensor of shape [batch_size, num_genes]
            mask: Optional binary mask [batch_size, num_genes] where 1 = masked
        Returns:
            Embeddings or reconstructed gene expression of shape [batch_size, num_genes]
        """
        b, n = x.shape
        # Reshape to sequence of gene tokens [batch_size, num_genes, 1]
        h = self.gene_embedding(x.unsqueeze(-1))
        
        # Apply transformer encoder
        h_encoded = self.transformer(h) # [batch_size, num_genes, d_model]
        
        # Pool across genes to get cell-level representation [batch_size, d_model]
        cell_embedding = h_encoded.mean(dim=1)
        return cell_embedding
