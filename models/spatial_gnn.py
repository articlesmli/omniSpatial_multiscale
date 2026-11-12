"""
models/spatial_gnn.py
Spatial k-NN Graph Convolutional Network & Cross-Attention Multimodal Fusion
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

class SpatialCrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 256, morph_dim: int = 128, struct_dim: int = 64):
        super().__init__()
        self.gcn = GCNConv(d_model, d_model)
        self.morph_proj = nn.Linear(morph_dim, d_model)
        self.struct_proj = nn.Linear(struct_dim, d_model)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, cell_embeddings: torch.Tensor, edge_index: torch.Tensor, 
                morphology: torch.Tensor, structural_features: torch.Tensor) -> torch.Tensor:
        """
        Fuses transcriptomic cell embeddings, spatial graph context, morphology, and structural features.
        """
        # 1. Spatial GCN neighborhood aggregation
        gcn_out = self.gcn(cell_embeddings, edge_index)
        
        # 2. Project auxiliary modalities
        morph_embed = self.morph_proj(morphology).unsqueeze(1) # [B, 1, d_model]
        struct_embed = self.struct_proj(structural_features).unsqueeze(1) # [B, 1, d_model]
        base_embed = gcn_out.unsqueeze(1) # [B, 1, d_model]
        
        # Combine auxiliary features as key/value memory for cross-attention
        memory = torch.cat([morph_embed, struct_embed], dim=1)
        
        # 3. Multi-head Cross-Attention
        attn_out, _ = self.cross_attn(query=base_embed, key=memory, value=memory)
        fused = self.layer_norm(base_embed + attn_out).squeeze(1)
        
        return fused
