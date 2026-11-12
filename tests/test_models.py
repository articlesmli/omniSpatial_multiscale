import torch
import pytest

from models.transformer import MaskedGeneTransformer
from models.spatial_gnn import SpatialCrossAttentionFusion
from models.perturbation import ConditionalLatentDiffusionDenoiser

def test_masked_gene_transformer():
    batch_size = 4
    num_genes = 100
    d_model = 64
    
    # Initialize model
    model = MaskedGeneTransformer(num_genes=num_genes, d_model=d_model, n_heads=2, num_layers=1)
    model.eval()
    
    # Dummy input gene expression
    x = torch.randn(batch_size, num_genes)
    
    # Forward pass
    with torch.no_grad():
        out = model(x)
        
    # Assertions
    assert out.shape == (batch_size, d_model), f"Expected shape ({batch_size}, {d_model}), got {out.shape}"
    assert not torch.isnan(out).any(), "Transformer output contains NaNs"


def test_spatial_cross_attention_fusion():
    batch_size = 4
    d_model = 64
    morph_dim = 32
    struct_dim = 16
    
    # Initialize model
    fusion_model = SpatialCrossAttentionFusion(d_model=d_model, morph_dim=morph_dim, struct_dim=struct_dim)
    fusion_model.eval()
    
    # Dummy inputs
    cell_embeddings = torch.randn(batch_size, d_model)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    morphology = torch.randn(batch_size, morph_dim)
    structural_features = torch.randn(batch_size, struct_dim)
    
    # Forward pass
    with torch.no_grad():
        out = fusion_model(cell_embeddings, edge_index, morphology, structural_features)
        
    # Assertions
    assert out.shape == (batch_size, d_model), f"Expected shape ({batch_size}, {d_model}), got {out.shape}"
    assert not torch.isnan(out).any(), "Spatial fusion output contains NaNs"


def test_conditional_latent_diffusion_denoiser():
    batch_size = 4
    latent_dim = 64
    num_perturbations = 10
    
    # Initialize model
    denoiser = ConditionalLatentDiffusionDenoiser(latent_dim=latent_dim, num_perturbations=num_perturbations)
    denoiser.eval()
    
    # Dummy inputs
    z_t = torch.randn(batch_size, latent_dim)
    z_ctrl = torch.randn(batch_size, latent_dim)
    perturb_ids = torch.randint(0, num_perturbations, (batch_size,))
    timesteps = torch.randint(0, 100, (batch_size,))
    
    # Forward pass
    with torch.no_grad():
        pred_noise = denoiser(z_t, z_ctrl, perturb_ids, timesteps)
        
    # Assertions
    assert pred_noise.shape == (batch_size, latent_dim), f"Expected shape ({batch_size}, {latent_dim}), got {pred_noise.shape}"
    assert not torch.isnan(pred_noise).any(), "Diffusion denoiser output contains NaNs"