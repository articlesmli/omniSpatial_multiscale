"""
scripts/fine_tune.py
Joint Multimodal Fine-Tuning Script for OmniSpatial-MultiScale
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from data.loader import MultimodalPerturbationDataset
from models.transformer import MaskedGeneTransformer
from models.spatial_gnn import SpatialCrossAttentionFusion
from models.structural_bridge import StructuralFeatureBridge
from models.docking_score import DockingScoreEncoder
from models.perturbation import ConditionalLatentDiffusionDenoiser

def train_multimodal_diffusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset
    h5ad_path = "perturbation_data.h5ad"
    if not os.path.exists(h5ad_path):
        print(f"Warning: {h5ad_path} not found. Please ensure dataset is prepared.")
        return

    dataset = MultimodalPerturbationDataset(h5ad_path=h5ad_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Instantiate Sub-modules
    transformer = MaskedGeneTransformer(num_genes=dataset.adata.shape[1], d_model=256).to(device)
    spatial_fusion = SpatialCrossAttentionFusion(d_model=256, morph_dim=128, struct_dim=64).to(device)
    struct_bridge = StructuralFeatureBridge(output_dim=64).to(device)
    docking_encoder = DockingScoreEncoder(input_dim=1, output_dim=64).to(device)
    denoiser = ConditionalLatentDiffusionDenoiser(latent_dim=256, num_perturbations=len(dataset.pert_classes)).to(device)

    optimizer = torch.optim.Adam(
        list(transformer.parameters()) + 
        list(spatial_fusion.parameters()) + 
        list(struct_bridge.parameters()) + 
        list(docking_encoder.parameters()) + 
        list(denoiser.parameters()), 
        lr=1e-4
    )
    mse_loss = nn.MSELoss()

    print("Starting multi-scale fine-tuning loop...")
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            expr = batch["gene_expression"].to(device)
            spatial = batch["spatial"].to(device)
            morph = batch["morphology"].to(device)
            pert_ids = batch["perturb_id"].to(device)
            docking = batch["docking_score"].to(device)

            optimizer.zero_grad()

            # Transcriptomic Foundation Backbone
            z_ctrl = transformer(expr)

            # Construct dummy k-NN edge index for PyG batching (simple chain/self-loop for demo stub)
            batch_size = expr.size(0)
            edge_index = torch.stack([
                torch.arange(batch_size, device=device),
                torch.arange(batch_size, device=device)
            ], dim=0)

            # Structural & Docking Encoders
            dummy_struct_desc = torch.zeros((batch_size, 32), device=device)
            struct_feat = struct_bridge(dummy_struct_desc)
            dock_feat = docking_encoder(docking)

            # Spatial & Multimodal Fusion
            fused_ctrl = spatial_fusion(z_ctrl, edge_index, morph, struct_feat)

            # Latent Diffusion Training Step
            timesteps = torch.randint(0, 100, (batch_size,), device=device).long()
            noise = torch.randn_like(fused_ctrl)
            # Simulated forward diffusion step
            z_t = fused_ctrl + noise

            # Predict noise / denoise
            pred_noise = denoiser(z_t, fused_ctrl, pert_ids, timesteps)
            loss = mse_loss(pred_noise, noise)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss / len(dataloader):.4f}")

    # Save trained checkpoint
    torch.save({
        'transformer': transformer.state_dict(),
        'spatial_fusion': spatial_fusion.state_dict(),
        'denoiser': denoiser.state_dict()
    }, "omnispatial_multiscale_finetuned.pth")
    print("Fine-tuning complete! Model saved to omnispatial_multiscale_finetuned.pth")

if __name__ == "__main__":
    train_multimodal_diffusion()
