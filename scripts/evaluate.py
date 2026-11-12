"""
scripts/evaluate.py
Reverse DDPM Sampling & Correlation Metrics Evaluation Script
"""

import os
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from data.loader import MultimodalPerturbationDataset
from models.transformer import MaskedGeneTransformer
from models.spatial_gnn import SpatialCrossAttentionFusion
from models.perturbation import ConditionalLatentDiffusionDenoiser

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    h5ad_path = "perturbation_data.h5ad"
    if not os.path.exists(h5ad_path):
        print(f"Dataset {h5ad_path} not found. Skipping evaluation.")
        return

    dataset = MultimodalPerturbationDataset(h5ad_path=h5ad_path)
    
    # Load model weights if available
    transformer = MaskedGeneTransformer(num_genes=dataset.adata.shape[1], d_model=256).to(device)
    spatial_fusion = SpatialCrossAttentionFusion(d_model=256, morph_dim=128, struct_dim=64).to(device)
    denoiser = ConditionalLatentDiffusionDenoiser(latent_dim=256, num_perturbations=len(dataset.pert_classes)).to(device)

    weights_path = "omnispatial_multiscale_finetuned.pth"
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        transformer.load_state_dict(checkpoint['transformer'])
        spatial_fusion.load_state_dict(checkpoint['spatial_fusion'])
        denoiser.load_state_dict(checkpoint['denoiser'])
        print("Loaded trained weights successfully.")
    else:
        print("Warning: Checkpoint not found. Running evaluation with initialized weights.")

    transformer.eval()
    spatial_fusion.eval()
    denoiser.eval()

    # Perform 100-step reverse DDPM sampling simulation on subset
    print("Running 100-step reverse DDPM sampling...")
    mse_list, pearson_list, spearman_list = [], [], []

    with torch.no_grad():
        for i in range(min(50, len(dataset))):
            sample = dataset[i]
            expr = sample["gene_expression"].unsqueeze(0).to(device)
            morph = sample["morphology"].unsqueeze(0).to(device)
            pert_id = sample["pert_id"].unsqueeze(0).to(device)

            z_ctrl = transformer(expr)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
            dummy_struct = torch.zeros((1, 64), device=device)
            fused_ctrl = spatial_fusion(z_ctrl, edge_index, morph, dummy_struct)

            # Reverse Diffusion Loop (100 steps simplified to final step demo)
            z_t = torch.randn_like(fused_ctrl)
            for t in reversed(range(100)):
                t_tensor = torch.tensor([t], device=device).long()
                pred_noise = denoiser(z_t, fused_ctrl, pert_id, t_tensor)
                z_t = z_t - 0.01 * pred_noise  # Simplified Euler step

            # Compute correlation metrics against control
            pred_vec = z_t.squeeze().cpu().numpy()
            ctrl_vec = fused_ctrl.squeeze().cpu().numpy()

            mse = np.mean((pred_vec - ctrl_vec) ** 2)
            p_corr, _ = pearsonr(pred_vec, ctrl_vec)
            s_corr, _ = spearmanr(pred_vec, ctrl_vec)

            mse_list.append(mse)
            pearson_list.append(p_corr if not np.isnan(p_corr) else 0.0)
            spearman_list.append(s_corr if not np.isnan(s_corr) else 0.0)

    print("\n--- Evaluation Results ---")
    print(f"Latent MSE: {np.mean(mse_list):.4f}")
    print(f"Mean Pearson Correlation: {np.mean(pearson_list):.4f}")
    print(f"Mean Spearman Correlation: {np.mean(spearman_list):.4f}")

if __name__ == "__main__":
    evaluate_model()
