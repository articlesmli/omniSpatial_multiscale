"""
models/perturbation.py
Control-Conditioned Latent Diffusion Head for Single-Cell Perturbation Forecasting
"""

import torch
import torch.nn as nn

class ConditionalLatentDiffusionDenoiser(nn.Module):
    def __init__(self, latent_dim: int = 256, num_perturbations: int = 50):
        super().__init__()
        self.time_embed = nn.Embedding(100, 64)
        self.pert_embed = nn.Embedding(num_perturbations, 64)
        
        # Concatenated input: z_t (latent_dim) + z_ctrl (latent_dim) + time_embed (64) + pert_embed (64)
        total_input_dim = latent_dim * 2 + 64 + 64
        
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, z_t: torch.Tensor, z_ctrl: torch.Tensor, 
                perturb_ids: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Predicts noise / reverse denoising step for the latent diffusion process.
        """
        t_emb = self.time_embed(timesteps)
        p_emb = self.pert_embed(perturb_ids)
        
        # Concatenate all conditioning streams
        h = torch.cat([z_t, z_ctrl, t_emb, p_emb], dim=-1)
        return self.net(h)
