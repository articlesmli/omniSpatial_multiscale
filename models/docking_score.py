"""
models/docking_score.py
AutoDock Vina Binding Affinity & GROMACS Dynamics Descriptor Integration
"""

import torch
import torch.nn as nn

class DockingScoreEncoder(nn.Module):
    def __init__(self, input_dim: int = 4, output_dim: int = 32):
        super().__init__()
        # Ingests AutoDock Vina binding energy (kcal/mol), RMSD, and GROMACS stability metrics
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, vina_metrics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vina_metrics: Tensor [batch_size, 4] (e.g., binding affinity, ligand efficiency, RMSD, MD energy drift)
        Returns:
            Embedded physics descriptors [batch_size, output_dim]
        """
        return self.net(vina_metrics)
