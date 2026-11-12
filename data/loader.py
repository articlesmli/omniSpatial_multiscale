"""
data/loader.py
Multimodal AnnData, Spatial, and Structural Perturbation Dataset Loader
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import anndata as ad
import scanpy as sc
from scipy.sparse import issparse

class MultimodalPerturbationDataset(Dataset):
    def __init__(self, h5ad_path: str, docking_scores_path: str = None, n_hvgs: int = 2000):
        """
        Args:
            h5ad_path (str): Path to the AnnData (.h5ad) file.
            docking_scores_path (str, optional): Path to precomputed AutoDock Vina binding scores (.npy or .pt).
            n_hvgs (int): Number of highly variable genes to select.
        """
        print(f"Loading AnnData from {h5ad_path}...")
        self.adata = ad.read_h5ad(h5ad_path)
        
        # Preprocess transcriptomics if not already log-normalized
        if 'hvg' not in self.adata.var:
            print(f"Preprocessing AnnData: selecting top {n_hvgs} highly variable genes...")
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_hvgs)
            self.adata = self.adata[:, self.adata.var['highly_variable']]

        # Extract gene expression matrix (X)
        X = self.adata.X
        self.expr_matrix = torch.tensor(X.toarray() if issparse(X) else X, dtype=torch.float32)
        
        # 1. Spatial coordinates fallback
        if 'spatial' in self.adata.obsm:
            self.spatial_coords = torch.tensor(self.adata.obsm['spatial'], dtype=torch.float32)
        else:
            # Zero tensor fallback for suspension data (e.g., standard Perturb-seq)
            self.spatial_coords = torch.zeros((self.adata.n_obs, 2), dtype=torch.float32)

        # 2. Cell morphology features fallback
        if 'morphology' in self.adata.obsm:
            self.morphology = torch.tensor(self.adata.obsm['morphology'], dtype=torch.float32)
        else:
            # Zero tensor fallback for missing imaging features
            self.morphology = torch.zeros((self.adata.n_obs, 128), dtype=torch.float32)

        # 3. Perturbation labels mapping
        if 'perturbation' in self.adata.obs:
            pert_labels = self.adata.obs['perturbation'].astype('category')
            self.pert_classes = pert_labels.cat.categories.tolist()
            self.pert_ids = torch.tensor(pert_labels.cat.codes.values, dtype=torch.long)
        else:
            # Default control assignment if perturbation column is missing
            self.pert_classes = ['control']
            self.pert_ids = torch.zeros(self.adata.n_obs, dtype=torch.long)

        # 4. Optional Binding / Docking affinity scores (AutoDock Vina integration)
        if docking_scores_path and os.path.exists(docking_scores_path):
            self.docking_scores = torch.tensor(np.load(docking_scores_path), dtype=torch.float32)
        else:
            self.docking_scores = torch.zeros((self.adata.n_obs, 1), dtype=torch.float32)

        print(f"Dataset initialized successfully! Total cells: {self.adata.n_obs}, Genes: {self.adata.shape[1]}")

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        return {
            "gene_expression": self.expr_matrix[idx],
            "spatial": self.spatial_coords[idx],
            "morphology": self.morphology[idx],
            "perturb_id": self.pert_ids[idx],
            "pert_id": self.pert_ids[idx],
            "docking_score": self.docking_scores[idx]
        }
