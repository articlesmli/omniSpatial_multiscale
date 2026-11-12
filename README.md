# A Multi-Scale Biological Digital Twin & Generative Framework

## Overview

**OmniSpatial-MultiScale** is an open-source biological AI simulator that bridges atomic-level biophysics, molecular property prediction, protein structure modeling, molecular dynamics, spatial single-cell perturbation forecasting, and genome-scale systems metabolism[cite: 1]. 

By combining **Boltz-1 / AlphaFold** 3D structural modeling, **AutoDock Vina** docking, **GROMACS** dynamics, **COBRApy** metabolic flux analysis, and your core **PyTorch / PyTorch Geometric** spatial latent diffusion engine, this framework models biological responses across multiple physical scales—from angstroms to whole-tissue neighborhoods[cite: 1].

---

## Architecture Overview

```text
[ Sequence / FastA ]    ──► Prodigal, HMMER, DeepSig ──────────┐
                                                               │
[ 3D Structures ]       ──► Boltz-1 / AlphaFold   ─────────────┼─► Structural & Binding ──┐
                                                               │   Score Integration      │
[ Small Molecules ]     ──► RDKit & AutoDock Vina / GROMACS   ─┘                          │
                                                                                          ▼ 
[ Single-Cell RNA-seq ] ──► Masked Gene Transformer ────────┐                           
                                                            ├─► Cross-Attention ──► LayerNorm ──► Conditional Latent ──► Predicted Perturbed
[ Spatial Coordinates ] ──► Spatial GNN (PyG k-NN)        ──┤   Fusion Head         (z_ctrl)      Diffusion Denoiser       Cell State & Metabolism
                                                            │                                                        │
[ Cell Morphology ]     ──► Vision Features               ──┘                                                        ▼
                                                                                                                [ COBRApy Metabolic Flux ]

```

The pipeline operates across modular multi-scale stages:

1. **Structural & Biophysics Bridge (`models/structural_bridge.py`, `models/docking_score.py`):** Ingests protein structures from **Boltz-1 / AlphaFold**, computes small-molecule binding energies via **AutoDock Vina**, and validates conformational stability using **GROMACS**.


2. **Foundation Backbone (`models/transformer.py`):** Encodes binned gene expression tokens and applies masked language modeling objectives to extract clean representations.


3. **Spatial Graph Integration (`models/spatial_gnn.py`):** Constructs local k-NN spatial neighborhood graphs and aligns transcriptomic, morphological, and structural features via multi-head cross-attention (`SpatialCrossAttentionFusion`).


4. **Conditional Perturbation Head (`models/perturbation.py`):** Concatenates noisy target states (`z_t`), multimodal untreated control embeddings (`z_ctrl`), perturbation/docking embeddings (`perturb_ids`), and diffusion timesteps (`t`) to predict reverse denoising steps.


5. **Systems Metabolism (`models/cobra_flux.py`, `scripts/metabolic_simulation.py`):** Translates predicted single-cell expression profiles into genome-scale metabolic flux balance models using **COBRApy** to verify phenotypic viability and cellular growth rates.



---

## Core Capabilities

* **Multi-Scale Simulation (Angstroms to Tissues):** Seamlessly bridges atomic-level ligand binding and 3D protein folding with single-cell spatial tissue neighborhoods.
* **In Silico Drug & CRISPR Screening:** Feed untreated control cells and drug/genetic edits into the reverse DDPM diffusion engine to predict post-treatment cellular states.
* **Structural Biophysics & Binding (`AutoDock Vina` & `GROMACS`):** Computes binding free energies ($\Delta G$) and validates conformational stability for small-molecule candidates.
* **Metabolic Phenotyping (`COBRApy`):** Evaluates whether predicted single-cell transcriptomic shifts maintain cellular viability, growth rates, and metabolic fluxes via Flux Balance Analysis (FBA).
* **Automated CI/CD & Containerization:** Fully containerized with Docker (`Dockerfile`) and integrated with GitHub Actions (`ci_cd.yaml`) for automated testing across Linux environments.



---

## Practical Use Cases

* **Rational Drug Discovery:** Screen virtual chemical libraries against diseased spatial tissue data to identify candidates that shift cells toward a healthy phenotypic state.
* **Target Identification & CRISPR Engineering:** Simulate knocking out specific genes within tumor microenvironments to evaluate downstream transcriptomic and metabolic consequences.
* **Pre-Clinical Prioritization:** Filter out unstable drug-target complexes using structural dynamics before committing expensive time and budget to wet-lab experiments.

---

## Data Sources & Preparation

OmniSpatial-MultiScale supports standard single-cell and spatial omics file formats:

* **Foundation Single-Cell Data:** Uses 10x Genomics PBMC datasets via `scvi-tools` for single-cell modeling and perturbation simulation.


* **Perturbation Datasets (`perturbation_data.h5ad`):** Ingests control-versus-stimulated single-cell AnnData files containing expression matrices and `pert_id` / `condition` observation columns.


* **Metabolic Models:** Supports SBML-format genome-scale metabolic models (e.g., BiGG `iJO1366` or human Recon models) for COBRApy flux balance analysis.



---

## Dataset & Model Setup

### 1. Prepare Perturbation Dataset (`perturbation_data.h5ad`)

Download real PBMC data via `scvi-tools` and format it for the perturbation pipeline:

```bash
python -c "
import scvi
import numpy as np

print('Downloading real 10x PBMC dataset via scvi-tools...')
adata = scvi.data.pbmc_dataset(save_path='./data')

adata.obs['condition'] = 'ctrl'
np.random.seed(42)
mask = np.random.rand(adata.n_obs) > 0.5
adata.obs.loc[mask, 'condition'] = 'stim'
adata.obs['pert_id'] = (adata.obs['condition'] == 'stim').astype(int)

adata.write('perturbation_data.h5ad')
print('Successfully saved perturbation_data.h5ad:', adata.shape)
"

```

### 2. Download Genome-Scale Metabolic Model (`iJO1366`)

Download the BiGG SBML model for COBRApy metabolic simulation:

```bash
mkdir -p data/metabolic_models
curl -L -o data/metabolic_models/ijo1366.xml [http://bigg.ucsd.edu/static/models/iJO1366.xml](http://bigg.ucsd.edu/static/models/iJO1366.xml)

```

---

## Project Structure

```text
omniSpatial_multiscale/
├── .github/
│   └── workflows/
│       └── ci_cd.yaml                 # Automated testing and CI/CD pipeline
├── data/
│   ├── loader.py                      # Multimodal AnnData, spatial & structural loaders
│   └── metabolic_models/              # Organism-specific SBML / COBRApy models (e.g. ijo1366.xml)
├── models/
│   ├── transformer.py                 # Masked Gene Modeling backbone
│   ├── spatial_gnn.py                 # Spatial k-NN GCN & cross-attention fusion
│   ├── structural_bridge.py           # Boltz-1 / AlphaFold structure feature extractor
│   ├── docking_score.py               # AutoDock Vina & GROMACS descriptor integration
│   ├── perturbation.py                # Control-conditioned Latent Diffusion head
│   └── cobra_flux.py                  # COBRApy metabolic flux balance integration
├── scripts/
│   ├── run_docking_pipeline.py        # Automated Vina docking & MD stability checks
│   ├── structure_prediction.py        # Boltz-1 / AlphaFold structure generation
│   ├── fine_tune.py                   # Multi-scale joint fine-tuning loop
│   ├── evaluate.py                    # Reverse DDPM sampling & correlation metrics
│   └── metabolic_simulation.py        # FBA phenotypic viability screening
├── tests/
│   ├── test_imports.py                # Core import & package tests
│   ├── test_models.py                 # Architecture unit tests
│   ├── test_docking.py                # Vina and structure feature tests
│   └── test_diffusion.py              # Diffusion conditioning & overfit tests
├── Dockerfile                         # Production container definition
├── environment.yaml                   # Conda environment dependency spec
├── LICENSE                            # MIT License
├── README.md
├── requirements.txt
└── setup.py

```

---

## Installation & Environment Setup

### 1. System Prerequisites (External Binaries)

Install **AutoDock Vina** and **GROMACS**:

* **Ubuntu / Debian Linux:**

```bash
sudo apt update
sudo apt install autodock-vina gromacs

```

* **macOS (via Homebrew):**

```bash
brew install autodock-vina
brew install gromacs

```

### 2. Conda / Mamba Environment Setup

Create and activate the managed Conda environment (`omnispatial_env`) using `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate omnispatial_env

```

---

## Quick Start Pipeline

### 1. Multi-Scale Spatial Fine-Tuning

```bash
PYTHONPATH=. python -u scripts/fine_tune.py --data perturbation_data.h5ad

```

### 2. DDPM Evaluation & Sampling

```bash
PYTHONPATH=. python -u scripts/evaluate.py

```

### 3. Metabolic Flux Phenotyping (`COBRApy`)

```bash
PYTHONPATH=. python -u scripts/metabolic_simulation.py --model data/metabolic_models/ijo1366.xml --predictions perturbation_data.h5ad

```

---

## License

This project is open-source and available under the terms of the [MIT License].

