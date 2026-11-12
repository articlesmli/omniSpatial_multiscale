"""
data/metabolic_models/load_model.py
Utility to load SBML metabolic models for COBRApy flux balance analysis.
"""

import os
import cobra
from cobra.io import read_sbml_model, load_model

def get_metabolic_model(model_name_or_path: str = "textbook"):
    """
    Loads an organism-specific metabolic model via COBRApy.
    
    Args:
        model_name_or_path (str): Either a built-in COBRApy name (e.g., 'textbook' for E. coli core) 
                                   or a path to a custom SBML/XML file.
    Returns:
        cobra.Model: Loaded metabolic model object.
    """
    if os.path.exists(model_name_or_path):
        print(f"Loading custom SBML model from {model_name_or_path}...")
        model = read_sbml_model(model_name_or_path)
    else:
        try:
            print(f"Loading built-in COBRApy model: '{model_name_or_path}'...")
            model = load_model(model_name_or_path)
        except Exception as e:
            print(f"Could not load built-in model '{model_name_or_path}'. Falling back to E. coli core model. Error: {e}")
            model = load_model("textbook")
            
    print(f"Model loaded: {model.id} | Reactions: {len(model.reactions)} | Metabolites: {len(model.metabolites)}")
    return model

if __name__ == "__main__":
    # Test loading the default textbook model
    model = get_metabolic_model("textbook")
    solution = model.optimize()
    print(f"Optimal Growth Rate (Objective Value): {solution.objective_value:.4f}")
