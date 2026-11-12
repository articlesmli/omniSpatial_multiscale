import omniSpatial_multiscale
import torch
import scanpy
import rdkit
import pytest

def test_imports():
    assert omniSpatial_multiscale is not None
    assert torch.__version__ is not None
    assert scanpy.__version__ is not None
