"""Canonical global RNG seeding for deterministic reproducibility."""
import numpy as np


def set_global_seed(seed):
    """Seed numpy, torch, and torch.cuda (if available) for deterministic reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
