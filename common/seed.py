"""One global random seed so every teammate gets the same numbers on their own laptop."""

import os
import random

SEED = 42  # Never change this without a team vote; reported numbers depend on it.


def set_global_seed(seed: int = SEED) -> int:
    """Seed Python, NumPy and (if installed) PyTorch. Call this first thing in every script."""
    # Python's own random generator.
    random.seed(seed)
    # Makes hash-based ordering repeatable between runs.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy is optional at import time so this file also works in a bare environment.
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch is only present from Day 2 onwards.
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed
