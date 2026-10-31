"""ROLE: ML ENGINEER  |  COMMON MODELKIT
Reusable machine learning and deep learning utilities for all stages.
Owner: Role 1 - ML Engineer
"""

from .feature_guard import assert_features_allowed, get_feature_policy
from .model_card import save_model_card
from .pipeline_stage01 import Stage01Pipeline
from .profiling import latency_probe
from .split_loader import load_split

__all__ = [
    "Stage01Pipeline",
    "assert_features_allowed",
    "get_feature_policy",
    "save_model_card",
    "latency_probe",
    "load_split",
    "load_canonical_nodes",
    "load_adjacency",
    "get_shuffled_adjacency",
    "LSTMForecaster",
    "TransformerForecaster",
    "GraphForecaster",
    "PinballLoss",
    "build_model",
    "count_parameters",
    "save_model",
    "load_model",
    "save_torch_model",
    "load_torch_model",
    "HORIZONS",
    "TARGETS",
    "QUANTILES",
]

# The deep-learning and graph names need PyTorch, which only Stage 02 uses. They are loaded on first use, so
# Stage 01 (and CI, which does not install torch) can import this package without PyTorch installed.
_LAZY = {
    "dl_models": {"HORIZONS", "QUANTILES", "TARGETS", "LSTMForecaster", "TransformerForecaster", "GraphForecaster",
                  "PinballLoss", "build_model", "count_parameters", "save_model", "load_model",
                  "load_torch_model", "save_torch_model"},
    "graph_utils": {"get_shuffled_adjacency", "load_adjacency", "load_canonical_nodes"},
}


def __getattr__(name):
    """Import a torch-dependent name from its module only when someone asks for it."""
    for module_name, names in _LAZY.items():
        if name in names:
            import importlib

            return getattr(importlib.import_module(f".{module_name}", __name__), name)
    raise AttributeError(f"module 'common.modelkit' has no attribute '{name}'")
