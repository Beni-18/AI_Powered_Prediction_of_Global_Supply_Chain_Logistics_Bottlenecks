"""ROLE: ML ENGINEER  |  COMMON MODELKIT
Generates and persists standardized model cards (JSON) for auditability and debate evidence.

Owner: Role 1 - ML Engineer
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Optional


def save_model_card(
    output_dir: Path | str,
    name: str,
    version: str,
    features: list[str],
    train_window: dict[str, Any] | str,
    metrics_on_val: dict[str, Any],
    seed: int = 42,
    hyperparameters: Optional[dict[str, Any]] = None,
    notes: Optional[str] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> Path:
    """Save an honest, code-computed model card to output_dir/model_card.json."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    card_file = output_path / "model_card.json"

    # Step 1: Capture package versions automatically so anyone can reproduce the environment.
    library_versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "torch", "shap"]:
        try:
            mod = __import__(pkg)
            library_versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            library_versions[pkg] = "not_installed"

    # Step 2: Assemble the structured metadata document.
    card_data = {
        "model_name": name,
        "model_version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "train_window": train_window,
        "feature_count": len(features),
        "features": list(features),
        "metrics_on_val": metrics_on_val,
        "hyperparameters": hyperparameters or {},
        "library_versions": library_versions,
        "notes": notes or "CASCADE Role 1 Model Handoff",
    }

    if extra_fields:
        card_data.update(extra_fields)

    # Step 3: Write out as nicely formatted JSON.
    with open(card_file, "w", encoding="utf-8") as f:
        json.dump(card_data, f, indent=2)

    return card_file
