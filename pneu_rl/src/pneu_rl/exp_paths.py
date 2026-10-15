"""Canonical experiment-output paths.

Experiment outputs are grouped by the model that produced them:

    pneu_rl/exp/<model_name>/<experiment_name>/
"""

from __future__ import annotations

from pathlib import Path

from pneu_utils.utils import get_pkg_path


def exp_root() -> Path:
    return Path(get_pkg_path("pneu_rl")) / "exp"


def model_exp_dir(model_name: str, experiment_name: str) -> Path:
    """Return the canonical directory for one model experiment."""
    if not model_name or not experiment_name:
        raise ValueError("model_name and experiment_name must be non-empty")
    return exp_root() / str(model_name) / str(experiment_name)


def ensure_model_exp_dir(model_name: str, experiment_name: str) -> Path:
    output_dir = model_exp_dir(model_name, experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_experiment_file(experiment_name: str, filename: str | None = None) -> Path:
    """Find a file in the new layout, with legacy flat-layout fallback."""
    filename = filename or experiment_name
    root = exp_root()
    candidates = [root / experiment_name / filename]
    candidates.extend(root.glob(f"*/{experiment_name}/{filename}"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Experiment file not found: {experiment_name}/{filename} under {root}"
    )
