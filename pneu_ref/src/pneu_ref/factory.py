"""Backward-compatible pressure-reference construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pneu_ref.feasible_random_ref import FeasibleRandomRef
from pneu_ref.joint_reachable_ref import JointReachableRandomRef
from pneu_ref.optimized_trajectory_ref import OptimizedTrajectoryBankRef
from pneu_ref.trajectory_ref_nd import TrajectoryBankRefND
from pneu_ref.random_ref import RandomRef
from pneu_ref.constant_ref import ConstantPressureRef


def make_pressure_reference(spec: Mapping[str, Any] | None):
    """Build a pressure reference from either old or new config syntax.

    Old model configs store the keyword arguments directly under ``rnd_ref``.
    New configs use ``reference: {kind: ..., kwargs: ...}``.
    """
    if spec is None:
        return RandomRef()

    config = dict(spec)
    if "kind" not in config:
        return RandomRef(**config)

    kind = str(config["kind"]).strip().lower().replace("-", "_")
    kwargs = dict(config.get("kwargs", {}))
    if kind in {"random", "legacy_random"}:
        return RandomRef(**kwargs)
    if kind in {"feasible_random", "rate_limited_random"}:
        return FeasibleRandomRef(**kwargs)
    if kind in {"joint_reachable_random", "joint_reachable"}:
        return JointReachableRandomRef(**kwargs)
    if kind in {"constant", "constant_pressure", "hold"}:
        return ConstantPressureRef(**kwargs)
    if kind in {"optimized_trajectory_bank", "simulator_trajectory_bank", "trajectory_bank"}:
        return OptimizedTrajectoryBankRef(**kwargs)
    raise ValueError(f"Unknown pressure reference kind: {config['kind']}")


def make_nd_reference(spec: Mapping[str, Any]) -> Any:
    """Build a fixed-dimensional trajectory reference for environments such as lib4."""
    config = dict(spec)
    kind = str(config.get("kind", "")).strip().lower().replace("-", "_")
    kwargs = dict(config.get("kwargs", {}))
    if kind in {"trajectory_bank_nd", "feasible_trajectory_bank_nd", "lib4_trajectory_bank"}:
        return TrajectoryBankRefND(**kwargs)
    raise ValueError(f"Unknown ND reference kind: {config.get('kind')}")


def reference_spec_from_model_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Select new ``reference`` config, falling back to legacy ``rnd_ref``."""
    if config.get("reference") is not None:
        return config["reference"]
    if config.get("rnd_ref") is not None:
        return config["rnd_ref"]
    return {}
