from __future__ import annotations

from pathlib import Path
from typing import Any

from pneu_env.sim4 import PneuSim
from pneu_utils.utils import get_pkg_path


class PneuPred(PneuSim):
    """lib4 predictor wrapper.

    This mirrors :class:`pneu_env.sim4.PneuSim` but loads the independent
    predictor shared object so MPObs rollouts do not mutate the observation
    simulator state.
    """

    def __init__(
        self,
        *args: Any,
        lib_path: str | Path | None = None,
        **kwargs: Any,
    ):
        if lib_path is None:
            env_pkg_path = Path(get_pkg_path("pneu_env"))
            lib_path = env_pkg_path / "src/pneu_env/lib4/libpneumatic_simulator_pred.so"
        super().__init__(*args, lib_path=lib_path, **kwargs)
