from __future__ import annotations

from pathlib import Path
from typing import Any

from pneu_env.sim5 import PneuSim
from pneu_utils.utils import get_pkg_path


class PneuPred(PneuSim):
    def __init__(self, *args: Any, lib_path: str | Path | None = None, **kwargs: Any):
        if lib_path is None:
            lib_path = (
                Path(get_pkg_path("pneu_env"))
                / "src/pneu_env/lib4/libpneumatic_simulator_pred.so"
            )
        super().__init__(*args, lib_path=lib_path, **kwargs)

