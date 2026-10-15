from __future__ import annotations

import numpy as np

from pneu_rl.sac_with_loss import SAC as SAC4


class SAC(SAC4):
    """SAC4 with diagnostics for five-reference residual control."""

    def _episode_diagnostics(self, epi: int, reward: float, infos: list[dict]) -> dict[str, float]:
        diag = super()._episode_diagnostics(epi, reward, infos)
        if not infos:
            return diag

        def values(key: str) -> np.ndarray:
            return np.asarray([info["obs"][key] for info in infos], dtype=np.float64)

        ch_pos = values("sen_pos")
        ch_neg = values("sen_neg")
        ref_ch_pos = values("ref_ch_pos")
        ref_ch_neg = values("ref_ch_neg")
        base_ctrl = np.asarray([info["obs"]["base_ctrl"] for info in infos], dtype=np.float64)
        residual = np.asarray([info["obs"]["residual_ctrl"] for info in infos], dtype=np.float64)
        diag.update(
            ch_pos_shortage_rmse=float(np.sqrt(np.mean(np.maximum(ref_ch_pos-ch_pos, 0.0) ** 2))),
            ch_neg_shortage_rmse=float(np.sqrt(np.mean(np.maximum(ch_neg-ref_ch_neg, 0.0) ** 2))),
            ch_pos_reserve_ok_ratio=float(np.mean(ch_pos >= ref_ch_pos)),
            ch_neg_reserve_ok_ratio=float(np.mean(ch_neg <= ref_ch_neg)),
            base_ctrl_mean=float(np.mean(base_ctrl)),
            residual_abs_mean=float(np.mean(np.abs(residual))),
        )
        return diag

