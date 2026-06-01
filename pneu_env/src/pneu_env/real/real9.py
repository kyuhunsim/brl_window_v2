from typing import Any, Dict

import numpy as np
import os
from pathlib import Path


from pneu_env.pid import ActuatorPressurePID
from pneu_env.real.real_act import PneuRealAct


STD_RHO = 1.20411831637462


def _lpm_to_mass_flow(flow_lpm: float) -> float:
    """Convert real flow sensor values [L/min] to the env9 mass-flow convention [kg/s]."""
    return float(flow_lpm) * STD_RHO / 60000.0


class PneuReal:
    """
    lib9/env9 인터페이스와 호환되는 Real 관측 래퍼.

    내부적으로 PneuRealAct(ctrl_act.json/obs_act.json 기반)를 사용하고,
    observe() 출력 형식을 sim9과 동일한
    [t, ch_pos, ch_neg, act_pos, act_neg]로 맞춘다.

    참고:
    - obs_act.json의 flowrate1~6은 현재 RT가 보내는 raw 채널 순서를 그대로 사용한다.
    - get_mass_flowrate_dict()의 flow1~flow6은 위 raw 채널을 [kg/s]로 변환한 값이다.
    - canonical alias는 sim9과 동일하게 사용한다:
      flow1=chamber pos valve, flow2=chamber neg valve, flow3=act pos in, flow4=act pos out,
      flow5=act neg in, flow6=act neg out
    """

    def __init__(
        self,
        freq: float = 50.0,
        scale: bool = True,
        init_pos_press: float = 101.325,
        init_neg_press: float = 101.325,
        init_act_pos_press: float = 101.325,
        init_act_neg_press: float = 101.325,
        **_: Any,
    ):
        self.freq = float(freq)
        self.scale = bool(scale)
        self.backend = PneuRealAct(freq=self.freq, scale=self.scale)
        self._step_count = 0
        self._last_time = 0.0
        self._raw_last_time: float | None = None
        self._time_origin: float | None = None
        self._stale_time_count = 0
        self._prev_press_vec: np.ndarray | None = None
        self._stale_obs_count = 0
        self._max_stale_obs_steps = int(
            os.getenv("PNEU_REAL9_MAX_STALE", os.getenv("PNEU_REAL8_MAX_STALE", "300"))
        )
        self._debug_every = int(
            os.getenv("PNEU_REAL9_DEBUG_EVERY", os.getenv("PNEU_REAL8_DEBUG_EVERY", "0"))
        )
        self.is_pid = False
        self.is_anti_windup = False
        self.obs = np.array(
            [
                init_pos_press,
                init_neg_press,
                init_act_pos_press,
                init_act_neg_press,
            ],
            dtype=np.float64,
        )
        self.set_init_press(
            init_pos_press,
            init_neg_press,
            init_act_pos_press,
            init_act_neg_press,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def _split_ctrl(self, ctrl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ctrl = np.asarray(ctrl, dtype=np.float64).reshape(-1).copy()
        if ctrl.size == 2:
            ctrl = np.array([ctrl[0], ctrl[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif ctrl.size != 6:
            raise ValueError(f"ctrl must be length 2 or 6, got {ctrl.size}")

        if not np.all(np.isfinite(ctrl)):
            ctrl = np.nan_to_num(ctrl, nan=0.0, posinf=1.0, neginf=0.0)
        ctrl = np.clip(ctrl, 0.0, 1.0)

        return ctrl[:2], ctrl[2:]

    def _flowrate_lpm_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (
            float(self.backend.flowrate1),
            float(self.backend.flowrate2),
            float(self.backend.flowrate3),
            float(self.backend.flowrate4),
            float(self.backend.flowrate5),
            float(self.backend.flowrate6),
        )

    def observe(
        self,
        ctrl: np.ndarray,
        goal: np.ndarray = np.array([101.325, 101.325], dtype=np.float64),
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        self._step_count += 1
        goal = np.asarray(goal, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(goal)):
            goal = np.nan_to_num(goal, nan=101.325, posinf=101.325, neginf=101.325)
        if goal.size >= 2:
            goal_posneg = goal[:2]
        else:
            goal_posneg = np.array([101.325, 101.325], dtype=np.float64)

        ctrl = np.asarray(ctrl, dtype=np.float64).reshape(-1).copy()
        if ctrl.size == 2:
            ctrl = np.array([ctrl[0], ctrl[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif ctrl.size != 6:
            raise ValueError(f"ctrl must be length 2 or 6, got {ctrl.size}")
        if not np.all(np.isfinite(ctrl)):
            ctrl = np.nan_to_num(ctrl, nan=0.0, posinf=1.0, neginf=0.0)

        if self.is_pid:
            pid_delta = self.pid.get_action(obs=self.obs[2:4], ref=goal_posneg)
            ctrl[2:6] += pid_delta
            if self.is_anti_windup:
                original_ctrl = ctrl.copy()

        ctrl = np.clip(ctrl, 0.0, 1.0)
        if self.is_anti_windup:
            self.pid.anti_windup(
                ctrl=original_ctrl[2:6],
                sat_ctrl=ctrl[2:6],
            )

        main_ctrl, act_ctrls = self._split_ctrl(ctrl)
        ctrl_unit = np.r_[main_ctrl, act_ctrls]
        if self.scale:
            # PneuRealAct currently maps bipolar input x to 0.1*x + 0.9 when
            # scale=True. Pre-compensate so the real command matches sim3's
            # trained valve command: 0.85 + 0.15*action.
            backend_ctrl = 1.5 * ctrl_unit - 0.5
        else:
            backend_ctrl = 2.0 * ctrl_unit - 1.0
        backend_ctrl = np.clip(backend_ctrl, -1.0, 1.0)
        _, raw_info = self.backend.observe(
            ctrl=backend_ctrl,
            goal=goal_posneg,
        )
        obs = raw_info["Observation"]

        dt = 1.0 / max(self.freq, 1e-6)
        raw_time = float(obs.get("time", np.nan))
        is_stale_or_invalid = (not np.isfinite(raw_time)) or (
            self._raw_last_time is not None and raw_time <= self._raw_last_time
        )
        if is_stale_or_invalid:
            self._stale_time_count += 1
            prev_raw_time = self._raw_last_time if self._raw_last_time is not None else 0.0
            raw_time = prev_raw_time + dt
            if self._stale_time_count in (50, 200, 500):
                print(
                    "[WARN] real9 observe time is stale. "
                    "Using local monotonic fallback time."
                )
        else:
            self._stale_time_count = 0

        self._raw_last_time = raw_time

        # Normalize external timestamps to elapsed seconds so env loop
        # termination works regardless of whether RT sends absolute or relative time.
        if self._time_origin is None:
            self._time_origin = raw_time
        time_value = raw_time - self._time_origin
        if time_value < self._last_time:
            time_value = self._last_time + dt
        self._last_time = time_value

        next_obs = np.array(
            [
                time_value,
                float(obs.get("pos_press", np.nan)),
                float(obs.get("neg_press", np.nan)),
                float(obs.get("act_pos_press", np.nan)),
                float(obs.get("act_neg_press", np.nan)),
            ],
            dtype=np.float64,
        )
        self.obs = next_obs[1:5].copy()

        press_vec = next_obs[1:5].copy()
        if np.all(np.isfinite(press_vec)):
            if self._prev_press_vec is not None and np.allclose(
                press_vec,
                self._prev_press_vec,
                rtol=0.0,
                atol=1e-9,
            ):
                self._stale_obs_count += 1
                if self._stale_obs_count in (100, 500):
                    print(
                        "[WARN] real9 pressure observation is unchanged for "
                        f"{self._stale_obs_count} steps. "
                        "Check RT/bridge receive loop."
                    )
                if self._max_stale_obs_steps > 0 and self._stale_obs_count >= self._max_stale_obs_steps:
                    raise RuntimeError(
                        "real9 observation stream appears stalled "
                        f"({self._stale_obs_count} unchanged steps). "
                        "Likely RT/bridge communication issue."
                    )
            else:
                self._stale_obs_count = 0
            self._prev_press_vec = press_vec

        if self._debug_every > 0 and self._step_count % self._debug_every == 0:
            print(
                "[DBG] real9 "
                f"step={self._step_count} "
                f"time={next_obs[0]:.3f} "
                f"ctrl=({main_ctrl[0]:.3f},{main_ctrl[1]:.3f}) "
                f"act=({act_ctrls[0]:.3f},{act_ctrls[1]:.3f},{act_ctrls[2]:.3f},{act_ctrls[3]:.3f}) "
                f"press=({next_obs[1]:.3f},{next_obs[2]:.3f},{next_obs[3]:.3f},{next_obs[4]:.3f}) "
                f"stale_obs={self._stale_obs_count}"
            )

        flowrate1_lpm, flowrate2_lpm, flowrate3_lpm, flowrate4_lpm, flowrate5_lpm, flowrate6_lpm = (
            self._flowrate_lpm_tuple()
        )

        observation_info = dict(
            curr_time=next_obs[0],
            sen_pos=next_obs[1],
            sen_neg=next_obs[2],
            sen_act_pos=next_obs[3],
            sen_act_neg=next_obs[4],
            P1_pos=next_obs[3],
            P1_neg=next_obs[4],
            ref_pos=float(goal_posneg[0]),
            ref_neg=float(goal_posneg[1]),
            ref_act_pos=float(obs.get("act_pos_ref", np.nan)),
            ref_act_neg=float(obs.get("act_neg_ref", np.nan)),
            ctrl_pos=float(obs.get("pos_ctrl", np.nan)),
            ctrl_neg=float(obs.get("neg_ctrl", np.nan)),
            ctrl_act_pos_in=float(obs.get("act_pos_ctrl1", np.nan)),
            ctrl_act_pos_out=float(obs.get("act_pos_ctrl2", np.nan)),
            ctrl_act_neg_in=float(obs.get("act_neg_ctrl1", np.nan)),
            ctrl_act_neg_out=float(obs.get("act_neg_ctrl2", np.nan)),
            act_pos_ctrl1=float(obs.get("act_pos_ctrl1", np.nan)),
            act_pos_ctrl2=float(obs.get("act_pos_ctrl2", np.nan)),
            act_neg_ctrl1=float(obs.get("act_neg_ctrl1", np.nan)),
            act_neg_ctrl2=float(obs.get("act_neg_ctrl2", np.nan)),
            flowrate1=flowrate1_lpm,
            flowrate2=flowrate2_lpm,
            flowrate3=flowrate3_lpm,
            flowrate4=flowrate4_lpm,
            flowrate5=flowrate5_lpm,
            flowrate6=flowrate6_lpm,
        )
        info = {
            "obs_w/o_noise": next_obs.copy(),
            "Observation": observation_info,
        }
        return next_obs, info

    def get_mass_flowrate_dict(self) -> Dict[str, float]:
        flowrate1_lpm, flowrate2_lpm, flowrate3_lpm, flowrate4_lpm, flowrate5_lpm, flowrate6_lpm = (
            self._flowrate_lpm_tuple()
        )

        flow1 = _lpm_to_mass_flow(flowrate1_lpm)
        flow2 = _lpm_to_mass_flow(flowrate2_lpm)
        flow3 = _lpm_to_mass_flow(flowrate3_lpm)
        flow4 = _lpm_to_mass_flow(flowrate4_lpm)
        flow5 = _lpm_to_mass_flow(flowrate5_lpm)
        flow6 = _lpm_to_mass_flow(flowrate6_lpm)

        return dict(
            flow1=flow1,
            flow2=flow2,
            flow3=flow3,
            flow4=flow4,
            flow5=flow5,
            flow6=flow6,
            chamber_pos_valve=flow1,
            act_pos_in=flow3,
            act_pos_out=flow4,
            chamber_neg_valve=flow2,
            act_neg_out=flow6,
            act_neg_in=flow5,
            act_pos_net_in=flow3 - flow4,
            act_neg_net_in=flow5 - flow6,
            flowrate1_lpm=flowrate1_lpm,
            flowrate2_lpm=flowrate2_lpm,
            flowrate3_lpm=flowrate3_lpm,
            flowrate4_lpm=flowrate4_lpm,
            flowrate5_lpm=flowrate5_lpm,
            flowrate6_lpm=flowrate6_lpm,
        )

    def get_mean_mass_flowrate(self) -> Dict[str, float]:
        return self.get_mass_flowrate_dict()

    def set_init_press(
        self,
        init_pos_press: float,
        init_neg_press: float,
        init_act_pos_press: float = 101.325,
        init_act_neg_press: float = 101.325,
    ) -> None:
        self._last_time = 0.0
        self._raw_last_time = None
        self._time_origin = None
        self._stale_time_count = 0
        self._stale_obs_count = 0
        self._prev_press_vec = None
        self.backend.pos_press = float(init_pos_press)
        self.backend.neg_press = float(init_neg_press)
        self.backend.act_pos_press = float(init_act_pos_press)
        self.backend.act_neg_press = float(init_act_neg_press)
        self.backend.obs = np.array(
            [self.backend.pos_press, self.backend.neg_press],
            dtype=np.float32,
        )
        self.obs = np.array(
            [
                self.backend.pos_press,
                self.backend.neg_press,
                self.backend.act_pos_press,
                self.backend.act_neg_press,
            ],
            dtype=np.float64,
        )

    def set_offset(
        self,
        pos_offset: float,
        neg_offset: float,
        act_pos_offset: float = 0.0,
        act_neg_offset: float = 0.0,
    ) -> None:
        del pos_offset, neg_offset, act_pos_offset, act_neg_offset

    def set_volume(self, vol1: float, vol2: float) -> None:
        del vol1, vol2

    def set_pid(
        self,
        Kp_act_pos_in: float,
        Ki_act_pos_in: float,
        Kd_act_pos_in: float,
        Kp_act_pos_out: float,
        Ki_act_pos_out: float,
        Kd_act_pos_out: float,
        Kp_act_neg_in: float,
        Ki_act_neg_in: float,
        Kd_act_neg_in: float,
        Kp_act_neg_out: float,
        Ki_act_neg_out: float,
        Kd_act_neg_out: float,
    ) -> None:
        self.is_pid = True
        self.pid = ActuatorPressurePID(
            Kp_act_pos_in,
            Ki_act_pos_in,
            Kd_act_pos_in,
            Kp_act_pos_out,
            Ki_act_pos_out,
            Kd_act_pos_out,
            Kp_act_neg_in,
            Ki_act_neg_in,
            Kd_act_neg_in,
            Kp_act_neg_out,
            Ki_act_neg_out,
            Kd_act_neg_out,
            freq=self.freq,
        )

    def set_anti_windup(self, Ka: float) -> None:
        if not self.is_pid:
            raise RuntimeError("PID controller is not turned on.")
        self.is_anti_windup = True
        self.pid.set_anti_windup(Ka)

    def reset_pid(self) -> None:
        if self.is_pid:
            self.pid.reset()
