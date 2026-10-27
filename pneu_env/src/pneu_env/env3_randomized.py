"""Training-only, hard-reset dynamics randomization for prepared lib3 runs.

The native library contains global simulator state: use one environment per
process (or distinct library files). This class is deliberately not selected by
the normal evaluation/Real environment factory. Scenario values never enter the
policy observation, action transformation, reference, or reward.
"""
from collections import deque
from ctypes import POINTER, c_double, c_int
import json
from pathlib import Path

import numpy as np

from pneu_env.env3_all_open import PneuEnv3AllOpen


class PneuEnv3Randomized(PneuEnv3AllOpen):
    def __init__(self, *args, dynamics_randomization=None, **kwargs):
        # Parent construction does not reset the environment or consume an RNG.
        super().__init__(*args, **kwargs)
        config = dict(dynamics_randomization or {})
        self.randomization_enabled = bool(config.get("enabled", False))
        self.current_dynamics_scenario = None
        self._dr_draw_count = 0
        if not self.randomization_enabled:
            return

        self._dr_rng = np.random.default_rng(int(config.get("seed", 0)))
        self._dr_first_nominal = bool(config.get("first_nominal", True))
        self._dr_log_path = (Path(config["log_path"]).expanduser()
                             if config.get("log_path") else None)
        self._dr_scenarios = []
        names = set()
        for item in config.get("scenarios", []):
            scenario = {
                "name": str(item["name"]),
                "delay_seconds": float(item["delay_seconds"]),
                "tau_scale": float(item["tau_scale"]),
                "flow3_scale": float(item["flow3_scale"]),
                "flow4_scale": float(item["flow4_scale"]),
                "weight": float(item.get("weight", 1.0)),
            }
            if not scenario["name"] or scenario["name"] in names:
                raise ValueError("Dynamics scenario names must be nonempty and unique")
            names.add(scenario["name"])
            values = [scenario[k] for k in ("tau_scale", "flow3_scale",
                                            "flow4_scale", "weight")]
            if not np.all(np.isfinite(values)) or min(values) <= 0:
                raise ValueError("Dynamics scales and weights must be finite and positive")
            delay = scenario["delay_seconds"]
            if not np.isfinite(delay) or delay < 0:
                raise ValueError("Observation delay must be finite and nonnegative")
            delay_steps = delay * self.obs.freq
            if not np.isclose(delay_steps, round(delay_steps), rtol=0, atol=1e-9):
                raise ValueError("Observation delay must be an integer number of control steps")
            if self.precondition_steps < self.num_obs + round(delay_steps):
                raise ValueError("Preparation must fill history and every scenario's delay buffer")
            self._dr_scenarios.append(scenario)
        if not self._dr_scenarios:
            raise ValueError("Enabled dynamics randomization requires at least one scenario")
        weights = np.asarray([s["weight"] for s in self._dr_scenarios])
        self._dr_probabilities = weights / weights.sum()

        nominal = [i for i, s in enumerate(self._dr_scenarios)
                   if s["tau_scale"] == s["flow3_scale"] == s["flow4_scale"] == 1.0
                   and np.isclose(s["delay_seconds"], self.obs.delay, rtol=0, atol=1e-12)]
        if self._dr_first_nominal and not nominal:
            raise ValueError("first_nominal requires a scenario with unit scales and baseline delay")
        self._dr_nominal_index = nominal[0] if nominal else None
        try:
            self._dr_setter = self.obs.lib.set_slew_diagnostic
            self._dr_getter = self.obs.lib.get_slew_diagnostic
        except AttributeError as exc:
            raise ValueError("Randomization requires the isolated diagnostic lib3 library") from exc
        self._dr_setter.argtypes = [c_double, c_double, c_double]
        self._dr_setter.restype = c_int
        self._dr_getter.argtypes = [POINTER(c_double)]
        self._dr_getter.restype = None

    def _will_hard_reset(self):
        # Match PneuEnv3.reset: carried RL horizons are one physical rollout.
        return not (
            self.episode_carry_state
            and self.has_episode_state
            and not self.force_hard_reset_next
            and (self.episode_carry_blocks == 0
                 or self.carry_block_index < self.episode_carry_blocks)
        )

    def _select_dynamics(self):
        if self._dr_draw_count == 0 and self._dr_first_nominal:
            index = self._dr_nominal_index
        else:
            index = int(self._dr_rng.choice(len(self._dr_scenarios),
                                            p=self._dr_probabilities))
        scenario = self._dr_scenarios[index].copy()
        controls = [scenario[k] for k in ("tau_scale", "flow3_scale", "flow4_scale")]
        if self._dr_setter(*controls) != 0:
            raise RuntimeError("Native lib3 rejected dynamics scenario")
        self.obs.delay = scenario["delay_seconds"]
        self.obs.obs_buf = deque(maxlen=round(self.obs.freq * self.obs.delay) + 1)
        self.current_dynamics_scenario = scenario
        self._dr_draw_count += 1

    def _check_native_controls(self):
        controls = (c_double * 3)()
        self._dr_getter(controls)
        expected = [self.current_dynamics_scenario[k]
                    for k in ("tau_scale", "flow3_scale", "flow4_scale")]
        if not np.array_equal(np.asarray(controls), expected):
            raise RuntimeError("Native dynamics changed during reset/carry; check library sharing")

    def reset(self):
        if not self.randomization_enabled:
            return super().reset()
        hard_reset = self._will_hard_reset()
        previous_resets = self.hard_reset_count
        if hard_reset:
            # Set before native reset and all-open preparation. Do not clear
            # history or the observation FIFO at a carried episode boundary.
            self._select_dynamics()
        result = super().reset()
        if (self.hard_reset_count > previous_resets) != hard_reset:
            raise RuntimeError("Parent reset semantics changed; scenario schedule no longer matches")
        self._check_native_controls()
        if hard_reset and self._dr_log_path is not None:
            record = {
                "hard_reset_count": self.hard_reset_count,
                "scenario_index_in_rollout_sequence": self._dr_draw_count - 1,
                "scenario": self.current_dynamics_scenario,
                "preparation_seconds": self.precondition_seconds,
                "policy_start_pressure_kpa": self.curr_full_press.tolist(),
                "policy_start_reference_kpa": self._current_actuator_reference().tolist(),
                "initial_tracking_error_kpa": (
                    self.curr_full_press[2:4] - self._current_actuator_reference()).tolist(),
                "effective_delay_steps": self.obs.obs_buf.maxlen - 1,
                "native_time_seconds": float(self.obs.lib.get_time_c()),
            }
            self._dr_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._dr_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        return result

    def get_info(self, state_info, reward_info):
        info = super().get_info(state_info, reward_info)
        if getattr(self, "randomization_enabled", False):
            info["dynamics_randomization"] = {
                "scenario": (dict(self.current_dynamics_scenario)
                             if self.current_dynamics_scenario is not None else None),
                "hard_reset_count": self.hard_reset_count,
                "carry_block_index": self.carry_block_index,
            }
        return info
