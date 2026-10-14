from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pneu_env.pid5 import PneumaticController5
from pneu_rl.sac_with_loss5 import SAC

from train5 import build_env, default_config


def main() -> None:
    controller = PneumaticController5()
    ctrl = controller.get_action(
        np.array([160.0, 70.0, 110.0, 88.0]),
        np.array([150.0, 50.0, 120.0, 80.0, 10.0]),
    )
    assert ctrl.shape == (6,)
    assert ctrl[2]*ctrl[3] == 0.0 and ctrl[4]*ctrl[5] == 0.0
    assert np.all((0.0 <= ctrl) & (ctrl <= 1.0))

    args = SimpleNamespace(
        no_pred=False, residual_mode="reserve", horizon=16, episodes=1
    )
    cfg = default_config(args)
    cfg["env"].update(num_prev=2, num_pred=2, num_act=1)
    cfg["model"].update(
        hidden_dim=32, batch_size=8, buffer_size=1000, horizon=16,
        start_epi=0, temporal_weight=0.0, spatial_weight=0.0,
    )
    env = build_env(cfg)
    model = SAC(env, **cfg["model"])
    state, info = env.reset()
    assert state.shape == env.observation_space.shape
    assert info["goal_traj"].shape[1] == 5
    rewards = []
    try:
        for _ in range(12):
            action = model.select_action(state)
            next_state, reward, done, _, step_info = env.step(action)
            model.buffer.add(state, action, reward, next_state, done)
            state = next_state
            rewards.append(reward)
            applied = np.array([
                step_info["obs"]["ctrl_act_pos_in"],
                step_info["obs"]["ctrl_act_pos_out"],
                step_info["obs"]["ctrl_act_neg_in"],
                step_info["obs"]["ctrl_act_neg_out"],
            ])
            opening = np.clip((applied-env.obs.scale_low)/(env.obs.scale_high-env.obs.scale_low), 0, 1)
            assert opening[0]*opening[1] < 1e-12
            assert opening[2]*opening[3] < 1e-12
        assert np.all(np.isfinite(rewards))
        model.update_parameters()
    finally:
        env.close()
    print("self_check5: PASS")


if __name__ == "__main__":
    main()
