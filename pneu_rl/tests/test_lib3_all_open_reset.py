"""Hardware-free integration checks for prepared reset and reference clocks."""
import sys
from pathlib import Path
import unittest
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'pneu_rl/scripts'))
for package in ('pneu_env','pneu_rl','pneu_ref','pneu_utils'):
    sys.path.insert(0,str(ROOT/package/'src'))
from train_screening import make_components
from evaluate_lib3_screening import reset_to_reference_member
from evaluate_lib3_all_open_start import prepare_all_open


def config(policy='H'):
    cfg=yaml.safe_load((ROOT/f'pneu_rl/models/260831_lib3_histreward2x2_r025_{policy}_smooth_seed0/cfg.yaml').read_text())
    cfg['env']['verbose']=False
    return cfg


class PreparedResetTests(unittest.TestCase):
    def test_matches_evaluated_preparation_for_both_observations(self):
        for policy in ('H','HR'):
            cfg=config(policy)
            old=make_components(cfg)
            reset_to_reference_member(old,0)
            expected,_,_=prepare_all_open(old,30)
            expected=np.asarray(expected).copy()
            buffer=np.asarray(old.obs.obs_buf).copy()
            debug=old.obs.get_valve_debug().copy()
            old.close()
            cfg['initialization']=dict(mode='pump_on_all_open',precondition_seconds=30)
            env=make_components(cfg)
            actual,_=reset_to_reference_member(env,0)
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(env.obs.obs_buf,buffer)
            self.assertEqual(env.obs.get_valve_debug(),debug)
            env.close()

    def test_carry_preserves_physical_state_and_task_clock(self):
        cfg=config()
        cfg['initialization']=dict(mode='pump_on_all_open',precondition_seconds=30)
        env=make_components(cfg)
        env.reset()
        for _ in range(3):
            env.step(np.ones(6))
            physical=env.obs.lib.get_time_c()
            task=env.t
            history=env.obs_traj.copy()
            debug=env.obs.get_valve_debug().copy()
            env.reset()
            self.assertEqual(env.hard_reset_count,1)
            self.assertEqual(env.obs.lib.get_time_c(),physical)
            self.assertEqual(env.t,task)
            np.testing.assert_array_equal(env.obs_traj,history)
            self.assertEqual(env.obs.get_valve_debug(),debug)
        env.step(np.ones(6))
        self.assertAlmostEqual(env.t,.08)
        env.reset()  # Fifth block starts a newly prepared rollout.
        self.assertEqual(env.hard_reset_count,2)
        self.assertEqual(env.t,0)
        self.assertEqual(env.policy_step,0)
        self.assertAlmostEqual(env.obs.lib.get_time_c(),30,places=7)
        np.testing.assert_array_equal(env.action_history,np.ones((10,6)))
        env.close()

    def test_preparation_does_not_consume_reference(self):
        cfg=config('HR')
        cfg['initialization']=dict(mode='pump_on_all_open',precondition_seconds=30)
        env=make_components(cfg)
        state,_=reset_to_reference_member(env,0)
        ref0=env.goal.buf.copy()
        # Reference interpolation at zero differs from nominal ATM by ~0.000125 kPa.
        np.testing.assert_allclose(ref0.reshape(-1,2)[env.num_obs-1],[101.325,101.325],atol=2e-4)
        queried=[]
        original=env.goal.get_ref
        def capture(t):
            queried.append(t)
            return original(t)
        env.goal.get_ref=capture
        for _ in range(5): env.step(np.ones(6))
        np.testing.assert_allclose(queried,[0,.02,.04,.06,.08])
        self.assertGreater(env.obs.lib.get_time_c(),30)
        self.assertAlmostEqual(env.t,.1)
        env.close()

if __name__=='__main__': unittest.main()
