"""Portable model references and shared lib3 evaluation/retraining startup."""
import copy
from pathlib import Path
import yaml

from pneu_utils.utils import get_pkg_path
from pneu_env.env3 import PneuEnv3
from pneu_env.env3_all_open import PneuEnv3AllOpen, PneuEnv3RealAllOpen
from pneu_env.sim3 import PneuSim
from pneu_env.real.real9 import PneuReal


def load_deployment_config(model_name, models_dir=None):
    root = Path(get_pkg_path('pneu_rl')).parent
    model_dir = Path(models_dir or root / 'pneu_rl/models') / model_name
    cfg = yaml.safe_load((model_dir / 'cfg.yaml').read_text())
    spec = cfg.get('reference') or {}
    kwargs = spec.get('kwargs') or {}
    if 'trajectory_npz' in kwargs:
        path = Path(kwargs['trajectory_npz'])
        candidates = [model_dir / path, root / path] if not path.is_absolute() else [path]
        resolved = next((p.resolve() for p in candidates if p.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(f'Missing reference bank for {model_name}: {path}')
        kwargs['trajectory_npz'] = str(resolved)
    return cfg


def build_deployment_env(config, obs, ref, pred=None):
    initialization = config.get('initialization') or {}
    mode = initialization.get('mode')
    if mode not in (None, 'pump_on_all_open'):
        raise ValueError(f'Unsupported initialization mode: {mode}')
    if mode == 'pump_on_all_open':
        cls = PneuEnv3AllOpen if isinstance(obs, PneuSim) else PneuEnv3RealAllOpen
        if not isinstance(obs, (PneuSim, PneuReal)):
            raise ValueError('Unsupported prepared observation backend')
        return cls(obs=obs, ref=ref, pred=pred,
                   precondition_seconds=initialization.get('precondition_seconds', 30),
                   **config['env'])
    return PneuEnv3(obs=obs, ref=ref, pred=pred, **config['env'])


def continuous_finetune_config(config):
    """One preparation; preserve physical state across all 200 RL boundaries."""
    cfg = copy.deepcopy(config)
    if (cfg.get('initialization') or {}).get('mode') != 'pump_on_all_open':
        return cfg
    spec = cfg['reference']
    rates = spec['design']['signed_reference_rate_limits_kpa_s']
    cfg['reference'] = dict(kind='continuous_trajectory_bank', design=copy.deepcopy(spec['design']),
                            kwargs=dict(trajectory_npz=spec['kwargs']['trajectory_npz'],
                                        rate_limits=copy.deepcopy(rates), freq=cfg['obs']['freq'],
                                        seed=spec['kwargs'].get('seed', 61098)))
    cfg['reference']['design']['runtime_protocol'] = 'continuous_with_minimum_jerk_bridges'
    cfg['reference']['design']['runtime_bridge_feasibility_certified'] = False
    cfg['env']['episode_carry_state'] = True
    cfg['env']['episode_carry_blocks'] = 0
    cfg['env']['verbose'] = False
    cfg['initialization']['preparation_scope'] = 'session_start_only'
    return cfg
