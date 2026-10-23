"""Evaluate frozen physical-model policies over every matched reference member.

Stores observed tracking data separately from instantaneous native physical state.
No model/config/reference files are changed, and no real hardware is accessed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for package in ('pneu_env', 'pneu_ref', 'pneu_rl', 'pneu_utils'):
    sys.path.insert(0, str(ROOT / package / 'src'))
os.chdir(ROOT)
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
import torch
import yaml
from train_screening import make_components
from evaluate_lib3_screening import reset_to_reference_member
from pneu_rl.sac_with_loss import SAC

OUT = ROOT / 'pneu_rl/exp/lib3_physical_training_review_20261023'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def evaluate(args):
    from check_lib3_closed_loop_feasibility import NativeStateSampler
    torch.set_num_threads(1)
    np.random.seed(0)
    torch.manual_seed(0)
    folder = ROOT / 'pneu_rl/models' / args.model
    checkpoint = folder / f'model_epi_{args.checkpoint:04d}.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Final checkpoint is not available: {checkpoint}')
    cfg = yaml.full_load((folder / 'cfg.yaml').read_text())
    assert cfg['lib'] == 3 and cfg['obs']['freq'] == 50
    assert cfg['initialization'] == dict(mode='pump_on_all_open', precondition_seconds=30.0,
        reference_clock='policy_step/freq; preserved across carried blocks')
    assert cfg.get('pid') is None and cfg.get('pred') is None
    assert cfg['env'].get('action_chunk_steps', 1) == 1
    cfg['env']['verbose'] = False
    library = Path(cfg['obs']['library_path'])
    bank_path = ROOT / cfg['reference']['kwargs']['trajectory_npz']
    assert digest(library) == cfg['plant_provenance']['simulator_library_sha256']
    assert digest(bank_path) == cfg['plant_provenance']['reference_bank_sha256']
    inputs = {str(p): digest(p) for p in (folder / 'cfg.yaml', checkpoint, library, bank_path)}
    env = make_components(cfg)
    assert env.obs.runtime_state_schema == 2
    sampler = NativeStateSampler(env.obs)
    model = SAC(env=env, **cfg['model'])
    model.load_model(str(checkpoint), evaluate=True)
    hold_targets = [(120., 75.), (130., 70.), (140., 80.)]
    steps = 1750 if args.constant_holds else cfg['model']['horizon'] * cfg['env']['episode_carry_blocks']
    members = len(hold_targets) if args.constant_holds else len(env.goal.ref.trajectories)
    destination = args.output / args.model
    if args.constant_holds:
        destination = destination / 'constant_holds'
    destination.mkdir(parents=True, exist_ok=True)
    shapes = dict(pressure_observed=(4,), reference=(2,), policy_action=(6,),
                  physical_command=(6,), native_state=(37,), valve_debug=(6, 9),
                  mass_flowrate=(10,), reward=(), native_time=())
    data = {key: np.empty((members, steps) + shape, dtype=np.float64)
            for key, shape in shapes.items()}
    initial = []
    started = time.perf_counter()
    original_get_goal = env.goal.ref.get_goal
    try:
        for member in range(members):
            env.goal.ref.get_goal = original_get_goal
            state, _ = reset_to_reference_member(env, 0 if args.constant_holds else member)
            if args.constant_holds:
                start_pressure = env.curr_full_press[2:4].copy()
                target_pressure = np.asarray(hold_targets[member])
                def hold_goal(t, p0=start_pressure, target=target_pressure):
                    s = np.clip(float(t) / 3., 0., 1.)
                    weight = 10*s**3 - 15*s**4 + 6*s**5
                    return tuple(p0 + weight * (target-p0))
                env.goal.ref.get_goal = hold_goal
                env.goal.buf = np.tile(start_pressure, env.goal.num_ref)
                env.prev_ref_current = start_pressure.copy()
                state, _ = env._state_from_current(env.last_policy_action)
            initial.append(dict(member=member, pressure=env.curr_full_press.tolist(),
                                history=env.obs_traj.tolist(),
                                action_history=env.action_history.tolist()))
            expected = cfg['reference']['design']['initialization']['policy_start_observation_kpa']
            np.testing.assert_allclose(env.curr_full_press, expected, atol=.02)
            np.testing.assert_allclose(env._current_actuator_reference(), expected[2:4], atol=.001)
            for step in range(steps):
                action = model.predict(state, evaluate=True)
                state, reward, done, truncated, info = env.step(action)
                native = sampler.sample()
                obs = info['obs']
                applied = np.asarray(info['ctrl_input']).reshape(-1, 6)[0]
                data['pressure_observed'][member, step] = env.curr_full_press
                data['reference'][member, step] = [obs['ref_act_pos'], obs['ref_act_neg']]
                data['policy_action'][member, step] = action
                data['physical_command'][member, step] = env.obs.scale_low + (
                    env.obs.scale_high - env.obs.scale_low) * applied
                for key in ('native_state', 'valve_debug', 'mass_flowrate'):
                    data[key][member, step] = native[key]
                data['reward'][member, step] = reward
                data['native_time'][member, step] = env.obs.lib.get_time_c()
                if not np.isfinite(state).all() or not np.isfinite(reward):
                    raise RuntimeError(f'Nonfinite policy state/reward: member={member}, step={step}')
                # Mirror the four carried training blocks without changing state or time.
                if (step + 1) % cfg['model']['horizon'] == 0 and step + 1 < steps:
                    before = state.copy()
                    state, _ = env.reset()
                    np.testing.assert_array_equal(state, before)
            assert abs(env.t - steps / 50) < 1e-8
            print(f'{args.model}: bank {member + 1}/{members}, '
                  f'RMSE={np.sqrt(np.mean((data["pressure_observed"][member, :, 2:] - data["reference"][member])**2, axis=0))}',
                  flush=True)
    finally:
        env.close()
    assert inputs == {path: digest(path) for path in inputs}
    data['time'] = (np.arange(steps) + 1) / 50.
    data['member_index'] = np.arange(members)
    np.savez_compressed(destination / 'rollout.npz', **data)
    meta = dict(model=args.model, checkpoint_episode=args.checkpoint,
                calibration_type=cfg['plant_provenance']['calibration_type'],
                policy='HR' if cfg['env']['include_physics_features'] else 'H',
                reference_members=members, steps_per_member=steps, frequency_hz=50,
                reference_mode='constant_holds' if args.constant_holds else 'training_bank',
                constant_targets_kpa=hold_targets if args.constant_holds else None,
                spectral_analysis_start_sec=10 if args.constant_holds else 5,
                initialization=cfg['initialization'], runtime_state_schema=2,
                elapsed_sec=time.perf_counter()-started, input_sha256=inputs,
                initial_states=initial,
                scope=('Deterministic policy mean; matched nominal simulator; three constant-target diagnostics with3s smooth startup.'
                       if args.constant_holds else
                       'Deterministic policy mean; matched nominal simulator and training reference bank; no learning/real hardware/OOD test.'))
    write_json(destination / 'metadata.json', meta)
    analyze(destination)


def band_statistics(values, fs=50.):
    f, p = welch(values, fs=fs, nperseg=min(1024, len(values)), detrend='linear')
    df = f[1]-f[0]
    energy = lambda lo, hi: float(np.sum(p[(f >= lo) & (f <= hi)]) * df)
    relevant = (f >= .1) & (f <= 20.)
    peak = float(f[relevant][np.argmax(p[relevant])])
    return dict(peak_01_20_hz=peak, rms_01_2_hz=float(np.sqrt(energy(.1, 2))),
                rms_2_20_hz=float(np.sqrt(energy(2, 20))),
                rms_8_12_hz=float(np.sqrt(energy(8, 12))))


def analyze(destination):
    with np.load(destination / 'rollout.npz') as bank:
        data = {key: bank[key] for key in bank.files}
    meta = json.loads((destination / 'metadata.json').read_text())
    errors = data['pressure_observed'][:, :, 2:4] - data['reference']
    t = data['time']
    rows, spectral_rows = [], []
    sos = butter(4, [2, 20], btype='bandpass', fs=50, output='sos')
    spectral_mask = t > meta.get('spectral_analysis_start_sec', 5)
    for member in range(len(errors)):
        for channel, name in enumerate(('pos', 'neg')):
            error = errors[member, :, channel]
            target = data['reference'][member, :, channel]
            rate = np.gradient(target, .02)
            low_rate = np.abs(rate) < .2
            plateau = (np.convolve(low_rate.astype(float), np.ones(25), mode='full')[:len(t)] >= 25) & (t > 5)
            for window, mask in (('whole', np.ones(len(t), bool)), ('after5', t > 5),
                                 ('after10', t > 10), ('hold_after5', plateau)):
                e = error[mask]
                rows.append(dict(model=meta['model'], type=meta['calibration_type'], policy=meta['policy'],
                                 member=member, channel=name, window=window, n=len(e),
                                 rmse=float(np.sqrt(np.mean(e**2))) if len(e) else None,
                                 bias=float(np.mean(e)) if len(e) else None,
                                 mae=float(np.mean(np.abs(e))) if len(e) else None,
                                 max_abs_error=float(np.max(np.abs(e))) if len(e) else None))
            highpass = sosfiltfilt(sos, error)
            for signal, values in (('error', error), ('pressure', data['pressure_observed'][member, :, channel+2]),
                                   ('reference', target)):
                stats = band_statistics(values[spectral_mask])
                spectral_rows.append(dict(model=meta['model'], type=meta['calibration_type'], policy=meta['policy'],
                                          member=member, channel=name, signal=signal, **stats))
            spectral_rows[-3]['bandpass_2_20_peak_to_peak_kpa'] = float(np.ptp(highpass[spectral_mask]))
    metrics = pd.DataFrame(rows)
    spectra = pd.DataFrame(spectral_rows)
    metrics.to_csv(destination / 'tracking_by_member.csv', index=False)
    spectra.to_csv(destination / 'vibration_by_member.csv', index=False)
    result = dict(model=meta['model'], type=meta['calibration_type'], policy=meta['policy'],
                  checkpoint_episode=meta['checkpoint_episode'], members=len(errors),
                  reference_mode=meta.get('reference_mode', 'training_bank'),
                  spectral_analysis_start_sec=meta.get('spectral_analysis_start_sec', 5),
                  tracking={}, vibration={}, commands={})
    for channel, name in enumerate(('pos', 'neg')):
        result['tracking'][name] = {}
        for window in ('whole', 'after5', 'after10', 'hold_after5'):
            group = metrics[(metrics.channel == name) & (metrics.window == window)]
            total = group.n.sum()
            result['tracking'][name][window] = dict(
                rmse=float(np.sqrt(np.sum(group.rmse**2 * group.n) / total)),
                bias=float(np.sum(group.bias * group.n) / total),
                mae=float(np.sum(group.mae * group.n) / total),
                worst_member_rmse=float(group.rmse.max()), samples=int(total))
        target_variance = float(np.var(data['reference'][:, :, channel]))
        result['tracking'][name]['whole']['r2'] = (
            1.-result['tracking'][name]['whole']['rmse']**2/target_variance
            if target_variance > 1e-12 else None)
        group = spectra[(spectra.channel == name) & (spectra.signal == 'error')]
        result['vibration'][name] = dict(
            rms_2_20_hz_kpa=float(np.sqrt(np.mean(group.rms_2_20_hz**2))),
            rms_8_12_hz_kpa=float(np.sqrt(np.mean(group.rms_8_12_hz**2))),
            worst_member_rms_2_20_hz_kpa=float(group.rms_2_20_hz.max()),
            worst_member=int(group.loc[group.rms_2_20_hz.idxmax(), 'member']),
            median_peak_01_20_hz=float(group.peak_01_20_hz.median()))
        if meta.get('reference_mode') == 'constant_holds':
            y = data['pressure_observed'][:, spectral_mask, channel+2]
            result['vibration'][name].update(
                held_target_pressure_ranges_kpa=np.ptp(y, axis=1).tolist(),
                held_target_pressure_std_kpa=np.std(y, axis=1).tolist())
            # An argmax of numerical noise is not a resolved oscillation peak.
            result['vibration'][name]['peak_resolved_above_001kpa_range'] = bool(np.max(np.ptp(y, axis=1)) >= .001)
            if not result['vibration'][name]['peak_resolved_above_001kpa_range']:
                result['vibration'][name]['median_peak_01_20_hz'] = None
    command = data['physical_command']
    delta = np.diff(command, axis=1)
    result['commands'] = dict(mean_abs_delta_per_step=np.abs(delta).mean(axis=(0, 1)).tolist(),
                              max_abs_delta_per_step=np.abs(delta).max(axis=(0, 1)).tolist(),
                              step_over_005_fraction=(np.abs(delta) > .05).mean(axis=(0, 1)).tolist(),
                              low_boundary_fraction=(command <= .801).mean(axis=(0, 1)).tolist(),
                              high_boundary_fraction=(command >= .999).mean(axis=(0, 1)).tolist())
    opening = data['native_state'][:, :, 7:].reshape(len(errors), len(t), 6, 5)[:, :, :, 1]
    result['commands']['mean_abs_opening_delta_per_step'] = np.abs(np.diff(opening, axis=1)).mean(axis=(0, 1)).tolist()
    result['limitations'] = [
        ('Three constant-reference diagnostics within the training pressure range; no comprehensive OOD or real transfer claim.'
         if meta.get('reference_mode') == 'constant_holds' else
         'Reference bank members were used in training; these are nominal deterministic tracking tests, not OOD or real transfer evidence.'),
        'Hold bias uses abs(reference rate)<0.2 kPa/s for at least 0.5s after startup; it is not an asymptotic steady-state guarantee.',
        f'Welch PSD uses samples after{meta.get("spectral_analysis_start_sec", 5)}s, linear detrending, 50Hz sampling; band RMS measures components, not stability or physical cause.',
        'The 8-12Hz band tests the earlier ~10Hz concern, but 50Hz records cannot attribute physical components above25Hz.',
    ]
    write_json(destination / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


def plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = []
    for kind in ('type1', 'type2'):
        for policy in ('H', 'HR'):
            folder = output / f'261023_lib3_opening_{kind}_{policy}_smooth_seed0'
            if not (folder / 'summary.json').is_file():
                continue
            analyze(folder)
            with np.load(folder / 'rollout.npz') as data:
                arrays = {key: data[key] for key in data.files}
            cases.append((json.loads((folder / 'summary.json').read_text()), arrays))
    if len(cases) != 4:
        raise RuntimeError('All four final evaluations are required for comparison plots')
    for kind in ('type1', 'type2'):
        pair = [data for meta, data in cases if meta['type'] == kind]
        np.testing.assert_array_equal(pair[0]['reference'], pair[1]['reference'])
    write_json(output / 'policy_summary.json', [meta for meta, _ in cases])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig_s, axes_s = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    colors = {'H': '#2878b5', 'HR': '#d95f02'}
    for row, kind in enumerate(('type1', 'type2')):
        for col, channel in enumerate(('pos', 'neg')):
            for meta, data in cases:
                if meta['type'] != kind:
                    continue
                ax = axes[row, col]
                if meta['policy'] == 'H':
                    ax.plot(data['time'], data['reference'][0, :, col], 'k--', label='Reference', lw=1.2)
                ax.plot(data['time'], data['pressure_observed'][0, :, col+2],
                        label=meta['policy'], color=colors[meta['policy']], lw=1)
                # Average PSD over independent members, never concatenate reset discontinuities.
                psds = []
                for member in range(16):
                    error = data['pressure_observed'][member, :, col+2] - data['reference'][member, :, col]
                    f, ps = welch(error[data['time'] > 5], fs=50, nperseg=1024, detrend='linear')
                    psds.append(ps)
                axes_s[row, col].semilogy(f, np.mean(psds, axis=0), label=meta['policy'], color=colors[meta['policy']])
            axes[row, col].set_title(f'{kind}: actuator {channel.upper()} / reference member0')
            axes[row, col].set_ylabel('Pressure [kPa absolute]')
            axes[row, col].grid(alpha=.25)
            axes[row, col].legend()
            axes_s[row, col].axvspan(8, 12, color='gray', alpha=.15, label='8-12 Hz')
            axes_s[row, col].set_xlim(.1, 25)
            axes_s[row, col].set_title(f'{kind}: {channel.upper()} tracking error PSD, mean of16 members')
            axes_s[row, col].set_ylabel('PSD [kPa²/Hz]')
            axes_s[row, col].grid(alpha=.25)
            axes_s[row, col].legend()
    for ax in axes[-1]: ax.set_xlabel('Policy time [s]')
    for ax in axes_s[-1]: ax.set_xlabel('Frequency [Hz]')
    fig.tight_layout(); fig.savefig(output / 'tracking_bank0.png', dpi=160); plt.close(fig)
    fig_s.tight_layout(); fig_s.savefig(output / 'error_spectra.png', dpi=160); plt.close(fig_s)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = [meta['type']+' '+meta['policy'] for meta, _ in cases]
    x = np.arange(4)
    for channel, offset, color in (('pos', -.18, '#b73737'), ('neg', .18, '#2878b5')):
        axes[0].bar(x+offset, [meta['tracking'][channel]['whole']['rmse'] for meta, _ in cases],
                    width=.36, label=channel.upper(), color=color)
        axes[1].bar(x+offset, [meta['vibration'][channel]['rms_2_20_hz_kpa'] for meta, _ in cases],
                    width=.36, label=channel.upper(), color=color)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis='y', alpha=.25); ax.legend()
    axes[0].set_ylabel('Tracking RMSE [kPa]'); axes[0].set_title('All16 matched reference members')
    axes[1].set_ylabel('2-20Hz error RMS [kPa]'); axes[1].set_title('After5s; separate from slow tracking error')
    fig.tight_layout(); fig.savefig(output / 'tracking_vibration_metrics.png', dpi=160); plt.close(fig)
    # Worst pressure-ripple member/channel per model, showing actual and filtered error.
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    sos = butter(4, [2, 20], btype='bandpass', fs=50, output='sos')
    for row, (meta, data) in enumerate(cases):
        channel = max(('pos', 'neg'), key=lambda c: meta['vibration'][c]['worst_member_rms_2_20_hz_kpa'])
        index = 0 if channel == 'pos' else 1
        member = meta['vibration'][channel]['worst_member']
        error = data['pressure_observed'][member, :, index+2]-data['reference'][member, :, index]
        filtered = sosfiltfilt(sos, error)
        win = np.convolve(filtered**2, np.ones(250)/250, mode='same')
        win[data['time'] < 7.5] = -1
        center = data['time'][np.argmax(win)]
        selected = np.abs(data['time']-center) <= 2.5
        axes[row, 0].plot(data['time'], data['reference'][member, :, index], 'k--', label='Reference')
        axes[row, 0].plot(data['time'], data['pressure_observed'][member, :, index+2], label='Pressure', lw=.9)
        axes[row, 1].plot(data['time'][selected], filtered[selected], label='2-20Hz error', lw=.9)
        axes[row, 0].set_title(f'{meta["type"]} {meta["policy"]}, {channel.upper()}, worst-ripple member{member}')
        axes[row, 0].set_ylabel('kPa absolute'); axes[row, 1].set_ylabel('Bandpassed error [kPa]')
        for ax in axes[row]: ax.grid(alpha=.25); ax.legend()
    for ax in axes[-1]: ax.set_xlabel('Policy time [s]')
    fig.tight_layout(); fig.savefig(output / 'worst_ripple_examples.png', dpi=160); plt.close(fig)
    hold_results = []
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    for row, (meta, _) in enumerate(cases):
        folder = output / meta['model'] / 'constant_holds'
        analyze(folder)
        hold_meta = json.loads((folder / 'summary.json').read_text())
        hold_results.append(hold_meta)
        with np.load(folder / 'rollout.npz') as data:
            time_values = data['time']
            for col, channel in enumerate(('pos', 'neg')):
                for member, target in enumerate(((120, 75), (130, 70), (140, 80))):
                    error = data['pressure_observed'][member, :, col+2] - data['reference'][member, :, col]
                    selected = time_values > 5
                    axes[row, col].plot(time_values[selected], error[selected],
                                        label=f'target {target[col]} kPa', lw=.85)
                axes[row, col].axhline(0, color='k', lw=.6)
                axes[row, col].axvline(10, color='gray', ls='--', lw=.8)
                axes[row, col].set_title(f'{meta["type"]} {meta["policy"]}: constant {channel.upper()} target')
                axes[row, col].set_ylabel('Pressure - target [kPa]')
                axes[row, col].grid(alpha=.25); axes[row, col].legend(fontsize=8)
    for ax in axes[-1]: ax.set_xlabel('Policy time [s]; reference constant after3s')
    fig.tight_layout(); fig.savefig(output / 'constant_hold_error.png', dpi=160); plt.close(fig)
    write_json(output / 'constant_hold_summary.json', hold_results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--checkpoint', type=int, default=2000)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--constant-holds', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        plots(args.output)
    elif args.model:
        evaluate(args)
    else:
        parser.error('--model or --plots-only is required')


if __name__ == '__main__':
    main()
