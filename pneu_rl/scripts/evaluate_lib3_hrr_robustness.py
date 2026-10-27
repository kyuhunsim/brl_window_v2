"""Frozen HR/HRR/HRR+DR comparison across fixed diagnostic plant scenarios.

No training, hardware, model changes, or simulator fitting. The seven training
scenarios and two new combinations are sensitivities, not identified Real ranges.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for package in ('pneu_env', 'pneu_ref', 'pneu_rl', 'pneu_utils'):
    sys.path.insert(0, str(ROOT / package / 'src'))

import numpy as np
import torch
import yaml

from train_lib3_hrr_comparison import make_training_components, policy_hash, sha256
from train_screening import seed_everything
from pneu_rl.sac_with_loss import SAC
from audit_lib3_hrr_runs import rollout, trace_metrics

EXPERIMENT = ROOT / 'pneu_rl/exp/lib3_hrr_dr_20261026'
DEFAULT_OUTPUT = EXPERIMENT / 'robustness_check'
LIBRARY = EXPERIMENT / 'build/lib3_type1_randomized.so'
PROFILE = ROOT / 'pneu_rl/config/domain_randomization/lib3_type1_v34_v1.yaml'
MODELS = {
    'HR': '261026_lib3_pressurecal_type1_HR_smooth_seed0',
    'HRR': '261026_lib3_pressurecal_type1_HRR_smooth_seed0',
    'HRR_DR': '261026_lib3_pressurecal_type1_HRR_DR_smooth_seed0',
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def conditions():
    scenarios = yaml.safe_load(PROFILE.read_text())['scenarios']
    for scenario in scenarios:
        scenario['group'] = 'nominal' if scenario['name'] == 'nominal' else 'training_single_factor'
    scenarios += [
        dict(name='heldout_fast_low_in', delay_seconds=.12, tau_scale=.75,
             flow3_scale=.95, flow4_scale=1.05, weight=1., group='heldout_combination'),
        dict(name='heldout_slow_high_in', delay_seconds=.08, tau_scale=1.5,
             flow3_scale=1.05, flow4_scale=.95, weight=1., group='heldout_combination'),
    ]
    return scenarios


def eval_config(cfg, scenario, library):
    result = copy.deepcopy(cfg)
    result['env']['verbose'] = False
    result['training_dynamics_randomization'] = {
        'enabled': True, 'seed': 610260, 'first_nominal': False,
        'scenarios': [dict(scenario)], 'library_path': str(library),
        'library_sha256': sha256(library),
    }
    return result


def flat_row(policy, condition, kind, metrics, feasibility):
    env = feasibility['operating_envelope_separate_from_physical_invariants']
    row = dict(policy=policy, condition=condition['name'], group=condition['group'],
               kind=kind, physical_invariants_passed=feasibility['passed'],
               supply_pos_max_kpa=feasibility['four_observed_port_pressures_kpa']['ch_pos']['max'],
               supply_neg_min_kpa=feasibility['four_observed_port_pressures_kpa']['ch_neg']['min'],
               operating_violation_samples=sum(env['out_of_bounds_sample_counts'].values()),
               command_tv_channel_mean_per_s=float(np.mean(metrics['command_total_variation_per_second'])))
    for ch in ('positive', 'negative'):
        for key, value in metrics[ch].items():
            row[ch + '_' + key] = value
    return row


def summarize(rows):
    result = {}
    for policy in MODELS:
        result[policy] = {}
        for kind in ('bank0', 'hold_last30'):
            result[policy][kind] = {}
            for group in ('nominal', 'training_single_factor', 'heldout_combination', 'all_non_nominal'):
                selected = [r for r in rows if r['policy'] == policy and r['kind'] == kind
                            and (r['group'] == group if group != 'all_non_nominal' else r['group'] != 'nominal')]
                if not selected:
                    continue
                summary = dict(conditions=len(selected),
                               all_physical_invariants_passed=all(r['physical_invariants_passed'] for r in selected),
                               operating_violation_samples=sum(r['operating_violation_samples'] for r in selected))
                for channel in ('positive', 'negative'):
                    summary[channel] = {
                        'pooled_tracking_rmse_kpa': float(np.sqrt(np.mean([r[channel+'_rmse_kpa']**2 for r in selected]))),
                        'mean_absolute_bias_kpa': float(np.mean([abs(r[channel+'_bias_kpa']) for r in selected])),
                        'maximum_tracking_rmse_kpa': max(r[channel+'_rmse_kpa'] for r in selected),
                        'maximum_pressure_band_2_20hz_rms_kpa': max(r[channel+'_pressure_band_2_20hz_rms_kpa'] for r in selected),
                        'pooled_pressure_band_2_20hz_rms_kpa': float(np.sqrt(np.mean([r[channel+'_pressure_band_2_20hz_rms_kpa']**2 for r in selected]))),
                    }
                result[policy][kind][group] = summary
    return result


def plot(rows, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scenarios = [s['name'] for s in conditions()]
    figure, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, constrained_layout=True)
    plots = [('bank0', 'positive_rmse_kpa', 'Bank0 POS tracking RMSE (kPa)'),
             ('bank0', 'negative_rmse_kpa', 'Bank0 NEG tracking RMSE (kPa)'),
             ('hold_last30', 'positive_pressure_band_2_20hz_rms_kpa', 'Hold130/70 POS ripple2–20Hz (kPa)'),
             ('hold_last30', 'positive_rmse_kpa', 'Hold130/70 POS tracking RMSE (kPa)')]
    for axis, (kind, metric, label) in zip(axes.flat, plots):
        for policy in MODELS:
            selected = {r['condition']: r for r in rows if r['policy'] == policy and r['kind'] == kind}
            axis.plot(np.arange(len(scenarios)), [selected[name][metric] for name in scenarios], marker='o', label=policy)
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
        axis.axvline(6.5, color='grey', ls='--', lw=.8)
        axis.set_xticks(np.arange(len(scenarios)), scenarios, rotation=30, ha='right', fontsize=8)
        axis.legend()
    figure.suptitle('Frozen deterministic policies; same refs; last two scenarios are held-out combinations')
    figure.savefig(directory/'robustness_comparison.png', dpi=145)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--library', type=Path, default=LIBRARY)
    args = parser.parse_args()
    output, library = args.output_dir.resolve(), args.library.resolve()
    if not output.is_relative_to(EXPERIMENT.resolve()):
        parser.error('Results must remain in this ignored experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(output/'.matplotlib')
    os.chdir(ROOT)
    torch.set_num_threads(1)
    started = time.monotonic()
    sources = [library, PROFILE]
    configs = {}
    for policy, name in MODELS.items():
        folder = ROOT/'pneu_rl/models'/name
        cfg = yaml.safe_load((folder/'cfg.yaml').read_text())
        configs[policy] = cfg
        sources += [folder/'cfg.yaml', folder/'model_epi_2000.pth', Path(cfg['obs']['library_path']),
                    ROOT/cfg['reference']['kwargs']['trajectory_npz']]
        if policy != 'HR' and not json.loads((folder/'wall_time.json').read_text()).get('completed'):
            raise RuntimeError('Never evaluate a changing checkpoint')
        if sha256(cfg['obs']['library_path']) != cfg['plant_provenance']['simulator_library_sha256']:
            raise RuntimeError('Nominal simulator provenance differs')
    before = {str(path):sha256(path) for path in sources}
    if sha256(library) != json.loads((library.parent/'build.json').read_text())['library_sha256']:
        raise RuntimeError('Randomized library differs from built snapshot')
    report = dict(protocol='frozen deterministic mean, no updates/replay/Real I/O',
                  conditions=conditions(), rows=[], detailed={}, input_sha256=before,
                  reference_handling='Same nominal bank0 and fixed hold start pressure for all models/conditions; no rebasing.',
                  scope=['7 training scenario values + 2 predeclared held-out parameter combinations, not identified Real ranges.',
                         'One bank member and one60s hold; not all references or all constant levels.',
                         'Hold statistics use30–60s, unlike completion_check120s holds90–120s.',
                         'Native physical invariants sampled at50Hz; not a substep/global stability certificate.',
                         'Models trained seed0 only; resumed training limitations remain applicable.'])
    reference_hashes = {}
    for policy, original in configs.items():
        report['detailed'][policy] = {}
        for condition in conditions():
            seed_everything(original['train_seed'])
            cfg = eval_config(original, condition, library)
            env = make_training_components(cfg)
            model = SAC(env=env, **cfg['model'])
            model.load_model(str(ROOT/'pneu_rl/models'/MODELS[policy]/'model_epi_2000.pth'), evaluate=True)
            frozen_hash = policy_hash(model)
            p0 = np.asarray(cfg['reference']['design']['initialization']['policy_start_observation_kpa'])[2:4]
            detail = {}
            try:
                for kind, steps, target in (('bank0', 2048, None), ('hold_last30', 3000, np.array([130.,70.]))):
                    arrays, result = rollout(env, model, steps, 0, target, p0)
                    metrics = trace_metrics(arrays, env.obs.freq, last_seconds=30 if target is not None else None)
                    key = (condition['name'], kind)
                    if key in reference_hashes and reference_hashes[key] != result['reference_sha256']:
                        raise RuntimeError('Paired models did not receive identical references')
                    reference_hashes[key] = result['reference_sha256']
                    result['metrics_selected'] = metrics
                    detail[kind] = result
                    report['rows'].append(flat_row(policy, condition, kind, metrics, result['feasibility']))
                    del arrays
                if policy_hash(model) != frozen_hash:
                    raise RuntimeError('Frozen actor changed')
            finally:
                env.close()
            report['detailed'][policy][condition['name']] = detail
            print(json.dumps({'policy':policy,'condition':condition['name'],
                              'pos_bank_rmse':detail['bank0']['metrics_selected']['positive']['rmse_kpa'],
                              'pos_hold_ripple':detail['hold_last30']['metrics_selected']['positive']['pressure_band_2_20hz_rms_kpa'],
                              'all_physical_passed':all(v['feasibility']['passed'] for v in detail.values())}), flush=True)
            write_json(output/'results.json', report)
    if before != {str(path):sha256(path) for path in sources}:
        raise RuntimeError('Frozen input changed during evaluation')
    report['summary'] = summarize(report['rows'])
    report['elapsed_seconds'] = time.monotonic()-started
    write_json(output/'results.json', report)
    write_json(output/'summary.json', report['summary'])
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['rows'][0]))
        writer.writeheader(); writer.writerows(report['rows'])
    plot(report['rows'], output)
    print(json.dumps({'completed':True,'elapsed_seconds':report['elapsed_seconds'],
                      'rollouts':len(report['rows']),'output':str(output)}), flush=True)


if __name__ == '__main__':
    main()
