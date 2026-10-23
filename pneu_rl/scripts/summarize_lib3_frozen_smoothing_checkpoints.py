"""Compare saved checkpoint rollouts without training or policy updates."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from check_lib3_closed_loop_feasibility import analyze_arrays

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'pneu_rl/exp/lib3_physical_training_review_20261023'
OUTPUT = REVIEW / 'smoothing_review'


def main():
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row, kind in enumerate(('type1', 'type2')):
        model = f'261023_lib3_opening_{kind}_HR_smooth_seed0'
        reference = None
        for checkpoint in (1500, 2000):
            base = OUTPUT / 'checkpoint1500' if checkpoint == 1500 else REVIEW
            folder = base / model
            result = dict(type=kind, policy='HR', checkpoint=checkpoint)
            for mode, suffix in (('bank', ''), ('holds', 'constant_holds')):
                directory = folder / suffix
                summary = json.loads((directory / 'summary.json').read_text())
                with np.load(directory / 'rollout.npz') as saved:
                    arrays = {key: saved[key] for key in saved.files}
                if mode == 'bank':
                    if reference is None:
                        reference = arrays['reference'].copy()
                    else:
                        np.testing.assert_array_equal(reference, arrays['reference'])
                feasibility = analyze_arrays(arrays)
                result[mode] = dict(
                    tracking=summary['tracking'], vibration=summary['vibration'],
                    commands=summary['commands'], physical_checks_passed=feasibility['passed'],
                    operating=feasibility['operating_envelope_separate_from_physical_invariants'],
                    rollout=str((directory / 'rollout.npz').relative_to(ROOT)))
                if mode == 'holds':
                    for col, channel in enumerate(('pos', 'neg')):
                        # Same constant target (130,70) for a direct frozen-policy overlay.
                        t = arrays['time']
                        error = arrays['pressure_observed'][1, :, col+2] - arrays['reference'][1, :, col]
                        axes[row, col].plot(t[t > 5], error[t > 5], lw=.9,
                                            label=f'checkpoint {checkpoint}')
            results.append(result)
        for col, channel in enumerate(('POS', 'NEG')):
            axes[row, col].axhline(0, color='black', lw=.7)
            axes[row, col].axvline(10, color='gray', ls='--', lw=.7)
            axes[row, col].set_title(f'{kind} HR: {channel}, constant target 130/70kPa')
            axes[row, col].set_ylabel('Pressure - reference [kPa]')
            axes[row, col].grid(alpha=.25)
            axes[row, col].legend()
    for ax in axes[-1]:
        ax.set_xlabel('Policy time [s]')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'frozen_checkpoint_hold_comparison.png', dpi=160)
    plt.close(fig)
    report = dict(training_performed=False, identical_reference_within_type=True,
                  scope='Frozen 1500 vs2000 checkpoints, same nominal simulator/reference/initialization; single seed.',
                  runs=results,
                  conclusion='No universal winner: checkpoint1500 improves type1 held-positive variation but worsens bank negative tracking; checkpoint selection alone does not remove the type2 positive ripple.')
    (OUTPUT / 'frozen_checkpoint_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    for entry in results:
        print(entry['type'], entry['checkpoint'],
              'bank RMSE', [round(entry['bank']['tracking'][c]['whole']['rmse'], 4) for c in ('pos', 'neg')],
              'hold POS range', [round(x, 3) for x in entry['holds']['vibration']['pos']['held_target_pressure_ranges_kpa']],
              'physical', entry['bank']['physical_checks_passed'],
              'max ch+', max(entry['bank']['operating']['ch_pos_maximum_excess_above_220_kpa'], 0))


if __name__ == '__main__':
    main()
