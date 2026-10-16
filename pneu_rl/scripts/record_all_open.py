"""Record pump-on/all-open pressures through an already-running TCP bridge.

Run alone: quick-viz/retrain must not write ctrl_act.json concurrently.
Does not start/stop the pump or TCP bridge. Leaves all six valves open.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from pneu_env.real.real_act import PneuRealAct

CONTROL_KEYS = ('pos_ctrl', 'neg_ctrl', 'act_pos_ctrl1', 'act_pos_ctrl2',
                'act_neg_ctrl1', 'act_neg_ctrl2')


def record(seconds=60.0, output=None):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be positive and finite')
    backend = PneuRealAct(freq=50, scale=False)
    if output is None:
        output = Path(backend.labview_path) / (
            'all_open_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
    output = Path(output)
    # Exclusive creation prevents accidental replacement of existing measurements.
    with output.open('x', newline='') as stream:
        writer = None
        start = time.monotonic()
        previous = None
        count = 0
        next_report = 0
        print(f'All-open recording: {seconds:g}s, 50Hz target -> {output}', flush=True)
        try:
            while time.monotonic() - start < seconds:
                _, info = backend.observe(np.ones(6), np.array([101.325, 101.325]))
                now = time.monotonic()
                row = dict(elapsed_s=now-start, received_unix_s=time.time(),
                           command_elapsed_s=backend._sent_time,
                           sample_dt_s='' if previous is None else now-previous,
                           **info['Observation'])
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                previous = now
                count += 1
                if now-start >= next_report:
                    stream.flush()
                    print(f'{now-start:.1f}s / {seconds:g}s, {count} samples', flush=True)
                    next_report += 5
        finally:
            for key in CONTROL_KEYS:
                setattr(backend, key, 1.0)
            backend.write_ctrl_file()
            stream.flush()
            print(f'Saved {count} samples. All-open command retained: {output}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=60.0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        record(args.seconds, args.output)
    except KeyboardInterrupt:
        print('Interrupted; partial CSV retained.')
