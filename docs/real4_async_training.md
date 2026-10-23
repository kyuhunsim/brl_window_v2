# Real4 asynchronous SAC retraining

`pneu_rl/scripts/retrain4.py` uses process-isolated learning when Observation
Mode is Real (2). Simulation retains the synchronous training path. Run the
existing TCP bridge separately as before. No RT/FPGA protocol changes are needed.

```text
Control process: inference -> env.step -> nonblocking experience queue
                       ^                            |
                       |                            v
              policy snapshot             Learner process
                                  replay insertion / SAC updates / logs / saves
```

The learner starts using `spawn`; the entrypoint is guarded by `main()` so the
child does not repeat prompts or initialize hardware. Only environment dimensions
are passed to the learner, not the real environment. Model/optimizer state is
serialized into private worker storage. The actor retains its own policy.

Real warmup transitions also go directly to the learner, with updates disabled.
After warmup, existing `start_epi`, `update_during_warmup`, `update_every`, and
`epoch` rules determine updates. Learning is paced by received transitions;
there is no unlimited background update loop. Experience order is preserved.

Defaults recorded in `retrain_cfg.yaml`:

- `queue_capacity: 4096`: bounded pending messages (transitions and episode markers).
  A full queue raises an error and closes control. The rejected message is retained
  by the actor and submitted after control closes, before the remaining queue drains.
- `sync_every: 5`: publish when the completed update count reaches a multiple of
  five at an update-group boundary. The actor checks for a new snapshot before
  inference. If the snapshot lock is busy, it keeps its previous policy.

Replay sampling, gradient computation, episode diagnostics, model checkpoints,
and buffer serialization run in the learner. Episode output may therefore lag
physical execution. `async_timing.csv` records actor step-start intervals, including
the first step's elapsed time. Preparation/reset time is excluded between episodes.
These are host-loop intervals, not FPGA application timestamps or TCP RTTs.

On completion or interruption, the existing `env.close()` runs before waiting
for queued work and final checkpoint/buffer saving (180-second default timeout).
On interruption/error, queued transitions enter replay without further gradient
updates, so shutdown does not spend its save budget training on the backlog.
The original control error is preserved if cleanup also fails. Worker failures
propagate to the caller. A timeout/forced termination can still leave pending
experience unsaved; do not treat the last checkpoint as containing every step.
Checkpoints, info files and replay snapshots replace their previous versions only
after the new file has been written. `async_state.json` records the final processed
step, completed episode and skipped pending updates. A partial episode checkpoint
uses `model_step_<step>.pth` rather than a completed-episode filename.
The parent model's replay buffer and critic are not synchronized back; saved
learner checkpoints are the authoritative trained state. Its policy receives the
final snapshot after a successful drain.

This removes synchronous training/storage stalls but does not guarantee hard
real-time 50 Hz: JSON transport, inference, prediction, CPU contention, and OS
scheduling still contribute. Existing real4 shutdown behavior is reused; this
change does not add an RT/FPGA watchdog or command acknowledgement.

Hardware-free checks:

```bash
source ./activate_env.sh
python -m unittest discover -s pneu_rl/tests -v
```

Tests cover actual spawned SAC updates, warmup accounting, independent policy
storage, snapshot/checkpoint agreement, slow learner isolation, queue overflow,
and environment closure on learner failure. Real hardware timing must be measured
separately; the synchronous `retrain4_hz_test.py` remains a baseline tool.

## Lib3 entrypoints

`retrain3.py` now uses this same learner in Real mode. It loads either a modern
`reference` specification or legacy `rnd_ref`, and starts a fresh replay buffer
without loading/copying the parent buffer. Retrain3 preserves the selected model's reward and CAPS settings; it does not
force action-delta/conflict penalties or a temporal-weight schedule.

For the August 31 H/HR checkpoints, `quick_viz3.py` reference option **6** (also
the blank-input default) selects **bank trajectory 0**, pins it across resets,
and evaluates its first **40 seconds without repetition**. The stored trajectory
is 40.96 seconds long, leaving room for the models' future reference window.
This evaluation override does not change the saved training bank or retraining's
coverage/reset behavior.

Quick-viz loads observation and predictor settings from the selected model.
The lib3 real wrapper (`real9.py`) now supports `scale_low` / `scale_high`, with
physical command defaults `0.8 + 0.2*a` on all six channels. An explicitly
configured `scale_low=0.85` retains that alternative range. The backend's supported
scaled range is [0.8, 1.0]. PID OFF evaluates the pure RL checkpoints.

Run from the repository root, with the existing TCP bridge in a separate terminal:

```bash
source ./activate_env.sh
python pneu_rl/scripts/quick_viz3.py  # REF: 6 or Enter; OBS: 2; PID: 2
# Or, for real fine-tuning:
python pneu_rl/scripts/retrain3.py   # OBS: 2; PID: 2
```

Lib3 real shutdown now publishes six physical commands of 1.0 directly, bypassing
PID and policy scaling, with no sensor wait. Quick-viz closes once in the rollout
finally block before saving/plotting, including Ctrl+C and runtime errors. This is
one JSON command publication; the separately running TCP bridge continues to send
the latest values. Publication does not establish FPGA receipt or valve position.

## Lib3 queue overflow recovery — 2026-10-23

The interrupted `261023_type2_HR_smooth_retrain` run retained a readable model and
info record through episode 2107 (1,078,784 steps), and both replay files through
episode 2100 (51,200 real transitions). The actor's recent mean period was about
20.08 ms; an offline benchmark of the saved model with that replay took a median
21.5 ms per gradient update, excluding episode logging and saving. One update per
transition could not keep up over the long run. The earlier 512-step checks did
not exercise this accumulated backlog.

Real `retrain3.py` now defaults to one update per five transitions, while retaining
50 Hz control. `--update-every` overrides that choice. This changes the update/data
ratio; the learning rate and reward are preserved. Each async episode log includes
`pending=...`, and `async_backlog.csv` records pending messages and the update ratio.

To continue the saved weights and replay in a **new folder** for the remaining
93 episodes (2108–2200), run the following from the project environment:

```bash
python pneu_rl/scripts/retrain3.py \
  --model 261023_type2_HR_smooth_retrain \
  --name 261023_type2_HR_smooth_retrain_recovered \
  --obs real --pid off --episodes 93 --update-every 5 --load-buffer
```

This starts a new 30-second measured preparation and restarts the continuous bank
reference. It restores the latest saved weights/optimizer and available replay;
it does not recreate the unsaved post-2107 interactions or the original physical
state. If `async_state.json` is present after a newer interrupted run, its final
processed step count is restored along with the replay. Without `--load-buffer`,
fine-tuning continues to start a fresh replay.
