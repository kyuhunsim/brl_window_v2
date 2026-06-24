# 260528 RL vs PID Failure Analysis

## Compared Runs

- RL quick viz: `pneu_rl/exp/260528_00_45_54_0527_lib3_Ours_Simulation`
- PID baseline: `pneu_rl/exp/lib3_pid_baseline_40s`
- Training diagnostics:
  - `pneu_rl/models/0527_lib3_Ours/train_diagnostics.csv`
  - `pneu_rl/models/0527_lib3_sac/train_diagnostics.csv`

## Key Numbers

### Quick Viz / Baseline, 40 s

| run | pos RMSE | neg RMSE | pos bias | neg bias | ch pos mean/end/max | ch neg mean/end/min |
|---|---:|---:|---:|---:|---:|---:|
| `0527_lib3_Ours` quick viz, PID off | 9.18 | 15.76 | 3.62 | -12.21 | 270.96 / 353.28 / 361.18 | 18.18 / 18.50 / 11.36 |
| `0527_lib3_Ours` quick viz, PID on | 6.26 | 4.00 | -0.41 | 0.01 | 158.25 / 149.87 / 184.00 | 17.89 / 24.29 / 10.09 |
| `lib3_pid_baseline_40s` | 6.59 | 4.68 | -2.39 | -3.04 | 170.73 / 173.09 / 185.55 | 18.31 / 18.21 / 12.12 |

The RL quick viz with PID off is worse mainly because the chamber positive pressure runs away to an extremely high value. PID keeps the chamber positive reserve around 170-185 kPa, while RL with PID off allows it to climb above 350 kPa.

When PID is turned on during quick viz, the same `0527_lib3_Ours` model becomes better than the PID baseline on actuator RMSE. So the weak result was not "RL is always worse"; it was mostly a train/eval mismatch plus chamber drift issue.

### Smoothness / Conflict Comparison

| run | act pos sign changes / s | act neg sign changes / s | pos conflict | neg conflict |
|---|---:|---:|---:|---:|
| `0527_lib3_Ours`, PID off | 0.275 | 0.725 | 0.770 | 0.788 |
| `0527_lib3_Ours`, PID on | 2.225 | 2.425 | 0.909 | 0.867 |
| `lib3_pid_baseline_40s` | 0.575 | 0.550 | 0.756 | 0.746 |

This is the main tradeoff:

- `0527_lib3_Ours` with PID on gives the best actuator tracking.
- `lib3_pid_baseline_40s` gives smoother actuator pressure and lower in/out conflict.
- `0527_lib3_Ours` with PID off fails because it was not trained for that standalone setting.

### Training Diagnostics, Last 100 Episodes

| model | reward | pos RMSE | neg RMSE | pos bias | neg bias | ch pos end/max | ch neg end/min | pos conflict | neg conflict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0527_lib3_Ours` | -3364.61 | 9.72 | 3.97 | 1.71 | 1.13 | 139.80 / 153.58 | 11.70 / 6.99 | 0.826 | 0.804 |
| `0527_lib3_sac` | -5404.53 | 31.37 | 10.52 | -3.50 | 0.65 | 204.91 / 221.20 | 29.24 / 17.82 | 0.834 | 0.921 |

`0527_lib3_Ours` learns something useful in training, especially on the negative actuator pressure. Plain `0527_lib3_sac` is much worse and drifts toward a high positive chamber regime.

## Main Diagnosis

### 1. Train/Eval PID Mismatch Is The Biggest Immediate Issue

`0527_lib3_Ours/cfg.yaml` contains a non-null `pid` block. In `train3.py`, this means:

```python
if kwargs["pid"] is not None:
    env.set_pid(**kwargs["pid"])
```

So `0527_lib3_Ours` was trained with PID assist inside the environment.

But the quick viz run printed:

```text
[ INFO] PID: Off
```

`quick_viz3.py` only calls `env.set_pid(...)` when the user selects PID on. Therefore the model was trained in one closed-loop system and evaluated in another. This can easily make it worse than PID alone, because the policy may have learned actions assuming the PID loop is also acting.

Interpretation:

- `0527_lib3_Ours` is not a pure RL valve controller.
- It is closer to a PID-assisted / residual-style controller during training.
- Evaluating it with PID off is an unfair and behavior-changing test.

This sanity check was confirmed by the new quick viz comparison:

- PID off: clearly worse than baseline.
- PID on: actuator RMSE becomes better than baseline.

So the question is no longer "why is RL always worse than PID?" The real question is:

```text
why does RL + PID track better, but with more oscillation/conflict than the baseline PID?
```

### 2. Ref Range Difference Is Not The Main Cause

Training random ref:

```yaml
pos_min_off: 115
pos_max_off: 140
pos_max_amp: 10
pos_max_ts: 5
neg_min_off: 60
neg_max_off: 80
neg_max_amp: 5
neg_max_ts: 5
```

Quick viz random ref:

```python
pos_min_off = 115
pos_max_off = 140
pos_max_amp = 10
pos_max_ts = 10
neg_min_off = 60
neg_max_off = 80
neg_max_amp = 5
neg_max_ts = 10
```

The pressure range is effectively the same. Quick viz changes more slowly, so it should not be harder. The larger difference is not actuator ref; it is chamber handling and PID on/off consistency.

### 3. Chamber Is An Uncontrolled Internal State For RL

PID baseline explicitly manages chamber reserve. RL reward mainly tracks actuator pressure, so the policy can use chamber pressure as an unpenalized hidden resource. In quick viz, chamber positive rises above 350 kPa. That is a strong sign that the policy is exploiting or ignoring chamber dynamics.

This explains why a policy can look acceptable for a short training episode and then degrade in a longer rollout.

### 4. Direct 6D Valve Action Is Still Too Ambiguous

Both learned policies keep many valves partially open at the same time. Even the better `0527_lib3_Ours` with PID on has higher conflict than the PID baseline:

- `0527_lib3_Ours` last100 training conflict: pos 0.826, neg 0.804
- `0527_lib3_sac` last100 conflict: pos 0.834, neg 0.921
- `0527_lib3_Ours` quick viz with PID on: pos 0.909, neg 0.867
- `lib3_pid_baseline_40s`: pos 0.756, neg 0.746

This is high common-mode valve behavior. It does not necessarily mean "physically impossible", but it means the policy is not cleanly resolving which valve should carry the pressure change.

The valve opening sweep also showed that the effective opening is very nonlinear around 0.90-0.95. Small action differences can have a large pressure effect, while lower commands may look active but produce little useful flow. SAC in raw 6D action space has to learn this allocation and nonlinear valve map at the same time.

### 5. Network Size Is Probably Not The First Bottleneck

The current `hidden_dim=256` is not obviously too small. The failure pattern is more consistent with:

- train/eval mismatch,
- missing chamber objective,
- redundant 6D action ambiguity,
- nonlinear valve opening,
- short-horizon training vs longer rollout drift.

Changing the network first may improve randomness, but it probably will not solve the root cause.

## What To Try Next

### Step 1: Decide Which Success Criterion Matters First

Right now there are two different notions of "better":

- lower actuator tracking error,
- smoother / more interpretable valve behavior.

At the moment:

- `0527_lib3_Ours` with PID on wins on tracking,
- PID baseline wins on smoothness and lower conflict.

So the next work should focus on reducing RL-induced conflict and oscillation, not on proving that RL can beat PID at all. It already can, in the PID-on setting.

### Step 2: Treat `0527_lib3_Ours` As Residual RL, Not Pure RL

This is the most honest interpretation of the current architecture.

Training:

```text
u_applied = clip(u_RL + u_PID)
```

Evaluation:

```text
PID must stay on
```

Then improve the residual policy so it helps the PID rather than constantly pushing all four actuator valves high.

### Step 3: Add RL Penalties For What PID Already Does Well

The baseline is smoother because it has a simpler structure. RL needs explicit pressure not to misuse its extra freedom.

Most useful next penalties:

- actuator valve delta penalty,
- actuator in/out common-mode penalty,
- chamber positive over-reserve penalty,
- chamber negative reserve deviation penalty.

Without these, RL can reduce tracking error while creating ugly valve behavior.

### Step 4: Do Not Change Network Size First

`hidden_dim=256` is not the bottleneck suggested by the current data.

The current gap is better explained by:

- action redundancy,
- chamber not being part of the main control objective,
- RL pushing common-mode valve openings,
- PID-assisted training dynamics.

So the next change should be reward/action structure, not a larger MLP.

### Step 5: If You Want Pure RL, Start A Clean No-PID Branch

The current `0527_lib3_Ours` should not be used to judge pure RL.

For pure RL:

```text
pid: null
evaluate with PID off
```

But then the controller should probably not stay as raw direct 6D SAC. The better direction is:

```text
policy -> low-dimensional pressure-rate / reserve command
allocator -> 6 valve commands
```

### Step 6: Practical Direction For You

If the near-term goal is "make RL look clearly better than PID in sim", the fastest route is:

1. keep PID on,
2. treat RL as residual,
3. add smoothness/conflict/chamber penalties,
4. compare against PID baseline on both RMSE and conflict.

If the research goal is "pure RL can solve lib3 alone", then do a separate branch with no PID and a lower-dimensional action design.

### Step 3: Add Chamber Terms If Doing Pure RL

Minimum reward additions:

```text
- chamber positive over-reserve penalty
- chamber negative under/over-reserve penalty
- chamber drift penalty
- action delta penalty
- common-mode / unnecessary simultaneous opening penalty
```

The chamber penalty should be weaker than actuator tracking, but nonzero. Without it, the policy can make actuator tracking temporarily better while destroying the chamber state.

### Step 4: Do Not Stay In Raw 6D Action Forever

For successful RL in this simulator, the better structure is:

```text
policy -> low-dimensional desired pressure-rate / reserve command
allocator -> 6 valve commands
```

Good first version:

```text
z = [act_pos_rate_cmd, act_neg_rate_cmd, reserve_cmd]
```

The allocator handles:

- chamber reserve,
- in/out valve selection,
- valve saturation,
- action smoothness.

Then SAC learns a meaningful control command instead of learning six nonlinear valve openings from scratch.

## Practical Experiment Order

1. Run quick viz for `0527_lib3_Ours` with PID on.
2. Train `0528_lib3_Ours_noPID` with the same observation/prediction setup but `pid: null`; evaluate with PID off.
3. Train a clear residual version and always evaluate with PID on.
4. Add chamber reward terms to the no-PID version.
5. If direct 6D still fails, implement 2D/3D latent action with a valve allocator.

## About `train_diag_atm_band`

`train_diag_atm_band` is diagnostic only. It is used to compute:

```python
abs(act_pressure - ATM) < train_diag_atm_band
```

It does not affect reward, policy action, simulator dynamics, or training updates.
