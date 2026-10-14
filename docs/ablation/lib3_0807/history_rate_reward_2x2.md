# lib3 history/rate × reward 2×2 experiment

## Question

Does pressure-rate observation improve control because it supplies a finite-SAC coordinate bias, or because the old reward does not directly penalize phase/rate error near a reference plateau?

## Fixed protocol

- Plant: lib3, 50 Hz, 0.1 s sensor delay, 6-valve pure RL.
- Observation history: 11 pressure samples (`num_prev=10`) and 10 prior six-valve actions.
- Training: 2000 episodes, horizon 512, same 16-member capacity-reference profile, carry reset every four blocks, seeds 0/1/2.
- No MPObs, CAPS, reserve observation, integral-error observation, future reward, action reward, or conflict reward.
- All four arms have the same chamber-bound coefficient 0.10. This is a safety envelope, not an observation feature. Historical 1000-episode models without this bound are not primary arms of this new factorial experiment.

| Observation | Base objective | Phase-aware rate objective |
|---|---|---|
| H: pressure/action history | H-base | H-smooth |
| HR: H plus explicit 4-channel pressure rate | HR-base | HR-smooth |

The base objective is absolute current actuator-pressure tracking plus the identical chamber bound:

\[
r_t^{\rm base}=-0.3(|e_{+,t}|+|e_{-,t}|)-0.10\,b_{ch,t}.
\]

The smooth arm adds only reference-rate tracking:

\[
r_t^{\rm smooth}=r_t^{\rm base}
-0.025\sum_{c\in\{+,-\}}
\left|\frac{\dot P_{act,c}-\dot r_c}{s_c}\right|,
\quad (s_+,s_-)=(12,6)\;\mathrm{kPa/s}.
\]

Raw \(\lvert\dot P\rvert\) is deliberately not penalized: it would penalize physically necessary motion during a reference ramp. The action first/second difference coefficients are zero in this experiment so rate reward is the only reward difference. They become a separate follow-up ablation only if this 2×2 identifies a rate-objective effect. The coefficient 0.025 is preselected from the old H full-bank audit: it makes the added rate term approximately 10% of the base penalty on plateau rollouts, rather than the approximately 21% produced by 0.05.

### Pre-training reward-scale audit

This calibration uses the 16 held-out full-bank rollouts of the existing H-base
seed-0 policy (32,768 control steps). A plateau is a post-startup sample with
both reference-rate magnitudes below 1.0 kPa/s (POS) and 0.5 kPa/s (NEG).
The common chamber-bound penalty was zero on these plateau samples, so it does
not set the coefficient.

| Quantity on plateau | Value |
|---|---:|
| Mean base penalty \(\bar b\) | 2.1211 |
| Mean normalized rate loss before \(\lambda\), \(\bar\ell_{rate}\) | 9.0340 |
| \(\lambda=0.05\): added mean / base mean | \(0.4517/2.1211=21.3\%\) |
| \(\lambda=0.025\): added mean / base mean | \(0.2259/2.1211=10.6\%\) |
| \(\lambda=0.025\): 95th-percentile added penalty | 0.5106 |

Thus 0.025 is a deliberately moderate *objective* change, not a claim that
the rate loss has the same relative scale under every policy. In particular,
it is expected to contribute more while H-base oscillates and less after a
policy becomes rate-consistent; this is the intended feedback of the new
objective. The audit is reproducible with
`pneu_rl/scripts/audit_lib3_rate_reward_scale.py`.

## Interpretation

- H-smooth improves over H-base: old objective admitted a phase/ripple failure mode.
- HR-base improves over H-base: explicit rate coordinate improves learning under the old objective.
- HR-smooth improves over H-smooth: rate-coordinate benefit remains at a fixed objective.
- H-smooth and HR-smooth converge to the same outcome: rate is not necessary information; any benefit is finite-budget/sample-efficiency bias.

Primary evaluation is paired full reference-bank RMSE, direction agreement, plateau pressure-rate RMS, plateau action-delta RMS, and chamber-bound violation. Training return is diagnostic only.

## Commands

```bash
bash pneu_rl/scripts/run_lib3_history_rate_reward_2x2.sh dry-run
# Terminal A: H-base -> H-smooth (sequential)
bash pneu_rl/scripts/train_lib3_h_reward_2x2_seed0.sh
# Terminal B: HR-base -> HR-smooth (sequential)
bash pneu_rl/scripts/train_lib3_hr_reward_2x2_seed0.sh
bash pneu_rl/scripts/run_lib3_history_rate_reward_2x2.sh evaluate-seed 0
```

The two row scripts may run concurrently only if the available GPU/CPU capacity
is sufficient; each row itself is sequential. Additional seeds are deferred
until this seed-0 screen identifies the comparison that needs confirmation.

## Seed-0 results (2026-09-01)

All four arms completed 2000 episodes. The final checkpoints were evaluated
deterministically on all 16 members of the same held-out full reference bank,
with 2048 physical steps per member. Error bars in the figure are standard
deviations across reference members, not training seeds.

랩미팅에서 바로 쓸 수 있는 한 페이지 한국어 요약과 H/HR 분리 그림은
[labmeeting_reward_shaping_2000.md](labmeeting_reward_shaping_2000.md)에 있다.

기존 2x2 aggregate 그림은 [summary](../../../pneu_rl/exp/lib3_0831_histreward2x2_analysis/history_rate_reward_2x2_summary.png)와
[trajectory overview](../../../pneu_rl/exp/lib3_0831_histreward2x2_analysis/history_rate_reward_2x2_trajectory.png)에 보관한다.
문서 안에서 보는 controller별 그림은 chamber/POS/NEG의 동일한 3-row format으로
[labmeeting_reward_shaping_2000.md](labmeeting_reward_shaping_2000.md)에만 넣었다.

| Arm | POS RMSE | NEG RMSE | Combined RMSE | Plateau rate RMS POS/NEG | Plateau action-Δ RMS |
|---|---:|---:|---:|---:|---:|
| H-base | 8.238 | 7.394 | 7.836 | 30.290 / 15.897 | .00542 |
| H-smooth | 6.576 | 4.920 | 5.834 | 11.506 / 3.372 | .00174 |
| HR-base | **.461** | 1.406 | 1.047 | **1.459** / 12.410 | .03380 |
| HR-smooth | .844 | **1.170** | **1.025** | 9.806 / **3.340** | .02699 |

The combined RMSE is the reference-member mean of
\(\sqrt{(\mathrm{RMSE}_{+}^{2}+\mathrm{RMSE}_{-}^{2})/2}\). Pressure-rate RMS
is evaluated only on channel-specific plateau samples after the first second.

### Factorial interpretation

1. **The old reward was part of the H oscillation problem, but not the whole
   explanation.** At fixed H observation, smooth reward reduced POS/NEG RMSE
   by 20.2%/33.5%, plateau POS/NEG rate RMS by 62.0%/78.8%, and plateau
   action-Δ RMS by 67.9%. Nevertheless, H-smooth still had 5.834 kPa combined
   RMSE and visibly attenuated/biased trajectories.
2. **The explicit rate coordinate retained a large effect under the same
   smooth objective.** HR-smooth reduced combined RMSE from 5.834 to 1.025 kPa
   relative to H-smooth (82.4%). Thus adding a rate-sensitive objective did not
   make final history-only control catch up within this finite training budget.
3. **The smooth reward moved HR to a different channel-wise trade-off rather
   than uniformly improving every metric.** Relative to HR-base, HR-smooth
   improved NEG RMSE by 16.8%, NEG plateau rate by 73.1%, and action variation
   by 20.2%, but worsened POS RMSE by 83.0% and POS plateau rate. Combined RMSE
   changed only from 1.047 to 1.025 kPa. The normalized rate objective was
   dominated by NEG rate error in HR-base and became more balanced in
   HR-smooth; this explains the redistribution without proving that 0.025 is
   the unique optimal coefficient.
4. **Low action variation alone is not successful control.** H-smooth had the
   smallest action-Δ metric partly because it under-tracked the reference.
   Smoothness must therefore be interpreted jointly with tracking and pressure
   motion, not as an isolated objective.

Training diagnostics add an optimization-stability result. HR-base and
HR-smooth converged and remained near 0.8 kPa combined training RMSE at the end.
Both H arms improved around episodes 1200--1300 and then degraded toward episode
2000. A post-hoc full-bank checkpoint diagnostic gave 4.207 kPa combined RMSE
for H-base episode 1200 and 4.736 kPa for H-smooth episode 1300, still far above
the final HR results. Hence the final-checkpoint gap is not explained solely by
choosing episode 2000 after an earlier equally good H solution.

### Conclusion boundary

This single-seed experiment does **not** prove that explicit rate contains new
information or is mathematically necessary: it remains a deterministic linear
function of pressure history. It does show that, for the tested MLP-SAC,
normalization, training budget, and reference distribution, explicit rate acts
as a strong finite-optimization/training-stability coordinate. Reward
misalignment explains part of H-base oscillation, but it does not explain away
the observation-coordinate effect. A second training seed is required only if
this effect is promoted to a stochastic reproducibility claim.

Machine-readable outputs are in
`pneu_rl/exp/lib3_0831_histreward2x2_analysis/final_checkpoint_metrics.csv` and
`paired_effects.csv`.
