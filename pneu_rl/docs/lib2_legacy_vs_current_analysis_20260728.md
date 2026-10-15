# lib2 과거 우수 사례와 2026-07 screening 비교

## 결론

현재 screening 결과가 과거보다 퇴행해 보이는 판단은 타당하다. 다만 원인을 SAC architecture 하나로 단정할 수 없다. 현재와 과거는 reference, 초기압력, 평가시간, PID/I-term, action 사용 범위가 동시에 달라졌다.

현재 결과는 `낮은 압력 범위의 reference를 평균 operating point에서 거의 고정 추종하는 현상`이 강하다. 과거 `Ours`는 큰 압력 범위와 대기압 초기조건에서 POS/NEG pressure를 실제로 움직이며 reference waveform을 따라간다.

## 조건 비교

| 항목 | 과거 260517 Ours | 현재 260727 screening |
|---|---|---|
| 초기압력 | POS/NEG 101.325 kPa | POS 140, NEG 25 kPa |
| 평가시간 | 약 100 s | 약 40.84 s |
| POS reference | 약 136–204 kPa | 약 134.5–149.4 kPa |
| NEG reference | 약 6–34.9 kPa | 약 21.4–26.3 kPa |
| PID/I-term | `Ki_pos=Ki_neg=0.01`, `Ka=10` | pure 모델은 off, I-term 모델은 on |
| action waveform | 약 0.84–1.0에서 크게 변화 | 일부 모델은 0.97–1.0 부근 고정 |
| reference 종류 | legacy RandomRef | FeasibleRandomRef |

현재 reference는 pressure 변화 폭 자체가 작고, 초기압력이 reference operating point에 이미 가깝다. 따라서 초반에 큰 pressure movement를 만들 필요가 없다. 이 조건은 policy가 action 방향을 탐색하지 않고 평균점에 머무르는 것을 쉽게 만든다.

## 과거 Ours 결과

파일:

- `exp/051lib2_Ours/260517_20_06_04_051lib2_Ours_Simulation`

cfg에는 다음 보조항이 명시되어 있다.

```yaml
Ki_pos: 0.01
Ki_neg: 0.01
Ka: 10
```

따라서 이 결과는 pure RL baseline이 아니라 I-term-assisted 결과다. 그래도 현재보다 waveform이 좋은 이유는 명확하다.

1. 초기 대기압에서 POS를 약 180–200 kPa까지 올리는 큰 tracking task를 수행한다.
2. POS/NEG reference의 amplitude와 range가 크다.
3. action이 `.84–1.0` 사이에서 실제로 움직인다.
4. I-term이 누적오차를 이용해 느린 pneumatic dynamics를 보정한다.

전체 100초 기준 metrics는 POS RMSE `12.37 kPa`, NEG RMSE `4.75 kPa`다. 첫 40.96초만 잘라 계산하면 POS RMSE `12.60 kPa`, NEG RMSE `6.20 kPa`다. 현재 I-term 모델의 40.96초 RMSE보다 수치가 반드시 낮지는 않지만, 과거 Ours는 훨씬 큰 reference excursion을 실제로 따라가는 waveform이라는 점이 중요하다.

## 과거 SAC 결과

파일:

- `exp/0515_lib2_sac/260517_15_56_17_0515_lib2_sac_Simulation`

이 결과도 evaluation cfg에 `Ki_pos=Ki_neg=0.01`이 들어 있다. 따라서 cfg만으로는 PID가 검증에서 완전히 off였다고 확정할 수 없다.

또한 전체 metrics는:

- POS RMSE `48.77 kPa`
- NEG RMSE `4.61 kPa`

마지막 구간에서는 실제 POS가 약 `287 kPa`, reference가 약 `214 kPa`까지 벌어진다. 따라서 waveform이 dynamic해 보이는 것과 안정적으로 tracking했다는 것은 다르다. 이 모델은 과거 pure SAC의 확실한 성공 근거로 사용하면 안 된다.

## 현재 reference 설계의 문제 가능성

현재 `FeasibleRandomRef`는 다음을 보장한다.

- pressure 범위
- 축별 pressure-rate 제한
- transition duration
- frequency 제한

하지만 이것만으로 joint plant feasibility가 보장되지는 않는다.

현재 capacity 산정은 여러 state/action sweep에서 POS up/down, NEG up/down의 축별 최대값을 추출한다. 각 축의 rate가 가능하더라도 동일한 action pair와 동일한 pump/valve internal state에서 두 rate를 동시에 만들 수 있다는 뜻은 아니다.

따라서 현재 reference는 다음 검증이 필요하다.

1. POS/NEG pressure state뿐 아니라 pump internal state를 포함한 상태에서 sweep한다.
2. 각 reference rate vector에 대해 실제 action pair가 존재하는지 확인한다.
3. episode 중간 internal state에서 maximum achievable pressure rate를 다시 측정한다.
4. reference derivative와 reachable joint rate set을 직접 비교한다.

현재의 `joint_ref_rate_violation_ratio=0`은 축별 capacity profile에 대한 결과이지, 모든 plant internal state에서의 엄밀한 reachability 증명은 아니다.

## 현재 모델 결과의 해석

현재 낮은-range reference 평가에서 낮은 RMSE가 나와도 pressure waveform을 실제로 추종했다는 뜻은 아니다.

- 실제 pressure가 reference 평균 근처에 고정될 수 있다.
- 낮은 reference amplitude 때문에 오차가 작게 보일 수 있다.
- action이 거의 움직이지 않아도 RMSE가 크게 증가하지 않을 수 있다.
- I-term은 이 고정점 오차를 누적 보정한다.

따라서 모델 평가는 RMSE 하나가 아니라 다음을 함께 사용해야 한다.

- pressure waveform shape
- reference 상승/하강 구간별 RMSE
- bias
- action waveform과 saturation
- pressure slew와 reference slew
- valve flow
- pressure direction change 대응률
- episode별 발산률

특히 `ref가 상승할 때 action이 상승 방향으로 바뀌었는가`, `ref가 하강할 때 반대 action을 선택했는가`를 별도 지표로 기록해야 한다.

## 다음 실험 순서

### 1. 먼저 evaluation 조건을 고정한다

현재 reference와 과거 legacy reference를 각각 파일로 고정하고, 동일한 reference trajectory를 모든 모델에 제공한다. 초기조건도 다음 두 조건을 모두 평가한다.

- ATM 초기조건: `(101.325, 101.325)`
- operating-point 초기조건: `(140, 25)`

각 조건은 `carry=False`와 `carry=True`를 섞지 않고 별도로 기록한다.

### 2. RL 학습 전에 reference feasibility를 다시 검사한다

특히 10–15초와 같이 reference가 내려가는데 action이 상한에 있는 구간을 replay한다. 동일한 simulator internal state에서 다음 action pair를 비교한다.

```text
(0.8, 0.8), (1.0, 0.8), (0.8, 1.0), (1.0, 1.0)
```

이 결과로 reference 문제가 있는지, policy가 action을 잘못 선택한 것인지 분리한다.

### 3. 현재 낮은-range reference를 다시 학습하지 않는다

현재 결과만으로는 낮은-range reference가 pure RL의 capability를 충분히 시험하지 못한다. 먼저 과거와 유사한 wide-range reference를 만들되, legacy RandomRef의 불연속 jump는 그대로 사용하지 않고 transition만 rate-limited하게 만든다.

권장 1차 profile:

```text
POS: 140–210 kPa
NEG: 10–32 kPa
초기압력: 101.325/101.325
transition: simulator joint reachable rate 기반
```

### 4. 그 후 pure SAC부터 다시 비교한다

순서는 다음이 적절하다.

1. pure SAC, wide reference, ATM reset
2. pure SAC, wide reference, operating-point reset
3. SAC+CAPS
4. SAC+MPObs
5. I-term model을 assisted upper bound로 비교

이 단계에서 waveform shape가 과거 Ours처럼 움직이는지 먼저 확인하고, 그 다음 RMSE를 비교한다.

## 현재까지의 판정

현재 실패의 원인은 단일 항이 아니다.

- 낮은 reference amplitude와 operating-point 초기조건이 탐색을 약화시켰을 가능성이 크다.
- reference의 축별 rate feasibility가 joint/internal-state feasibility를 보장하지 않는다.
- CAPS는 dynamic action을 줄일 수 있다.
- MPObs는 현재 implementation에서 action/prediction 구조를 함께 바꾼다.
- I-term은 느린 dynamics와 temporal credit assignment 문제를 보완한다.

따라서 다음 단계는 `.85 action bound`를 먼저 적용하는 것이 아니라, 과거와 같은 wide-range/ATM 조건을 복원한 controlled comparison과 joint reachability 검증이다.

## 2026-07-28 재학습 없는 추가 검사

현재 모델을 재학습하지 않고 초기압력과 reference만 바꾸어 평가했다. 이 검사는 모델의 out-of-distribution 성능을 보는 것이므로 학습 성공률로 해석하지 않고, 현재 모델이 어느 operating point에 묶여 있는지 확인하는 용도로 사용한다.

### 현재 6개 모델 + wide smooth reference + ATM 초기압력

사용 profile: `config/reference/lib2_feasible.yaml`  
초기압력: `101.325/101.325 kPa`  
평가시간: `40.84 s`

| 모델 | POS RMSE | NEG RMSE | 해석 |
|---|---:|---:|---|
| SAC | 61.79 | 59.22 | operating-point 학습에 강하게 의존 |
| SAC+CAPS | 60.66 | 44.92 | action smoothing만으로 초기 상태를 회복하지 못함 |
| SAC+MPObs+CAPS | 42.94 | 46.03 | 일부 변화는 있으나 기준점 회복 실패 |
| SAC+MPObs+CAPS+I-term | 21.93 | 43.18 | POS는 보정하지만 NEG에서 큰 bias |
| History SAC | 39.36 | 55.57 | action variation은 있으나 안정적인 초기 recovery 실패 |
| MPObs1 SAC | 44.08 | 42.20 | smooth하지만 ATM 조건에서는 충분히 움직이지 못함 |

이 결과는 현재 모델들이 `140/25` 주변에서 학습된 정책이라는 것을 보여준다. 그러나 이것만으로 wide reference가 불가능하다고 결론 내릴 수는 없다. 현재 정책의 초기 상태 일반화 실패와 reference feasibility는 별도 문제다.

### 현재 6개 모델 + historical legacy reference + ATM 초기압력

사용 profile: `config/reference/lib2_legacy_easy.yaml`

legacy reference는 RandomRef의 offset/step 변경을 포함하므로 reference slew가 매우 크다. 이번 평가에서는 POS/NEG reference rate violation이 약 `0.26`, joint 최대 utilization이 약 `191`까지 나왔다. 따라서 이 reference는 현재 feasible profile 기준으로는 물리적으로 feasible하지 않다.

그럼에도 역사적 Ours가 이 모양을 어느 정도 따라간 것은 큰 pressure excursion과 I-term/기존 architecture가 함께 작동했기 때문일 수 있다. legacy reference를 “쉬운 reference”라고 부르면 안 된다.

### fixed feed-forward + I-term 검사

별도 PID-only 진단도 수행했다. `Ki_pos=Ki_neg=0.01`, `Ka=10`, 고정 normalized feed-forward `(0.5,0.5)`를 사용했다.

| reference | 초기압력 | POS RMSE | NEG RMSE |
|---|---|---:|---:|
| operating point | ATM | 20.38 | 37.16 |
| wide smooth | ATM | 36.56 | 41.89 |
| legacy random | ATM | 31.94 | 7.45 |

이 결과로 확인할 수 있는 것은 두 가지다.

1. 현재 operating reference가 PID로 자동으로 잘 제어된다고 아직 말할 수 없다.
2. fixed feed-forward 값 하나만 사용한 PID 검사는 과거 Ours의 RL feed-forward를 재현한 것은 아니므로, PID 자체의 한계로 단정할 수 없다.

따라서 다음 PID 검사는 고정 base action 하나가 아니라 base command sweep 또는 과거 Ours의 실제 action feed-forward를 재생해야 한다.

### 평가 산출물

대체 reference 평가 결과는 각 model 아래에 저장되었다.

```text
exp/<model>/screen_eval_wide_atm_2048step/
exp/<model>/screen_eval_legacy_atm_2048step/
```

PID-only 결과는 다음에 저장되었다.

```text
exp/_analysis/pid_reference/<tag>/
```

## 우선순위 재정리

현재는 바로 재학습하지 않고 아래 순서로 판정한다.

1. 과거 Ours의 실제 training/evaluation reference와 carry 여부를 복원한다.
2. 현재 모델을 같은 reference와 ATM 초기조건에서 평가한다.
3. PID-only가 아니라 `PID + feed-forward command`의 base command sweep을 수행한다.
4. wide reference에서 reference derivative와 joint reachable action pair를 비교한다.
5. 그 결과 reference가 제어 가능하다고 확인된 뒤에만 재학습한다.

재학습 시 첫 비교는 다음 4개로 제한한다.

1. pure SAC + wide reference + ATM + reset
2. pure SAC + operating reference + ATM + reset
3. I-term upper bound + wide reference + ATM + reset
4. 과거 Ours 구성 재현 + wide reference + ATM + reset

이 네 개가 끝난 뒤 CAPS와 MPObs를 각각 추가한다. 현재처럼 낮은 reference에서 CAPS/MPObs를 먼저 비교하면 `reference가 너무 쉬움`, `초기점이 다름`, `CAPS가 action을 고정함`을 분리할 수 없다.
## 2026-07-28 baseline PID follow-up

The baseline controller was evaluated with the same simulator and with the
reference profile selectable from the command line.  The results were:

| Reference / initial pressure | POS RMSE [kPa] | NEG RMSE [kPa] |
|---|---:|---:|
| Legacy random, 100 s | 11.61 | 8.31 |
| Operating-point profile, atmospheric initialization | 6.01 | 9.87 |
| Operating-point profile, 140/25 kPa initialization | 5.29 | 8.23 |
| Wide smooth profile, atmospheric initialization | 22.37 | 43.71 |

The legacy random result is strong evidence that the old reference distribution
was controllable by the baseline PID.  It does not, by itself, prove that the
new reference is physically impossible: the same baseline PID also tracks the
operating-point profile reasonably well.  The more defensible conclusion is
that the current RL training distribution is poorly conditioned for learning:
the fixed 140/25 kPa reset removes the atmospheric-to-operating-point transient,
and the narrow operating region does not expose the policy to the pressure and
valve combinations needed outside that region.

The custom PID diagnostic must not be compared numerically with this baseline
table without care.  The baseline uses different PID gains and action logic
from the custom fixed-feed-forward/I-term diagnostic.  Therefore, the baseline
controller result is the appropriate reference for the present comparison.

### Updated priority

1. Fix the reference and initialization experiment first.  Use a wide,
   rate-limited reference and atmospheric initialization for the main training
   condition; evaluate the 140/25 kPa operating-point reset as a separate
   condition, not as the only condition.
2. Keep `carry=False` while establishing this baseline.  Otherwise the
   inter-episode state and the reference distribution are confounded.
3. Re-test SAC and the I-term model on the identical reference/init conditions.
   This separates reference/reset effects from policy-learning effects.
4. Reduce or remove CAPS only after the above comparison.  The nearly straight,
   saturated control trajectories indicate that CAPS may be suppressing the
   action changes needed for tracking, but this must be tested with the same
   reference and initialization.
5. Treat MPObs step initialization as a later ablation; it is not the first
   suspect until reference and reset effects are controlled.

## Why the current pure SAC can fail even when PID succeeds

The baseline PID has integral memory.  A persistent pressure offset therefore
changes the accumulated integral term and gradually shifts the valve command.
Pure SAC has no equivalent mechanism unless the observation contains enough
history or the policy learns an internal action bias from the replay data.

The current screening configuration also has several learning difficulties:

- `num_prev=0` for pure SAC, so the policy mainly sees the current pressure and
  reference.  It cannot directly distinguish the same pressure while rising
  from the same pressure while falling, even though the valve/pump internal
  state and delay make those cases different.
- `delay=0.1` s adds five simulator steps of delay.
- `gamma=0.9` at 50 Hz gives an effective discounted horizon of roughly ten
  steps, or 0.2 s.  Pressure changes that take several seconds receive very
  little credit.
- `start_epi=10` is not uniform random exploration in the current runner.  It
  repeats the configured safe warm-up action.  After that, the initial policy
  standard deviation is small because `log_std` starts near the midpoint of
  `[-10, 1]`.
- Automatic entropy tuning regulates entropy; it does not guarantee that the
  policy visits the valve combinations needed to reverse a pressure bias.

Thus the result is not contradictory: PID demonstrates that a feedback law can
track the operating-point reference, while SAC can still fail to discover that
law because of partial observability, delayed credit assignment, insufficient
action coverage, and a biased reference/reset distribution.  The steady-state
error visible in the PID plot is a symptom of limited integral authority and
actuator saturation, not a sufficient explanation for the complete SAC failure.

## 2026-07-29 command-first reference redesign

The next experiment starts from the control command, not from an independent
pressure rectangle followed by a smoothing filter. lib2 has a coupled map
`(u_pos, u_neg) -> (P_pos, P_neg)`, so a POS value reachable with one `u_neg`
may not be reachable with another. A valid target must come from a measured
joint command pair.

`map_lib2_control_commands.py` swept physical commands in `[0.85, 1.0]` from
atmospheric initialization for 40.96 seconds. It found 121 command pairs, 106
valid rollouts, and 102 bounded rollouts. The bounded steady-state envelope
was approximately POS `101.26–188.12 kPa` and NEG `2.46–27.00 kPa`, but this
set is not a rectangle. Near `188 kPa` POS occurs around `(0.925,1.0)`, while
`(1.0,1.0)` gives only about `141/26 kPa`.

`config/reference/lib2_joint_reachable.yaml` now contains 33 measured joint
points, POS `115–190 kPa`, NEG `10–27 kPa`, an initial target near
`(150.24,24.33) kPa`, and a maximum waypoint distance of `25 kPa`. Each
`design.command_points` entry keeps a pressure point with its physical
command. This is necessary because smoothing a line between feasible
endpoints can still create intermediate POS/NEG pairs that were never
observed. Smoothness is a secondary dynamic constraint, not feasibility.

With atmospheric physical initialization and the same profile, the PID gate
gave POS RMSE `8.95 kPa` and NEG RMSE `8.45 kPa`. The result is stored in
`exp/lib2_baseline/260729_baseline_pid_joint_full_reachable_atm/`. This does
not prove reachability from every internal pump state, but it shows that the
candidate target is not simply impossible for the simulated plant.

The first command-prior smoke was stable during warm-up, but unconstrained SAC
then produced high-frequency POS/NEG changes. When a joint profile is used,
the runner therefore applies a labelled constrained-command layer: measured
map points for warm-up, nearest measured-point projection, five-step command
hold at 50 Hz (`0.1 s`), and one command-grid increment per update. This is
not pure unconstrained SAC and must be identical across SAC/CAPS/MPObs/I-term
comparisons.

The 12-episode smoke evidence is:

| Condition | Episode 11 POS RMSE | Episode 12 POS RMSE |
|---|---:|---:|
| Old profile, unconstrained action | 487 | 506 |
| Full profile, command layer, `gamma=0.9`, history 0 | 21 | 86 |
| Full profile, command layer, `gamma=0.99`, history 10 | 8.8 | 33.7 |

These are diagnostics, not final training scores. The next fixed comparison is
pure SAC versus SAC+I-term with the command map, reference, ATM initialization,
reward, command layer, and `carry=False` held constant, using `gamma=0.99` and
`num_prev=10`. Only after that should gamma/history, CAPS, MPObs, and carry be
ablated in sequence.

## Live carry/h512 checkpoint inspection

The two currently running models were inspected from their latest saved
checkpoints, not from reward totals alone. Evaluation used the same 33-point
joint reference for 2048 steps (`40.96 s`) and now reproduces the training
command projection and five-step hold.

| checkpoint | POS RMSE | NEG RMSE | POS success (`±5 kPa`) | NEG success (`±5 kPa`) | POS bias | NEG bias |
|---|---:|---:|---:|---:|---:|---:|
| SAC, saved around epi 271 | 4.46 | 7.01 | 97.6% | 76.3% | +1.62 | +2.88 |
| SAC+I-term, saved around epi 263 | 7.63 | 6.71 | 56.9% | 83.9% | -2.87 | -3.31 |

The waveform is more informative than this table. SAC follows the POS
reference's rises and falls and keeps NEG near its moving target, although the
initial atmospheric recovery is large and the action/flow plots show repeated
POS command changes. The I-term model is smoother and has lower NEG error, but
its POS curve stays above the reference during several falling/low-reference
segments; this is a genuine shape/bias problem, not a reward-length artifact.

Training diagnostics are normalized by `n=512` steps before comparison. In the
recent saved windows, SAC had approximately `-4.1 reward/step`, POS RMSE
`11.7 kPa`, and NEG RMSE `3.4 kPa`; SAC+I-term had approximately `-9.7
reward/step`, POS RMSE `29.8 kPa`, and NEG RMSE `4.0 kPa`. Because carry changes
the initial state of every episode, these training-window values and the
single ATM-start evaluation are intentionally reported separately.

Matched plots are stored at:

```text
exp/260729_lib2_joint_full_sac_gamma099_history10_carry_h512_seed0/
  screen_eval_live_matched_checkpoint_2048step/trajectory.png
exp/260729_lib2_joint_full_sac_iterm_gamma099_history10_carry_h512_seed0/
  screen_eval_live_matched_checkpoint_2048step/trajectory.png
```
