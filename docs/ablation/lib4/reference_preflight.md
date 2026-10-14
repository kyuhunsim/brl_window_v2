# lib4 Phase 0–1 preflight 기록

작성일: 2026-08-21  
상태: Phase 1.5 multi-member train/ID/edge split 및 독립 replay 검증 완료

이 문서는 [lib4 RL 계획](README.md)의 학습 전 gate를 실제로 실행한 기록이다. 아직 RL training 결과는 없으며, 이 단계의 목적은 reference·carry·simulator configuration을 검증하는 것이다. R1/R2는 one-trajectory smoke이고, 실제 screening source는 새 R3 16-member core다.

## 결론

- legacy lib4 random pressure/displacement reference는 **버리는 대신 역할을 분리**한다. 기존 PID baseline의 40초 결과는 POS/NEG/displacement RMSE가 각각 `10.14 kPa`, `8.69 kPa`, `11.54 mm`였고, reference 자체도 `t=0`에서 `0→112/87 kPa` jump 및 static displacement clamp plateau를 만든다. 그러므로 A0–A5의 training target은 아니며, 기존 model regression/OOD stress 평가로 유지한다.
- lib4의 `PneuEnv4`에 bounded carry와 hard reset을 추가했다. 4개의 512-step block 사이에서는 simulator/reference/history/action을 유지하고, 네 번째 경계에서만 hard reset한다.
- tuned lib4 simulator에서 실제 6-valve command를 replay해 pressure 2개와 displacement를 동시에 얻는 `4 × 512` oracle-feasible bank 두 종류를 만들었다. R1은 independent-smooth small-signal bank, R2는 valve topology에 맞춘 drive/release pulse high-stroke bank다.
- R2는 contraction `16.21 mm`, actuator POS `130.36 kPa`까지 커버한다. 이는 기존 R1의 낮은 stroke가 plant 한계가 아니라 independent valve proposal의 coverage 부족이었음을 보인다. 두 bank 모두 **oracle-feasible**이지 PID-certified reference는 아니다.
- tuning CSV에서 역산한 contraction 관측 범위는 약 `0.31–21.47 mm`, simulator의 의도한 hard stroke는 `0–32 mm`다. 이 둘을 같다고 보지 않고, `≤21.47 mm` core와 `>21.47 mm` nominal edge를 분리했다.
- R3 train은 16 members, R3-ID는 다른 seed의 8 members이며 각각 contraction `20.99 mm`, `20.61 mm`까지 커버한다. 두 bank 모두 독립 replay `all_checks_pass: true`다.
- R4 edge는 chamber+ pressure hysteresis와 displacement gate를 쓰는 offline state-conditioned oracle로 `27.03 mm`까지 도달했다. 독립 replay는 통과했지만 실제 수집 범위 밖이므로 profile에 `training_eligible: false`를 기록했다.

2026-08-12 current-worktree 재검증에서는 같은 NPZ (SHA256 `a95166b89b20251a72ac1c6b4a61c5f4a865743d628568781b645bb748f7453c`)를 독립 simulator instance에서 두 번 replay했고, 두 번 모두 `all_checks_pass: true`, 최대 절대 오차 `0.05601484`를 얻었다. 이 재검증은 artifact shape `[1, 2049, 3]`와 state band, rate, reserve, 512-step boundary도 함께 확인한다.

## 새 코드와 역할

| 파일 | 역할 |
|---|---|
| [env4.py](../../../pneu_env/src/pneu_env/env4.py) | bounded carry, explicit hard reset, reset 시 hidden low action 미주입 |
| [trajectory_ref_nd.py](../../../pneu_ref/src/pneu_ref/trajectory_ref_nd.py) | lib4의 `[act_pos, act_neg, displacement]` bank를 carry-aware하게 제공 |
| [preflight_lib4.py](../../../pneu_rl/scripts/preflight_lib4.py) | phase 0 simulator/action mapping/carry preflight |
| [generate_lib4_feasible_reference.py](../../../pneu_rl/scripts/generate_lib4_feasible_reference.py) | independent-smooth 또는 coordinated topology-pulse six-valve replay로 3D feasible bank 생성 |
| [validate_lib4_reference.py](../../../pneu_rl/scripts/validate_lib4_reference.py) | state/rate/reserve/boundary/oracle replay 독립 검증 |
| [visualize_lib4_reference.py](../../../pneu_rl/scripts/visualize_lib4_reference.py) | pressure/contraction/reserve/oracle-action/carry boundary plot과 summary 생성 |
| [audit_lib4_legacy_random_reference.py](../../../pneu_rl/scripts/audit_lib4_legacy_random_reference.py) | legacy random의 jump, clamp, rate를 oracle proof 없이 계측/시각화 |
| [generate_lib4_capacity_frontier_reference.py](../../../pneu_rl/scripts/generate_lib4_capacity_frontier_reference.py) | contraction 목표별 sampled projection 및 state-gated nominal edge 생성 |
| [R3 train profile](../../../pneu_rl/config/reference/lib4_capacity_core_16x4x512_seed64001.yaml) | 현재 A0–A5 공통 학습 reference |
| [R3-ID profile](../../../pneu_rl/config/reference/lib4_capacity_core_idtest_8x4x512_seed65002.yaml) | unseen in-distribution 검증 reference |
| [R4 edge profile](../../../pneu_rl/config/reference/lib4_capacity_edge_4x4x512_seed66001.yaml) | nominal capacity stress 전용, 학습 금지 |

## Phase 0 결과 — 기존 환경과 carry

실행:

```bash
source activate_env.sh
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/preflight_lib4.py \
  --model-cfg 0625_lib4_Ours \
  --state-source current_tuned \
  --reference-profile pneu_rl/config/reference/lib4_feasible_oracle_4x512_seed61098.yaml \
  --carry-blocks 4 --steps-per-block 512 \
  --output pneu_rl/exp/lib4_phase01_preflight/lib4_feasible_oracle_carry.yaml
```

결과: `all_checks_pass: true`.

- 초기 reset은 simulator를 advance하지 않는다.
- normalized policy action과 applied physical valve action의 scale/mapping이 일치한다.
- 첫 세 episode boundary에서 simulator time, environment time, observation history, reference buffer, 마지막 policy action이 변하지 않는다.
- 네 번째 boundary에서 hard reset이 발생하고, 그 다음 첫 physical step의 시간은 정확히 `1/50 s`부터 다시 시작한다.

current-worktree 재실행에서도 위 carry preflight가 다시 `all_checks_pass: true`였다. 즉 첫 세 경계에서 simulator/environment time, observation history, reference buffer, 마지막 action이 보존되고, 네 번째 경계에서만 hard reset한다.

세부 로그: [carry preflight YAML](../../../pneu_rl/exp/lib4_phase01_preflight/lib4_feasible_oracle_carry.yaml).

기존 random ref baseline 실행:

```bash
source activate_env.sh
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/baseline_lib4.py \
  --controllers pid --model-cfg 0625_lib4_Ours \
  --ref random --max-time 40 --chamber-mode fixed \
  --save-name phase0_legacy_random_pid_40s
```

이 결과는 reference가 plant 출력 범위와 displacement mapping을 자동으로 만족한다는 보장이 없음을 보여 준다. 여기서 PID 실패만으로 random ref 전체가 수학적으로 infeasible하다고 증명한 것은 아니지만, 새 training baseline에 사용하기에는 근거가 부족하다.

legacy reference 자체의 40.96 s audit plot은 [legacy random reference PNG](../../../pneu_rl/exp/lib4_phase0_legacy_random_audit/legacy_random_reference.png), 수치 summary는 [legacy random reference YAML](../../../pneu_rl/exp/lib4_phase0_legacy_random_audit/legacy_random_reference.yaml)에 저장했다.

- 첫 `20 ms` sample은 pressure target이 `0→112/87 kPa`로 jump한다. 따라서 sampled maximum rate는 POS `5650 kPa/s`, NEG `4349 kPa/s`이며 startup을 포함한 smooth-task training source가 될 수 없다.
- static `ΔP→displacement` fit은 raw target의 `37.97%`를 upper clamp `21 mm`에 고정한다. 이는 plant state를 replay해 나온 motion이 아니므로 해당 plateau를 물리적으로 feasible하다고 해석할 수 없다.
- 따라서 R0는 **사용한다**: 이전 model/PID regression, 그리고 R2에서 학습한 policy의 OOD stress 테스트. 단, R0 결과를 R2의 feasible tracking score와 합산하거나 A0–A5 training에 넣지는 않는다.

## Phase 1 결과 — displacement 포함 oracle-feasible bank와 시각화

생성 기준:

- simulator: `260624_11_21_35_tuner4_output48plus49_linit34807_leak_tuned` parameter와 해당 initial-state CSV
- sample rate: `50 Hz`
- bank: `1 trajectory × 2049 samples = 4 × 512` continuous physical steps
- RL reference: actuator POS pressure, actuator NEG pressure, total contraction `mm`
- stored raw action: audit/oracle replay 전용이며 policy observation에는 넣지 않음
- hard reset 전까지 profile을 이어서 사용하고, `repeat_after_end: false`

### R1 — independent-smooth low-stroke bank (regression/smoke)

R1의 canonical artifact는 [profile](../../../pneu_rl/config/reference/lib4_feasible_oracle_4x512_seed61098.yaml), [plot](../../../pneu_rl/exp/lib4_phase1_reference_preflight/lib4_feasible_oracle_4x512_seed61098.png), [visualization summary](../../../pneu_rl/exp/lib4_phase1_reference_preflight/lib4_feasible_oracle_4x512_seed61098_visualization.yaml)다.

현재 artifact plot을 다시 계측한 range는 POS `101.28–110.23 kPa`, NEG `87.85–101.92 kPa`, contraction `0–6.69 mm`다. 이전 문서에 있던 `0–9.74 mm` 값은 이 canonical NPZ와 일치하지 않는 과거 run 기록이므로 더 이상 R1의 근거로 쓰지 않는다. R1은 carry 및 low-stroke regression에는 유용하지만 primary training task로는 좁다.

### R2 — topology-pulse high-stroke bank (generator smoke)

R2는 단순히 모든 valve를 독립적으로 움직이지 않는다. `drive=[0,1,1,0,1,0]`과 `release=[1,0,0,1,0,1]` raw posture를 `1.5–3.0 s` minimum-jerk transition과 `1–2 s` hold로 교대한다. 이 command는 **offline candidate 생성에만** 쓰며 RL policy에는 노출하지 않는다. 각 block 후보는 처음부터 full-prefix replay한 뒤 state band, rate, chamber reserve, terminal state, displacement peak 조건을 통과해야 채택된다.

R2의 canonical artifact는 [profile](../../../pneu_rl/config/reference/lib4_feasible_pulse_high_disp_4x512_seed62001.yaml), [plot](../../../pneu_rl/exp/lib4_phase1_reference_preflight/lib4_feasible_pulse_high_disp_4x512_seed62001.png), [visualization summary](../../../pneu_rl/exp/lib4_phase1_reference_preflight/lib4_feasible_pulse_high_disp_4x512_seed62001_visualization.yaml), [validator log](../../../pneu_rl/exp/lib4_phase1_reference_preflight/lib4_feasible_pulse_high_disp_4x512_seed62001_validation.yaml)다.

| 항목 | R2 결과 |
|---|---:|
| actuator POS range | `97.41–130.36 kPa` |
| actuator NEG range | `76.43–109.01 kPa` |
| contraction range | `0.00–16.21 mm` |
| minimum POS/NEG reserve | `7.96 / 60.46 kPa` |
| max POS/NEG/contraction rate (startup 제외) | `44.54 / 29.33 kPa/s / 18.88 mm/s` |
| max ref step at 512 boundary | `0.1203 / 0.0158 kPa`, `0.0453 mm` |
| float32 oracle replay max abs error | `0.02054` |
| all checks | pass |

R2 NPZ SHA256은 `474e0c475c07c75564bad576c0322e21569771bc280594e5f9a376a0e10a62db`다. 이 bank는 generator regression/smoke로 남고 A0–A5 장기 학습에는 쓰지 않는다.

### R3 — data-supported core train/ID split

R3는 원하는 contraction peak를 먼저 배정하고, smooth conflict-free valve program을 full tuned-simulator replay한 결과 중 state band, rate, reserve, target projection error와 coverage를 통과한 출력만 채택한다. RL reference는 여전히 `[P_act+, P_act-, contraction]`뿐이며 oracle action은 float64 replay audit 데이터다.

| split | members × samples | contraction max | NPZ SHA256 | validator |
|---|---:|---:|---|---|
| [train](../../../pneu_rl/config/reference/lib4_capacity_core_16x4x512_seed64001.yaml) | `16 × 2049` | `20.99 mm` | `8f1e5de783c7ca77549ed7c75b7ba491ad7ed0fe7f0c4fdde7965a086a76560b` | [pass](../../../pneu_rl/exp/lib4_capacity_core_16x4x512_seed64001/validation.yaml) |
| [unseen ID](../../../pneu_rl/config/reference/lib4_capacity_core_idtest_8x4x512_seed65002.yaml) | `8 × 2049` | `20.61 mm` | `0c5b1204c0f94a72588b376f1fa1736ef59848f1b96f7bc8f58a0fc440ed805f` | [pass](../../../pneu_rl/exp/lib4_capacity_core_idtest_8x4x512_seed65002/validation.yaml) |

train bank 전체 모양은 [bank overview](../../../pneu_rl/exp/lib4_capacity_core_16x4x512_seed64001/bank_overview.png), 최대변위 member의 압력·변위·속도·reserve·oracle command는 [trajectory plot](../../../pneu_rl/exp/lib4_capacity_core_16x4x512_seed64001/trajectory_10.png)에서 확인한다. pressure-difference와 contraction은 임의로 독립 샘플링되지 않고 plant replay가 만드는 비선형 manifold를 따른다.

### R4 — state-conditioned nominal edge probe

R4 생성 시 chamber+가 hysteresis band를 넘으면 공급 chamber command만 charge/vent 전환하고, 목표 변위 근처에서 네 actuator valve를 별도로 감쇠한다. 이 offline feedback은 지속 가압으로 chamber upper bound를 넘는 문제를 해결하지만 policy observation이나 teacher action이 아니다.

- artifact: [profile](../../../pneu_rl/config/reference/lib4_capacity_edge_4x4x512_seed66001.yaml), [overview](../../../pneu_rl/exp/lib4_capacity_edge_4x4x512_seed66001/bank_overview.png), [27 mm trajectory](../../../pneu_rl/exp/lib4_capacity_edge_4x4x512_seed66001/trajectory_3.png), [validator](../../../pneu_rl/exp/lib4_capacity_edge_4x4x512_seed66001/validation.yaml)
- shape/range: `4 × 2049 × 3`, contraction `0–27.03 mm`
- SHA256: `933ab3c415330c8c3cdffc4fb81995dcf3876f4a20454f3da8f7e7d070a7aaf5`
- 해석: nominal simulator에서 rate/reserve/state-band와 exact replay를 통과했다는 뜻이다. `21.47–27.03 mm`는 tuning data 외삽이므로 실제 feasibility나 학습 안전성을 증명하지 않는다.

### R1의 historical validator 결과

| 항목 | 결과 |
|---|---:|
| float32 oracle replay max abs error | `0.05601` (current-worktree replay) |
| all checks | pass |

세부 로그: [reference validation YAML](../../../pneu_rl/exp/lib4_phase1_reference_preflight/lib4_feasible_oracle_4x512_seed61098_validation.yaml).

중요한 발견은 `7–21 mm` static clamp도, “0–16 mm까지만 가능”이라는 해석도 모두 부족했다는 점이다. full replay core는 수집 범위 안에서 약 `21 mm`, state-gated nominal probe는 `27 mm`까지 갔다. 따라서 학습은 R3 core에서 시작하고, R4는 robust parameter sweep과 실제 저속 확인이 생길 때까지 stress test로만 둔다.

## 재생성/검증 명령

아래 명령은 현재 통과한 첫 bank의 생성 조건을 그대로 남긴 것이다.

```bash
source activate_env.sh
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/generate_lib4_feasible_reference.py \
  --model-cfg 0625_lib4_Ours --state-source current_tuned \
  --seed 61098 --bank-size 1 --blocks 4 --horizon-steps 512 \
  --candidates-per-block 16 \
  --ch-pos-raw 0.5,1.0 --ch-neg-raw 0.8,1.0 \
  --ch-pos-band 100,200 --ch-neg-band 1,105 \
  --act-pos-band 90,160 --act-neg-band 20,110 --disp-band 0,32 \
  --ch-pos-terminal 100,195 --ch-neg-terminal 1,100 \
  --act-pos-terminal 90,155 --act-neg-terminal 20,108 --disp-terminal 0,31 \
  --min-reference-span 1,1,0.1 \
  --npz-output pneu_rl/config/reference/lib4_feasible_oracle_4x512_seed61098.npz \
  --profile-output pneu_rl/config/reference/lib4_feasible_oracle_4x512_seed61098.yaml

/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/validate_lib4_reference.py \
  --reference-profile pneu_rl/config/reference/lib4_feasible_oracle_4x512_seed61098.yaml \
  --output pneu_rl/exp/lib4_phase1_reference_preflight/lib4_feasible_oracle_4x512_seed61098_validation.yaml
```

R2 high-stroke bank 생성/검증/시각화:

```bash
source activate_env.sh
PY=/home/rise/miniconda3/envs/env_1/bin/python

$PY pneu_rl/scripts/generate_lib4_feasible_reference.py \
  --model-cfg 0625_lib4_Ours --state-source current_tuned \
  --seed 62001 --bank-size 1 --blocks 4 --horizon-steps 512 \
  --candidates-per-block 12 --proposal-mode topology_pulse \
  --initial-hold-sec 0.8 --transition-sec 1.5,3.0 --hold-sec 1.0,2.0 \
  --min-hold-sec 0.8 --min-reference-span 1,1,0.5 --min-disp-peak 8 \
  --npz-output pneu_rl/config/reference/lib4_feasible_pulse_high_disp_4x512_seed62001.npz \
  --profile-output pneu_rl/config/reference/lib4_feasible_pulse_high_disp_4x512_seed62001.yaml

$PY pneu_rl/scripts/validate_lib4_reference.py \
  --reference-profile pneu_rl/config/reference/lib4_feasible_pulse_high_disp_4x512_seed62001.yaml \
  --output pneu_rl/exp/lib4_phase1_reference_preflight/lib4_feasible_pulse_high_disp_4x512_seed62001_validation.yaml

$PY pneu_rl/scripts/visualize_lib4_reference.py \
  --reference-profile pneu_rl/config/reference/lib4_feasible_pulse_high_disp_4x512_seed62001.yaml \
  --output pneu_rl/exp/lib4_phase1_reference_preflight/lib4_feasible_pulse_high_disp_4x512_seed62001.png
```

legacy R0 visualization은 아래 명령으로 매번 재생성할 수 있다.

```bash
$PY pneu_rl/scripts/audit_lib4_legacy_random_reference.py \
  --model-cfg 0625_lib4_Ours --duration-sec 40.96 \
  --output pneu_rl/exp/lib4_phase0_legacy_random_audit/legacy_random_reference.png
```

## 다음 gate

Phase 2–6용 [`train_lib4_screening.py`](../../../pneu_rl/scripts/train_lib4_screening.py)를 R3 train profile로 다시 dry-run했다. no-PID K=1 Pure SAC, History SAC, physical feature, integral, K=2의 여섯 조합이 모두 shape/finite/carry 검사를 통과했고, K=2는 두 policy action을 순서대로 적용하는 12차원 action과 256-decision/512-physical-step horizon을 확인했다.

이제 A0부터 동일한 `512 physical steps × 4 carry` protocol로 seed 0 장기 screening을 실행할 수 있다. 아직 어떤 A0–A5도 학습시키지 않았다. 결과와 실행 상태는 [results.md](results.md)에 기록하며, 기존 `train4.py`의 interactive random-reference path는 이 matrix에 사용하지 않는다.
