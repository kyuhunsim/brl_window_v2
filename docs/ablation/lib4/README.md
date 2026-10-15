# lib4 강화학습 단계별 계획

작성일: 2026-08-23  
상태: A2–A5 1000-episode 학습·R3-ID 검증 완료; A0는 870에서 중단, A1 History 미생성. 정량 결과는 [results.md](results.md) 참조

진행 기록: [Phase 0–1 preflight](reference_preflight.md)에서 bounded carry와 legacy random reference를 audit한 뒤, displacement-targeted full-replay projection으로 train core 16개, unseen ID core 8개, nominal edge 4개를 분리했다. 세 artifact 모두 독립 oracle replay validator를 통과했다. edge는 수집 범위 밖의 모델 외삽 진단이며 학습 source가 아니다. OOD와 feasible-reference의 연구적 위치는 [research.md의 17–21절](../../research.md#17-feasible-reference와-ood를-novelty로-가져갈-수-있는가)에 정리했다.

이 문서는 lib3에서 사용한 여정—기존 구현 검증, feasible reference 작성, SAC, History SAC, physical feature, integral error, K=2—을 lib4에 재현하기 위한 실행 계획이다. 목표는 lib3의 모든 중간 실험을 복사하는 것이 아니라, 같은 인과관계를 최소한의 ablation으로 확인하고 lib4의 압력-변위 결합 특성까지 누락하지 않는 것이다.

## 1. lib3에서 가져올 핵심 교훈

lib3 기록은 [lib3 ablation README](../lib3_0807/README.md)와 [lib3 overnight note](../../../pneu_rl/docs/lib3_overnight_ablation_260805.md)에 정리되어 있다.

- 먼저 reference와 simulator/controller plumbing을 검증해야 한다. 학습 실패와 reference 불가능성을 섞으면 원인을 해석할 수 없다.
- Pure SAC는 현재 압력만으로 지연, valve hysteresis, chamber reserve를 충분히 복원하지 못했다.
- History SAC는 delayed action effect를 보완했지만 steady-state offset과 reserve 문제를 단독으로 해결하지 못했다.
- 물리 feature는 history를 대체하는 것이 아니라 pressure-rate와 reserve처럼 현재 측정 가능한 정보를 보강하는 역할이었다.
- integral error는 offset 보정에는 도움을 줄 수 있지만 wind-up과 seed 민감도를 반드시 확인해야 한다.
- K=2는 tracking을 개선할 수 있지만, chamber over-charge와 두 번째 action의 open-loop credit assignment를 숨길 수 있다.
- CAPS/MPOBS는 lib3에서 핵심 경로의 해결책으로 바로 채택하지 않는다. 먼저 physical/history/K=2의 효과를 분리한 뒤, 필요할 때만 별도 pair로 재검증한다.

## 2. lib4의 현재 상태와 검증 대상

현재 lib4에는 tuned simulator와 `PneuEnv4`가 있으며, 기존 모델과 진단 결과가 이미 있다.

| 항목 | 현재 자산 | 이번 계획에서의 처리 |
|---|---|---|
| simulator | `pneu_env/src/pneu_env/lib4/`, `sim4.py` | tuned parameter와 초기 상태를 고정하고 smoke/replay 검증 |
| environment | `pneu_env/src/pneu_env/env4.py` | observation/action 차원과 K=2 mapping을 먼저 검사 |
| 기존 모델 | `0624_lib4_Ours`, `0625_lib4_Ours`, `0626_lib4_Ours*` | 회귀 기준으로 matched validation |
| 기존 분석 | [model diagnostics](../../model_diagnostics.md) | 참고 결과로 사용하되 factorial ablation 결론으로 과대해석하지 않음 |
| baseline | `pneu_rl/scripts/baseline_lib4.py` | hold/constant/PID의 reference 추종 가능성 확인 |
| RL entry point | `pneu_rl/scripts/train4.py` | 재현 가능한 lib4 전용 실행 설정으로 고정할 후보 |

기존 `Ours` 모델은 PID+MPOBS+CAPS와 reference/reward 설정이 함께 들어간 성공 사례다. 따라서 이것은 “lib4가 동작하는가”를 보는 regression baseline이지, 새 단계별 ablation의 SAC baseline으로 사용하지 않는다.

## 3. 공통 protocol을 먼저 고정한다

아래 조건은 feasible reference가 통과한 뒤 모든 핵심 RL 실험에서 동일하게 유지한다.

- lib4 tuned parameter bundle, 초기 pressure/length, simulator library, Python wrapper 버전을 기록한다.
- control frequency `50 Hz`, simulator delay `0.1 s`, observation/prediction scaling, action physical range를 고정한다.
- reference profile, reference seed, reference horizon, carry/reset 규칙을 모든 모델의 config와 결과 디렉터리에 저장한다.
- 논문용 학습은 16-member R3 train core를 사용하고, ID test/edge-capacity/OOD stress bank를 별도 seed와 hash로 분리한다. one-member R2는 smoke/debug에만 사용한다.
- PID는 핵심 ablation에서는 `off`로 고정한다. PID-assisted 기존 모델은 별도 regression 비교로 남긴다.
- 외부 exploration noise는 `none`으로 둔다. SAC stochastic policy, alpha, log-std, action saturation만 기록한다.
- 학습 budget은 우선 lib3와 맞춰 `1000 episodes`, `horizon=512`, `carry=4 blocks`로 시작한다. lib4 기존 모델의 `2048` horizon 결과는 별도 historical reference로만 보관한다.
- 모든 모델은 seed 0으로 screening하고, 최종 후보만 seed 1/2를 추가한다.
- 검증은 최소 2048 physical simulator steps로 통일한다. K=2는 1024 policy decisions로 맞춰 physical duration을 동일하게 한다.

### lib3와 동일하게 고정한 것과 lib4에서만 달라지는 것

“같은 세팅”은 공통 학습 protocol을 뜻한다. `1000 episodes`, seed 0, `512` physical steps/episode, 4-block carry, history 10, reference lookahead 15, `start_epi=10`, `gamma=0.99`, `tau=0.005`, initial alpha `0.5`, hidden 256, buffer 500000, batch 128, learning rate `3e-4`, action mapping `u=0.8+0.2a`, physical warm-up `0.95`, warm-up 중 update 없음, warm-up 뒤 hard reset, observation normalization on, PID/MPOBS/CAPS/external exploration off를 lib3 8월 10일 실험과 맞춘다. 추가 rate/action/reserve penalty도 모두 0이다.

lib4의 task 정의상 아래 항목은 동일하게 만들 수 없으며 의도적으로 다르다.

| 항목 | lib3 | lib4 |
|---|---|---|
| 기본 관측 | pressure 4 | pressure 4 + displacement + displacement velocity |
| reference | actuator pressure 2 | actuator pressure 2 + displacement 1 |
| tracking reward | pressure POS/NEG 각 0.3 | pressure POS/NEG 각 0.3 + displacement 0.2 |
| 초기 상태/plant | lib3 초기 상태 | `0625_lib4_Ours` current-tuned pressure/length state |
| train reference | lib3 pressure bank | lib4 R3 3D feasible bank |

lib4 normalization은 pressure `100 kPa`, displacement `32 mm`, displacement velocity `20 mm/s`, pressure rate `50 kPa/s`, reserve `100 kPa`의 fixed unit scale을 사용한다. reward와 logger에는 원래 물리 단위를 그대로 사용한다.

평가에는 RMSE 하나만 쓰지 않는다.

- actuator pressure: RMSE, MAE, bias, tolerance-in-band, reference와의 방향 일치율
- displacement: RMSE, MAE, bias, peak/range 및 pressure-reference와의 일관성
- dynamics: pressure/displacement rate, optimal lag, amplitude attenuation
- safety/authority: chamber pressure range, chamber-actuator reserve, over-charge/under-pressure, simulator non-finite
- control: `mean |Δu|`, `max |Δu|`, `mean |Δ²u|`, valve conflict, saturation
- 학습: final-100 reward뿐 아니라 final-100 tracking metrics, alpha, log-std, critic loss

### Reference를 둘 다 쓰는 방식

두 reference를 같은 의미로 섞지 않는다. lib3처럼 **feasible bank는 training의 인과 비교를 위한 source**, legacy random은 **기존 구현 회귀와 OOD/stress 평가**를 위한 source로 분리한다.

| ID | profile / generator | range (현재 bank) | 사용처 | A0–A5 training |
|---|---|---|---|---|
| R0 | legacy `RandomRef` + static pressure-difference displacement clamp | initial `0→112/87 kPa` jump, displacement `7–21 mm` clamp | 기존 model regression, PID stress, OOD test | 사용하지 않음 |
| R1 | `lib4_feasible_oracle_4x512_seed61098` independent-smooth oracle bank | POS `101.28–110.23`, NEG `87.85–101.92`, contraction `0–6.69 mm` | small-signal/carry smoke, low-stroke regression | 사용하지 않음 |
| R2 | `lib4_feasible_pulse_high_disp_4x512_seed62001` topology pulse oracle trajectory | POS `97.41–130.36`, NEG `76.43–109.01`, contraction `0–16.21 mm` | high-stroke generator/smoke와 exact replay 검증 | 사용하지 않음 |
| R3 | `lib4_capacity_core_16x4x512_seed64001` displacement-targeted full-replay core | contraction `0–20.99 mm`, 16 members | A0–A5 공통 train source | **사용** |
| R3-ID | `lib4_capacity_core_idtest_8x4x512_seed65002` unseen core | contraction `0–20.61 mm`, 8 members | in-distribution validation | 사용하지 않음 |
| R4 | `lib4_capacity_edge_4x4x512_seed66001` state-gated nominal edge | contraction `0–27.03 mm`, 4 members | 수집 범위 밖 capacity stress | **사용 금지** (`training_eligible: false`) |

R0가 “영구 폐기”된 것은 아니다. 다만 independent random pressure와 static displacement fit은 corresponding valve trajectory를 보장하지 않고, 시작점 jump와 clamp plateau가 있다. R0에서 좋은 점수만 낸 모델을 feasible control로 결론내리지 않고, feasible bank에서 학습한 모델을 R0에서도 **별도** 평가한다. R1은 실제 low-stroke feasible regression으로 남기며, R2는 valve drive/release pulse가 만드는 중·고변위 구간을 포함한 high-stroke seed trajectory다. 두 oracle trajectory 모두 PID-certified가 아니라, current-tuned simulator에서 full six-valve prefix replay로 검증됐다.

R2를 1000 episode 반복하면 한 개의 40.96 s sequence를 외울 수 있으므로 R3로 대체했다. R3/R3-ID는 tuning CSV에서 확인된 contraction 최대 `21.47 mm` 안에 있고, R4만 그 밖을 탐색한다. nominal stroke는 `0–32 mm`지만 21.47 mm 밖은 실제 데이터 지지가 없으므로 R4 성능을 sim-to-real 가능성으로 해석하지 않는다.

처음부터 쉬운 궤적으로 시작하지 않고 R3 전체를 1000 episode 학습한다. R3는 16개 member, 4-block 연속 구간, 최대 20.99 mm를 포함하므로 데이터가 지지하는 범위에서는 이미 높은 난이도다. 더 어려워 보인다는 이유로 R4를 학습에 쓰는 것은 난이도 상승이 아니라 미검증 외삽을 학습 target에 섞는 것이므로 하지 않는다.

## 4. 단계별 실행 여정

### Phase 0 — 기존 lib4 및 simulator 회귀 검증

목표는 새 RL을 돌리기 전에 현재 lib4 자산이 서로 같은 조건을 보고 있는지 확인하는 것이다.

1. `baseline_lib4.py`에서 `hold`, `constant`, `pid`를 같은 random/sine/step reference에 실행한다.
2. `0624_lib4_Ours`, `0625_lib4_Ours`, `0626_lib4_Ours_no_vel_action_penalty_high`를 `quick_viz4.py`로 matched validation한다.
3. 각 결과에 simulator parameter source, initial state, reference hash, PID/CAPS/MPOBS 여부를 남긴다.
4. 다음을 fail-fast assertion으로 만든다: finite observation/reward, action channel mapping, pressure ordering, displacement sign/scale, reset/carry, simulator time step, model observation/action dimension.

완료 조건: 기존 성공 모델의 결과가 [model diagnostics](../../model_diagnostics.md)의 대략적인 범위에서 재현되고, baseline PID가 reference를 실제로 움직이며 추종한다. 이 조건이 안 되면 RL 단계로 넘어가지 않는다.

### Phase 1 — lib4 feasible reference 작성 및 검증

현재 lib4의 `RandomRef`/displacement reference를 곧바로 feasible하다고 가정하지 않는다. lib3에서 했던 것처럼 pressure-rate, chamber reserve, displacement rate를 기준으로 profile을 만든다.

권장 reference 구성은 다음과 같다.

- actuator pressure 2개와 displacement 1개를 함께 저장하는 fixed profile
- 50 Hz 샘플링, 4 blocks × 512 steps의 기본 bank
- abrupt parameter jump가 없는 rate-limited transition
- positive/negative chamber target을 actuator target보다 충분히 앞서도록 제한
- pressure difference와 displacement mapping이 simulator의 현재 튜닝 범위를 벗어나지 않도록 제한

reference validation에서 최소한 아래를 출력한다.

```text
reference hash / seed / duration
actuator pressure min/max and max |dP/dt|
displacement min/max and max |dD/dt|
chamber target min/max
minimum chamber-actuator reserve
rate-limit violations / non-finite values
oracle replay 또는 tuned PID의 RMSE와 방향 일치율
```

feasible 판정은 “reference 파일이 생성된다”가 아니라, bounded controller/oracle replay가 pressure와 displacement를 유한하고 안정적으로 추종하는지로 한다. oracle/PID가 실패하면 reference 범위·rate·초기 상태를 먼저 줄여 다시 만든다.

### Phase 1.5 — reference distribution과 OOD split 완성

2026-08-21 현재 1, 3, 4, 5번은 완료했다. 2번의 state-conditioned capacity는 R4에서 pressure-hysteresis와 displacement gate를 사용해 구현했지만, 실제-data/parameter robustness gate 전에는 평가 전용이다.

1. exact-oracle high-stroke train bank를 최소 16 members로 확장한다.
2. action proposal을 target으로 복사하지 않는 pressure–displacement capacity-conditioned bank를 별도로 만든다.
3. train, in-distribution unseen test, edge-of-capacity test, legacy/OOD stress를 서로 다른 seed와 file hash로 분리한다.
4. 각 member에 pressure/displacement range, rate, reserve, capacity utilization, initial state와 oracle replay residual을 저장한다.
5. coverage plot에서 `(P_act+, P_act-, displacement, velocity)`와 rate phase plane의 중복/빈 영역을 확인한다.

권장 reference ablation은 `legacy random → global rate-limited → state-conditioned capacity → exact oracle-output`의 네 단계다. exact oracle action은 어느 경우에도 policy input이나 teacher action으로 제공하지 않는다.

### Phase 2 — Pure SAC (K=1)

lib3의 첫 RL 기준선과 동일하게, 현재 observation과 현재/미래 reference만 사용한다. lib4에서는 displacement가 task에 포함되므로 현재 pressure 4개 + displacement 1개(+ 설정된 velocity 1개)와 reference만 사용한다. action은 한 번에 한 policy step만 출력한다.

목적은 “lib4의 직접형 SAC가 어디서 실패하는가”를 수치로 고정하는 것이다. 이 단계에서는 History, physical feature, integral, K=2, CAPS, MPOBS, PID를 동시에 넣지 않는다.

### Phase 3 — History SAC (K=1)

최근 pressure/displacement observation과 이전에 실제 적용된 action history를 추가한다. 기본 history 길이는 lib3와 같이 10 step으로 시작한다.

반드시 확인할 것:

- policy action과 simulator에 실제 적용된 physical action의 순서/scale이 일치하는가
- history가 reset/carry 경계에서 잘못 초기화되거나 중복 기록되지 않는가
- Pure SAC 대비 lag, 방향 일치율, pressure/displacement bias가 개선되는가

History SAC가 Pure SAC보다 좋아지지 않아도 바로 feature를 더하지 않는다. replay warm-up, action mapping, reference alignment, observation normalization을 먼저 검사한다.

### Phase 4 — History SAC + rate/reserve 분리

lib3의 `+ physical`을 lib4에 그대로 복사하지 않고, lib4에서 실제로 측정 가능하고 deploy 가능한 feature만 넣는다. seed-0 최소 screening에서는 다음을 각각 분리한다.

1. rate-only: chamber/actuator pressure rate 4개
2. reserve-only: `P_ch,pos-P_act,pos`, `P_act,neg-P_ch,neg` 2개
3. rate+reserve: 위 6개 결합

displacement velocity는 lib4의 기본 관측에 이미 포함되므로 rate-only에 중복 추가하지 않는다. 최근 raw action은 History SAC의 10-step action history에 이미 포함된다. simulator 내부의 hidden valve state, `z`, `mdot`를 policy observation에 넣지 않는다.

판정은 RMSE 감소만이 아니라, 같은 pressure error에서 action 방향이 맞아졌는지와 chamber reserve 고갈이 줄었는지를 본다.

### Phase 5 — Integral error (최소 screening 이후)

Phase 4의 best K=1 구조에 actuator pressure tracking integral error 2개를 추가한다. lib4의 chamber controller integral과 policy observation용 actuator integral을 섞지 않는다.

- `e_int[t+1] = clip(e_int[t] + e[t] * dt, -I_max, I_max)`
- reference/observation scaling 후 policy에 공급한다.
- hard reset, carry, K=2 action block에서 누적 규칙을 명시한다.
- saturation/deadband에서 anti-windup 또는 integration freeze를 적용한다.

적분항이 만드는 효과는 steady-state bias 감소이며, 변동 reference의 tracking 개선이라고 미리 가정하지 않는다. integral on/off pair에서 bias, saturation, action delta, recovery time을 비교한다.

### Phase 6 — K=2 action chunking

Phase 4의 best K=1 구조를 기준으로 `num_act=2`를 사용해 두 개의 연속 policy action을 한 번에 결정한다. lib4는 chamber controller 설정에 따라 policy action dimension이 달라질 수 있으므로, K=1/K=2에서 chamber control mode와 action expansion을 절대 바꾸지 않는다.

- K=1: 1 policy decision → 1 physical step
- K=2: 1 policy decision → 2 physical steps
- 검증 시간은 K=1 2048 decision, K=2 1024 decision으로 맞춘다.
- 두 번째 action이 첫 번째 action의 결과를 보지 못한다는 credit-assignment 한계를 기록한다.

반드시 chamber over-charge, reserve collapse, pressure overshoot를 별도 fail condition으로 둔다. tracking이 좋아져도 안전 범위를 넘으면 K=2 성공으로 판정하지 않는다.

### Phase 7 — 최종 조합 및 seed 확인

최소 seed-0 screening은 다음 여섯 개만 비교한다.

```text
Direct SAC
History SAC
History SAC + rate-only
History SAC + reserve-only
History SAC + rate+reserve
History SAC + rate+reserve + K=2
```

integral은 위 여섯 개에서 steady-state bias가 남는지 확인한 다음 별도 on/off pair로 추가한다. seed 0 screening에서 최종 후보를 1~2개로 줄인 후 seed 1/2를 추가한다. seed 간 평균·표준편차와 최악 seed를 보고 최종 모델을 고른다. 단일 seed의 최고 RMSE를 논문 최종 결론으로 삼지 않는다.

## 5. 최소 ablation matrix

| ID | 구성 | K | 목적 | seed |
|---:|---|---:|---|---:|
| A0 | Pure SAC | 1 | lib4 직접형 기준선 | 0 |
| A1 | History SAC | 1 | action/observation history 효과 | 0 |
| A2 | History SAC + rate-only | 1 | 변화 방향 정보의 효과 | 0 |
| A3 | History SAC + reserve-only | 1 | state-dependent control margin 효과 | 0 |
| A4 | History SAC + rate+reserve | 1 | 두 정보의 결합 및 offset 효과 | 0 |
| A5 | History SAC + rate+reserve | 2 | 동일 관측에서 action chunk 효과 | 0 |
| A6 | A0–A5 중 상위 1~2개 | 동일 | seed 분산 확인 | 1, 2 |

현재 seed-0 여섯 모델은 3 terminals에서 모두 돌려 한 번에 비교한다. 성능 악화를 보고 중간에 다른 run을 취소하지 않되, 추가 integral/CAPS/MPOBS와 seed 1/2는 on/off pair, shape/mapping, reward 로그를 검토한 뒤에만 진행한다.

## 6. 판정 기준

수치는 Phase 1 feasible reference 결과에 따라 최종 확정하되, screening의 임시 기준은 다음과 같다.

- pressure actuator POS/NEG RMSE 각 `5 kPa` 이하
- displacement RMSE `2 mm` 이하
- reference 방향 일치율 각 `0.8` 이상
- chamber pressure 및 reserve가 reference validation envelope 안에 있음
- non-finite, uncontrolled saturation, severe over-charge 없음
- 성능 개선이 flat output으로 얻어진 것이 아님: action variance와 output range를 함께 확인
- CAPS/추가 feature는 baseline 대비 tracking이 10% 이상 악화되면 채택하지 않음

최종 비교 표에는 반드시 `training config`, `reference hash`, `sim parameter source`, `PID`, `MPOBS`, `CAPS`, `integral`, `K`, seed, physical-step duration을 함께 기록한다.

## 7. 산출물과 실행 순서

### 문서/설정 산출물

```text
docs/ablation/lib4/
├── README.md                         # 본 계획 및 결과 누적 기록
├── reference_preflight.md            # Phase 1에서 추가
├── results.md                        # Phase 2 이후 표/해석 누적
└── run_6_screening_groups.sh         # 3 terminals × 2 sequential runs

pneu_rl/config/reference/
└── lib4_feasible_*.yaml/.npz         # 검증 통과 reference

pneu_rl/exp/lib4_*/                    # preflight, training, validation 결과
```

### 실행 순서

1. `baseline_lib4.py`로 기존 simulator/controller와 tuned parameter를 검증한다.
2. feasible reference generator/validator를 실행하고 hash와 preflight plot을 저장한다.
3. 최소 16-member train bank와 별도 unseen test bank를 만든 뒤 A0부터 A5까지 같은 reference와 protocol로 screening한다.
4. 각 모델을 최소 2048 physical steps matched validation한다.
5. A6 seed sweep 후에만 최종 모델과 CAPS/MPOBS optional ablation을 결정한다.

### Screening runner와 실행 명령

`train_lib4_screening.py`는 A0–A5를 위한 전용 entry point다. 기존 `train4.py`의 random-reference/PID/MPOBS/CAPS preset을 우회하고, 검증된 3D bank와 current-tuned simulator만 사용한다. 각 실행 시 model directory에 reference/profile hash, simulator source, environment feature/action-chunk 설정이 들어간 `cfg.yaml`을 저장한다.

먼저 모든 variant는 아래 dry run을 통과해야 한다. 이 검사는 observation/action shape, finite reward, K=2의 2-physical-step expansion, 첫 carry boundary의 state preservation을 실제 simulator로 확인한다.

```bash
source activate_env.sh
PY=/home/rise/miniconda3/envs/env_1/bin/python
SMOKE_REF=pneu_rl/config/reference/lib4_capacity_core_16x4x512_seed64001.yaml

for V in sac history_sac history_rate history_reserve \
         history_rate_reserve history_rate_reserve_k2; do
  $PY pneu_rl/scripts/train_lib4_screening.py \
    --variant "$V" --reference-profile "$SMOKE_REF" --dry-run
done
```

seed 0 screening은 R3로 각 모델을 1000 episode 학습한다. 세 terminal에서 아래 세 명령을 하나씩 실행하면 terminal마다 두 모델이 순차 실행된다. K=1/K=2 모두 `512` physical step per episode와 4-block carry를 유지하므로, `K=2`의 policy horizon만 `256`으로 자동 축소된다.

```bash
bash docs/ablation/lib4/run_6_screening_groups.sh 1
bash docs/ablation/lib4/run_6_screening_groups.sh 2
bash docs/ablation/lib4/run_6_screening_groups.sh 3
```

결과는 [results.md](results.md)에 즉시 기록한다. `train4.py`의 기존 `Ours` 성공 결과는 계속 regression reference일 뿐, A0의 대체 결과가 아니다.

## 7.1 후속 계획 실행 코드

2026-08-24 후속 계획은 다음 세 파일을 single source of truth로 사용한다.

- experiment manifest: [`lib4_followup_plan.yaml`](../../../pneu_rl/config/experiments/lib4_followup_plan.yaml)
- gated runner: [`run_lib4_followup_plan.py`](../../../pneu_rl/scripts/run_lib4_followup_plan.py)
- fixed evaluation protocol: [`lib4_p0_protocol.yaml`](../../../pneu_rl/config/evaluation/lib4_p0_protocol.yaml)

runner는 기본적으로 명령만 출력하며 `--execute`를 붙여야 실제 학습/평가를 실행한다. 기존 model directory가 있으면 덮어쓰지 않고 중단한다.

```bash
# P1 네 작업과 평가 명령 확인
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/run_lib4_followup_plan.py \
  --phase p1_causal_recovery

# 예: terminal별 작업 분배
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/run_lib4_followup_plan.py --stage train --execute \
  --job 260824_lib4_r3_direct_clean_seed0

/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/run_lib4_followup_plan.py --stage train --execute \
  --job 260824_lib4_r3_history_seed0

/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/run_lib4_followup_plan.py --stage train --execute \
  --job 260824_lib4_r3_history_rate_reserve_seed1 \
  --job 260824_lib4_r3_history_rate_reserve_seed2

# 네 checkpoint가 모두 생긴 뒤 같은 R3-ID/protocol로 일괄 평가
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/run_lib4_followup_plan.py \
  --phase p1_causal_recovery --stage evaluate --execute
```

P2/P3/P4 job은 manifest에서 `enabled: false`다. 이전 gate를 확인한 뒤 필요한 job 하나만 `--include-gated --job NAME`으로 선택한다. `--include-gated`만 사용해 gated phase 전체를 무조건 실행하지 않는다.

추가 variant의 의미는 다음과 같다.

| variant | policy action | 목적 |
|---|---:|---|
| `history_rate_reserve_integral` | 6 | plateau bias가 확인된 경우의 integral on/off |
| `history_rate_reserve_smooth` | 6 | action-delta reward 또는 작은 temporal CAPS 단독 진단 |
| `history_rate_reserve_k2_hold` | 6 | 한 action을 2 physical step 반복; 25 Hz decision-rate 효과 분리 |
| `history_rate_reserve_k2_consistency` | 12 | 두 자유 sub-action 사이 actor consistency penalty |

### Structured-history follow-up (HRR / PIDE / GPIDE)

이 비교의 1차 기준은 lib3이다. lib4 script는 lib3에서 feature 의미와 reward
조건을 먼저 확인한 뒤 displacement/velocity가 포함된 시스템으로 이식할 때
사용한다.

세 구조화 history 비교는 동일한 lib4 screening 조건에서 다음 스크립트로
준비한다.

```bash
# simulator/reference/observation shape만 검사
DRY_RUN=1 bash pneu_rl/scripts/run_lib4_structured_history.sh

# 실제 학습 (기본 seed=0, 1000 episode)
bash pneu_rl/scripts/run_lib4_structured_history.sh
```

`hrr`는 기존 `history_rate_reserve`와 완전히 같은 6차원 feature다. `pide`는
현재 actuator tracking error의 P/I/D 6차원 요약이고, `gpide`는 base observation
difference 6차원과 현재/차분 action 12차원을 합친 18차원 고정 요약이다. 둘은
hidden simulator state를 읽지 않으며 NeurIPS 2023의 difference/sum inductive
bias를 검증하는 첫 baseline이다. 논문의 learned PIDE/GPIDE encoder와 동일하다고
표기하지 말고, 필요하면 후속 recurrent implementation으로 확장한다.

평가 CSV에는 이제 deterministic twin-Q `value` 열이 추가된다. 동일 trace에서
TD(0)와 TD(0.95) return 차이를 계산하려면:

```bash
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/analyze_lambda_discrepancy.py \
  pneu_rl/exp/<evaluation-root>/<model-name> \
  --output pneu_rl/exp/<evaluation-root>/<model-name>_lambda.json
```

Lambda discrepancy는 controller score가 아니라 value-relevant Markovness
진단이다. 서로 다른 policy의 validation 결과를 바로 순위화하지 말고, 가능하면
같은 frozen policy/trajectory에서 representation만 바꾼 trace를 사용한다.

K2 환경은 `action_chunk_mode=free/hold`를 cfg에 저장한다. 모든 K2 training diagnostics는 이제 마지막 sub-action만 보지 않고 두 physical sub-step의 `info`를 펼쳐 계산한다. evaluator CSV도 `policy_step`, `sub_action_index`를 저장해 within-chunk와 chunk-boundary `|Δu|`를 분리한다.

P0 evaluator는 reference derivative `1e-9` 기준을 폐기하고 protocol에 고정된 physical rate deadband와 0.1 s causal lag를 사용한다. transition RMSE/direction, plateau bias/oscillation/rate, settling, reserve violation, `|Δu|/|Δ²u|`와 provisional gate를 member별로 기록하며 protocol SHA256를 결과에 남긴다. 현재 protocol threshold는 real noise floor/step delay 측정 전 provisional 값이므로, real 측정으로 교체할 때는 새 protocol version과 hash를 만든다.

회귀 검사는 다음 명령으로 실행한다.

```bash
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/test_lib4_followup_plan.py
```

## 8. 기록 템플릿

각 실험 결과를 추가할 때 아래 필드를 빠짐없이 남긴다.

```text
experiment_id:
model_name:
parent_stage:
seed:
reference_profile:
reference_hash:
simulator_parameter_source:
initial_state:
pid / mpobs / caps / physical / integral / K:
observation_dim / action_dim:
training_episodes / horizon / carry:
validation_physical_steps:
pressure_rmse_pos / pressure_rmse_neg:
displacement_rmse:
direction_agreement_pos / direction_agreement_neg:
reserve_min / chamber_peak:
mean_delta_u / max_delta_u / mean_delta2_u:
reward_final100:
decision: pass | fail | investigate
failure_mode:
```
