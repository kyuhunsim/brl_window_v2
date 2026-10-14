# lib4 최소 ablation 결과

작성일: 2026-08-24  
상태: 4개 1000-episode 완료, Direct 870에서 중단, History 미생성; 완료 checkpoint의 R3-ID matched validation 완료

## Protocol 고정값

- train reference: `lib4_capacity_core_16x4x512_seed64001.yaml` (R3; `16×2049×3`, contraction `0–20.99 mm`; SHA256 `8f1e5de783c7ca77549ed7c75b7ba491ad7ed0fe7f0c4fdde7965a086a76560b`)
- unseen ID reference: `lib4_capacity_core_idtest_8x4x512_seed65002.yaml` (`8×2049×3`, contraction `0–20.61 mm`; SHA256 `0c5b1204c0f94a72588b376f1fa1736ef59848f1b96f7bc8f58a0fc440ed805f`)
- nominal edge reference: `lib4_capacity_edge_4x4x512_seed66001.yaml` (`4×2049×3`, contraction `0–27.03 mm`; SHA256 `933ab3c415330c8c3cdffc4fb81995dcf3876f4a20454f3da8f7e7d070a7aaf5`, training 금지)
- 세 artifact 모두 독립 oracle replay validator `all_checks_pass: true`; R1/R2는 regression/smoke, R0는 OOD stress에 사용
- low-stroke R1 and legacy random R0는 학습 source가 아니라 각각 regression/smoke와 OOD stress 평가에 사용
- simulator: `0625_lib4_Ours`의 `current_tuned` parameter source
- controller: PID off, CAPS off, MPOBS/predictor off, external exploration noise off
- episode: 512 physical steps; 4 episode blocks 뒤에 hard reset
- training: 1000 episodes, seed 0, history 10, lookahead 15, observation normalization on
- optimizer/SAC: learning rate `3e-4`, gamma `0.99`, tau `0.005`, initial alpha `0.5`, hidden 256, batch 128, buffer 500000, start episode 10
- reward: actuator POS/NEG `0.3/0.3`, displacement `0.2`; 그 외 future/pred/rate/action/reserve penalty `0`
- warm-up/action: physical command `0.95`, warm-up update off, post-warm-up hard reset, action range `0.8–1.0`
- evaluation: candidate별 최소 2048 physical steps, K=1/K=2의 physical duration 동일
- per-episode diagnostics: pressure RMSE/bias/reserve/action과 함께 displacement RMSE/bias/range/direction agreement 및 POS/NEG/displacement reward 성분을 기록

## Entry-point smoke 결과

2026-08-23에 합의한 여섯 variant를 R3 16-member profile과 actual lib4 simulator로 다시 reset/step/carry dry-run 했다. 모두 finite state/reward, unit-aware observation scale, normalized SAC policy construction, action/observation shape 검사를 통과했다.

| ID | variant | observation dim | action dim | physical steps / policy step | 결과 |
|---:|---|---:|---:|---:|---|
| A0 | `sac` | 12 | 6 | 1 | ready |
| A1 | `history_sac` | 204 | 6 | 1 | ready |
| A2 | `history_rate` | 208 | 6 | 1 | ready |
| A3 | `history_reserve` | 206 | 6 | 1 | ready |
| A4 | `history_rate_reserve` | 210 | 6 | 1 | ready |
| A5 | `history_rate_reserve_k2` | 210 | 12 | 2 | ready |

`K=2`의 12차원 action은 two consecutive six-valve commands다. runner가 이를 two simulator steps으로 차례로 적용하며, episode horizon은 256 policy decisions로 자동 변환한다.

평가도 같은 원칙으로 K=1은 2048 decisions, K=2는 1024 decisions를 사용해 둘 다 2048 actual simulator steps로 맞춘다. K=2 결과를 2048 decisions로 돌려 K=1의 두 배 시간과 비교하지 않는다.

## 학습 완료성 audit

사용자가 6개 학습 완료로 인지했지만 model artifact를 직접 확인한 결과는 다음과 같다.

| ID | model | 저장된 episode | 상태 |
|---:|---|---:|---|
| A0 | `260823_lib4_r3_direct_seed0` | 870 | **중단됨**; 1000-episode 비교에 사용 불가 |
| A1 | `260823_lib4_r3_history_seed0` | 0 | **model directory 없음** |
| A2 | `260823_lib4_r3_history_rate_seed0` | 1000 | 완료 |
| A3 | `260823_lib4_r3_history_reserve_seed0` | 1000 | 완료 |
| A4 | `260823_lib4_r3_history_rate_reserve_seed0` | 1000 | 완료 |
| A5 | `260823_lib4_r3_history_rate_reserve_k2_seed0` | 1000 | 완료 |

A0 terminal이 870에서 끝난 뒤 같은 group의 A1이 시작되지 않은 형태다. 저장된 stdout/stderr log가 없어 종료 원인은 artifact만으로 확정할 수 없다. A0 checkpoint는 참고용으로만 평가했고 A0/A1을 포함하는 완전한 History 효과는 아직 결론낼 수 없다.

[학습 curve](../../../pneu_rl/exp/lib4_0824_training_analysis/training_comparison.png) · [final-100 CSV](../../../pneu_rl/exp/lib4_0824_training_analysis/training_final100.csv)

## R3-ID unseen matched validation

2026-08-24에 R3-ID 8개 member를 모델별로 각각 `2048 actual simulator steps` 평가했다. 모든 모델에서 같은 eight-member coverage order를 사용했고 deterministic policy로 평가했다. K=2는 1024 policy decisions에서 생성된 두 sub-action을 모두 기록해 K=1과 똑같이 2048 actual step으로 계산했다.

여기서 `member`는 모델이나 seed가 아니라 trajectory bank 안의 개별 unseen reference다. R3-ID 한 member는 `초기점 1 + 4 blocks × 512 steps = 2049 samples`, 약 40.96초짜리 `[P_act+, P_act-, displacement]` 궤적 하나다. CSV에는 policy raw action `0–1`과 simulator physical command `0.8–1.0`을 모두 저장하지만, 6-control 그림과 아래 `|Δu|` 지표는 lib3와 동일하게 physical command를 사용한다.

| ID | model | POS RMSE | NEG RMSE | disp RMSE | worst reserve POS/NEG | mean `|Δu|` | max `|Δu|` | 판정 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| A0* | Direct, 870 epi | 4.01 | 3.72 | 2.95 | 1.85 / 46.69 | .0114 | .1891 | incomplete; tracking 실패 |
| A1 | History | — | — | — | — | — | — | missing |
| A2 | History + rate | **.73** | 1.22 | .88 | **-.30** / 39.15 | .00240 | .1148 | tracking 우수, POS reserve 실패 |
| A3 | History + reserve | 2.24 | 2.29 | 1.38 | **-.04** / 43.02 | .00562 | .1446 | margin 경향은 있으나 tracking 저하, 한 member reserve 실패 |
| A4 | History + rate+reserve | .90 | 1.39 | .87 | **4.29** / 39.31 | **.00217** | .1668 | **현재 safe-tracking 후보** |
| A5 | A4 + K=2 | 1.13 | **1.07** | **.58** | **-.24** / 43.85 | **.03269** | .1943 | displacement 우수, reserve 및 action 진동 실패 |

RMSE는 eight-member mean이고 reserve는 eight members 전체에서 가장 작은 값이다. A4만 모든 unseen member에서 설계 기준 `POS reserve ≥ 3 kPa`를 만족했다. 다만 A4도 일부 plateau에서 actuator/displacement oscillation이 보이고, POS/NEG/displacement per-step direction agreement `.60/.59/.64`는 임시 기준 `.8`을 넘지 못했다. 따라서 최종 성공으로 확정하지 않고 추가 seed 및 실기 gate 전의 **provisional winner**로 둔다.

전체 [validation summary](../../../pneu_rl/exp/lib4_0824_r3_id_validation_actual2048/summary.yaml)와 모델별 `members.csv`, 각 member의 main/6-control PNG 및 raw CSV를 저장했다.

- A2 worst member 07: [main](../../../pneu_rl/exp/lib4_0824_r3_id_validation_actual2048/260823_lib4_r3_history_rate_seed0/member_07/member_07_main.png) · [6-control](../../../pneu_rl/exp/lib4_0824_r3_id_validation_actual2048/260823_lib4_r3_history_rate_seed0/member_07/member_07_ctrl.png) · [CSV](../../../pneu_rl/exp/lib4_0824_r3_id_validation_actual2048/260823_lib4_r3_history_rate_seed0/member_07/validation.csv)
- A4 worst member 07: [main](../../../pneu_rl/exp/lib4_0824_r3_id_validation_actual2048/260823_lib4_r3_history_rate_reserve_seed0/member_07/member_07_main.png) · [6-control](../../../pneu_rl/exp/lib4_0824_r3_id_validation_actual2048/260823_lib4_r3_history_rate_reserve_seed0/member_07/member_07_ctrl.png) · [CSV](../../../pneu_rl/exp/lib4_0824_r3_id_validation_actual2048/260823_lib4_r3_history_rate_reserve_seed0/member_07/validation.csv)
- A5 representative member 00: [main](../../../pneu_rl/exp/lib4_0824_r3_id_validation_actual2048/260823_lib4_r3_history_rate_reserve_k2_seed0/member_00/member_00_main.png) · [6-control](../../../pneu_rl/exp/lib4_0824_r3_id_validation_actual2048/260823_lib4_r3_history_rate_reserve_k2_seed0/member_00/member_00_ctrl.png) · [CSV](../../../pneu_rl/exp/lib4_0824_r3_id_validation_actual2048/260823_lib4_r3_history_rate_reserve_k2_seed0/member_00/validation.csv)

## 최소 결론

1. **변위를 학습했는가:** yes. A2/A3/A4/A5의 unseen displacement RMSE는 `.58–1.38 mm`이며 Direct의 `2.95 mm`보다 낮다. 단, History-only가 없어 improvement를 history와 physical feature로 완전히 분해할 수는 없다.
2. **rate의 역할:** rate-only가 가장 낮은 POS pressure RMSE와 높은 displacement direction agreement를 보였다. 변화 방향과 dynamic tracking에는 rate가 핵심이다.
3. **reserve의 역할:** reserve-only는 여유를 크게 확보하는 경향이 있지만 tracking을 희생한다. reserve는 단독 추적 feature라기보다 control authority/safety 정보를 제공한다.
4. **rate+reserve의 역할:** A4는 A2보다 RMSE가 조금 크지만, POS reserve를 `-.30 → 4.29 kPa`로 회복하고 mean action change도 가장 작다. 현재 결과는 **rate가 tracking을 만들고 reserve가 그 tracking을 feasible pressure authority 안에 묶는다**는 해석을 지지한다.
5. **K=2의 역할:** displacement RMSE는 `.87 → .58 mm`로 개선됐지만 physical-command actual-step `mean |Δu|`가 `.00217 → .03269`로 약 15배 증가하고 POS reserve가 음수가 됐다. K=2 training logger는 두 sub-action 중 마지막 info만 남겨 이 within-chunk oscillation을 가렸으므로, matched actual-step validation에서 발견된 안정성 실패다.

따라서 현재 채택 순위는 `A4 rate+reserve K=1 > A2 rate-only > A5 K=2 > A3 reserve-only > incomplete Direct`다. 논문 결론 전에는 누락된 A1 History와 A0 1000 episode를 복구하고, A4 seed 1/2를 추가해야 한다.

## K=2 진동 진단

현재 K=2는 하나의 6-control command를 두 step 동안 hold하는 action repeat가 아니다. 한 관측 $o_t$에서 서로 독립적인 6차원 sub-action 두 개를 한꺼번에 출력하고, 두 번째 action은 $t+1$의 새 관측 없이 open-loop로 실행한다.

\[
\pi(o_t)=[a_t^{(1)},a_t^{(2)}],\qquad
R_t=r_t+\gamma r_{t+1}.
\]

R3-ID eight-member actual-step CSV를 chunk 내부와 chunk 경계로 분해한 결과는 다음과 같다.

| 구간 | mean `|Δu|` | `|Δu| > .02` 비율 |
|---|---:|---:|
| A4 K=1 전체 | .00217 | 1.6% |
| A5 K=2, sub-action 1→2 | .03253 | 46.0% |
| A5 K=2, 이전 sub-action 2→다음 sub-action 1 | .03286 | 46.5% |

따라서 진동은 chunk 내부에서만 생긴 일회성 jump가 아니라, policy가 두 sub-action phase에 따라 서로 다른 50 Hz command pattern을 학습한 결과다. 예를 들어 chamber NEG command의 평균은 첫 sub-action `.8807`, 두 번째 `.9255`이고, POS actuator out command는 `.8543 → .8845`로 체계적으로 갈린다. 첫/두 번째 sub-action correlation도 chamber POS `.275`, chamber NEG `.525`로 낮다.

가능성이 큰 원인은 다음 네 가지이며, 아직 하나로 확정하지 않는다.

1. 두 번째 sub-action은 중간 pressure/displacement feedback을 받지 않는 open-loop command다.
2. reward가 two-step discounted sum이므로 느린 displacement에서 pulse pair가 평균적으로 좋은 결과를 만들면 sub-step command 차이는 벌점을 받지 않는다.
3. 현재 reward의 `action_delta_rwd_coeff=0`이고, chunk 내부 consistency constraint도 없다.
4. action dimension이 `6 → 12`가 되며 automatic entropy target도 `-6 → -12`가 된다. 실제 episode 1000의 alpha도 A4 `.0545`, A5 `.0704`였으므로 exploration/entropy scale 차이가 진동을 강화했을 가능성이 있다. 이는 별도 entropy-matched pair로 확인해야 한다.

기존 training diagnostics는 매 chunk의 마지막 `info`만 저장해 첫 번째 sub-action과 within-chunk 변화량을 보지 못했다. 이후 K>1 학습은 physical sub-step별 control/reward/state를 기록하지 않으면 유효한 실험으로 판정하지 않는다.

## 후속 연구 계획과 stop gate

아래 계획의 실행 가능한 manifest, gated runner, P0 metric protocol과 회귀 검사는 [README의 후속 계획 실행 코드](README.md#71-후속-계획-실행-코드)에 고정했다. 새 실험은 ad-hoc 명령으로 설정을 바꾸지 않고 manifest job으로 추가하며, 모든 평가 결과에 protocol hash를 남긴다.

### P0 — 평가 기준부터 고정

현재 direction agreement는 `|Δref| > 1e-9`인 모든 step을 사용한다. 이 정의는 plateau의 float-level 변화와 pneumatic delay까지 오방향으로 셀 수 있다. R3-ID에서 A4 displacement agreement는 현재 `.643`이지만 `|Δref| > .005 mm/step`에서 `.838`이다. 이 값은 threshold를 유리하게 바꿔 A4를 합격시키라는 뜻이 아니라, 현재 `.8` gate가 아직 물리적으로 정의되지 않았다는 뜻이다.

추가 학습 전에 evaluator를 다음처럼 고정한다.

- transition: 실측 noise floor보다 큰 reference velocity만 사용한 direction agreement와, 독립 step-response에서 정한 causal lag를 적용한 agreement
- plateau: pressure/displacement 표준편차, peak-to-peak, settling time, high-frequency power를 별도 계산
- action: actual-step `mean/max |Δu|`, `|Δ²u|`, saturation, chunk 내부/경계 변화량
- safety: POS/NEG reserve minimum, chamber peak, constraint violation duration
- tracking: 전체 RMSE뿐 아니라 transition/plateau RMSE와 bias를 분리

성공 기준은 이 evaluator와 threshold를 checkpoint 비교 전에 고정하고 test 결과를 본 뒤 바꾸지 않는다.

### P1 — 인과관계와 seed 복구: 지금 바로 수행

1. 기존 A0 870 checkpoint는 보존하고, 동일 설정의 Direct seed 0을 새 이름으로 clean 1000-episode 재학습한다.
2. History-only seed 0을 1000 episode 학습한다.
3. A4 rate+reserve seed 1/2를 각각 1000 episode 학습한다.
4. 모두 같은 R3-ID eight-member, 2048 actual-step evaluator로 비교한다.

이 단계 전에는 `History가 효과적이다` 또는 `A4가 재현된다`고 결론내리지 않는다. A4가 seed 1/2 중 하나라도 reserve/safety gate를 반복적으로 깨거나 tracking 분산이 크면 provisional winner 지위를 철회한다. 논문에서 rate/reserve의 인과 효과를 주장하려면 최종적으로 A2/A3도 seed 1/2를 추가해 main-effect와 interaction을 단일 seed가 아닌 통계로 확인한다.

### P2 — A4 plateau 문제를 bias와 oscillation으로 분리

P0의 segment metric으로 먼저 원인을 나눈다.

- plateau mean error가 크고 진동은 작을 때만 integral on/off를 수행한다. leaky integral, limit, hard-reset/carry 규칙과 anti-windup을 함께 고정한다.
- plateau mean error는 작지만 진동이 크면 integral보다 action-rate penalty 또는 작은 CAPS를 먼저 A4에 한 항씩 추가한다.
- transition tracking까지 둔해지면 smoothness 계수를 폐기한다. RMSE 개선만으로 채택하지 않고 reserve와 settling time을 동시에 통과해야 한다.

즉 integral은 다음 실험의 기본값이 아니다. 현재 관찰된 진동에 integral을 바로 붙이면 wind-up과 overshoot를 키울 수 있다.

### P3 — K=2 원인 분리: 현재 A5는 중단 후보

현재 free two-action K=2에 seed를 더 쓰지 않는다. 아래 seed-0 pair만 먼저 수행한다.

1. `K2-hold`: 6차원 action 하나를 두 physical step에 동일하게 적용해 단순 decision-rate 감소 효과를 측정한다.
2. `K2-free + consistency`: 현재 12차원 구조에 `||a^(2)-a^(1)||²` 또는 bounded residual parameterization을 넣어 sequence 유연성은 남기되 pulse pair를 제한한다.
3. 필요할 때만 entropy target/alpha를 K=1과 matched한 진단 pair를 추가한다.

`K2-hold`가 안정적이면 기존 A5 실패 원인은 K=2 시간척도 자체가 아니라 독립 two-sub-action parameterization이다. 두 구조 모두 진동하면 25 Hz open-loop decision이 lib4에 맞지 않는 것이므로 K=1을 최종 구조로 유지한다. K=2는 displacement RMSE뿐 아니라 A4의 reserve, plateau stability, actual-step action variation을 모두 넘을 때만 재채택한다.

### P4 — training-bank overfitting과 reference generalization

A4가 P1/P2를 통과하면 다음 test-only reference를 순서대로 사용한다.

1. 현재 R3-ID unseen feasible bank
2. R4 edge bank: safety/graceful degradation 전용이며 검증 전까지 학습 금지
3. simulator trajectory replay로 만들지 않은 analytic dynamic reference: 진폭, 주기, baseline이 시간에 따라 변하는 sine/triangle/chirp 계열
4. 동일 analytic family의 real-safe 축소 버전

analytic reference는 원 곡선을 test 결과에 맞춰 변형하지 않고, 사전에 rate/reserve/initial-state feasibility만 인증한다. 이것이 data-derived R3에만 맞은 policy인지 동적 reference family에도 일반화되는지를 분리한다.

### P5 — novelty 검증과 real gate

안정적인 A4 계열을 확보한 뒤에만 state-dependent control authority를 추가한다.

- 같은 tracking error라도 pressure reserve, displacement velocity, recent action이 다른 state에서 bounded action sweep을 수행한다.
- short-horizon achievable `ΔP_act+/ΔP_act-/Δdisp`의 signed bound를 예측한다.
- history-only, raw rate+reserve, learned authority 세 가지의 held-out failure prediction력을 비교한다.
- real에서는 작은 stroke/low-rate부터 시작해 safety gate를 통과할 때만 amplitude와 frequency를 올리고, 반복 trial의 평균·표준편차·worst case를 보고한다.

raw rate+reserve만으로 같은 예측력과 제어 성능이 나오면 authority estimator를 novelty로 주장하지 않고 A4를 reserve-aware engineering controller로 축소한다. 반대로 authority가 unseen dynamic reference의 failure와 real transfer를 일관되게 예측할 때 contribution으로 승격한다.

## 기존 screening 표

| ID | model | seed | POS RMSE | NEG RMSE | disp RMSE | reserve min (POS/NEG) | reward final-100 | decision | failure / note |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| A0 | `260823_lib4_r3_direct_seed0` | 0 | 4.01 | 3.72 | 2.95 | 1.85 / 46.69 | -1249.95 | incomplete | 870 episodes |
| A1 | missing | 0 | — | — | — | — | — | rerun | History SAC |
| A2 | `260823_lib4_r3_history_rate_seed0` | 0 | .73 | 1.22 | .88 | -.30 / 39.15 | -258.82 | safety fail | + rate-only |
| A3 | `260823_lib4_r3_history_reserve_seed0` | 0 | 2.24 | 2.29 | 1.38 | -.04 / 43.02 | -326.63 | reject | + reserve-only |
| A4 | `260823_lib4_r3_history_rate_reserve_seed0` | 0 | .90 | 1.39 | .87 | 4.29 / 39.31 | -251.40 | provisional winner | + rate+reserve |
| A5 | `260823_lib4_r3_history_rate_reserve_k2_seed0` | 0 | 1.13 | 1.07 | .58 | -.24 / 43.85 | -274.93 | stability fail | + rate+reserve, K=2 |

각 행에는 model directory의 `cfg.yaml` path, matched validation output, 그리고 reference SHA256를 함께 연결한다. seed 1/2는 seed 0에서 후보를 1–2개로 좁힌 뒤만 추가한다.
