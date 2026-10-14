# lib2 강화학습 ablation 기록

이 문서는 lib2에서 수행한 최신 history/action/rate/K=2 모듈 실험과, 그 결과를 해석하기 위한 과거 기준 모델을 정리한다. 검증 그림은 모두 실제 simulator에 policy를 다시 넣은 결과이며, PID는 사용하지 않았다.

## 먼저 분리해야 하는 두 가지 reference

lib2 결과를 비교할 때 가장 중요한 변수는 구조보다 reference protocol이다.

### 이번 2026-08-11 ablation

이번 새 모델들은 다음의 legacy RandomRef로 학습했다.

```text
reference-kind: legacy_random
seed: 61098
POS offset: 140–240 kPa
NEG offset: 10–32 kPa
POS/NEG max amplitude: 20 / 5 kPa
period: 최대 10 s
transition: RandomRef parameter를 block에서 즉시 변경
```

이 reference는 parameter jump 때문에 simulator reference rate가 매우 크게 튄다. 실제 2048-step 검증에서도 POS reference rate 최대가 약 `3529 kPa/s`, NEG가 약 `982 kPa/s`로 계산됐다. 이는 lib2 actuator가 안정적으로 따라갈 수 있는 smooth trajectory의 난이도가 아니라, 불연속 reference에 대한 stress test에 가깝다.

### 과거에 잘 되었던 2026-07-29 모델

과거 `260729_lib2_final_*` 모델은 `joint_reachable_random`을 사용했다. 이 reference는 simulator command-map에서 jointly reachable한 pressure point를 사용하고, transition rate와 frequency를 제한한다.

따라서 다음 비교는 반드시 분리해서 써야 한다.

```text
legacy RandomRef 결과 = discontinuous/high-rate reference에 대한 robustness
joint_reachable_random 결과 = 도달가능 pressure domain 안에서의 tracking
```

이번 260811 모델들의 POS RMSE가 24–27 kPa로 나왔다고 해서 history/rate 구조가 무조건 실패했다고 단정할 수 없다. 학습 reference 자체가 high-rate jump였기 때문이다. 반대로 과거 모델의 좋은 RMSE를 lib2의 모든 reference에 대한 일반적 우수성으로 해석해서도 안 된다.

## 공통 조건

- lib2 pneumatic simulator, 2-action POS/NEG control
- 초기 state: 대기압 `101.325 kPa`
- physical command: `u = 0.8 + 0.2a`
- frequency: 50 Hz
- simulator delay: 0.1 s
- action history: 최근 10 step
- observation normalization: 사용
- external exploration noise: 0
- warm-up: 첫 10 episode 동안 physical action `0.95` 고정
- warm-up 중 update 보류, 이후 policy training 시작
- buffer size: `500000`
- gamma: `0.9`
- training: 1000 episode, horizon 2048, episode state reset
- validation: 1 episode, 2048 simulator step, tolerance 5 kPa, PID off

## 최신 260811 ablation 결과

검증 output은 각 모델의 `screen_eval_0813_ablation_2048_2048step`에 저장되어 있다. 아래 reward는 validation episode total reward, RMSE와 success ratio는 2048-step validation 값이다.

| 모델 | 구성 | train 완료 | validation reward | POS/NEG RMSE | POS/NEG bias | ±5 kPa 비율 | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| `260811_lib2_h2048_history_action_seed0` | history + action | 완료 | -10715.6 | 23.97 / 5.84 | 3.90 / .88 | .521 / .941 | POS tracking이 크게 남고 NEG는 비교적 안정 |
| `260811_lib2_h2048_history_action_rate_seed0` | history + action + rate | 완료 | -11201.3 | 24.60 / 6.01 | 7.68 / 1.36 | .535 / .934 | rate 추가만으로 legacy jump를 해결하지 못함 |
| `260811_lib2_h2048_history_action_rate_k2_seed0` | 위 + K=2 | 완료 | -21714.2 | 26.81 / 4.26 | -1.78 / 1.11 | .521 / .957 | NEG는 약간 좋아졌지만 POS가 여전히 실패 |
| `260811_lib2_h2048_history_action_rate_k2_caps001_seed0` | K=2 + mean-action CAPS(.01) | 완료 | -21621.5 | 26.99 / 4.29 | -1.96 / .89 | .520 / .956 | CAPS가 legacy jump를 보상하지 못함 |
| `260811_lib2_h2048_history_action_rate_k2_mpobs1_seed0` | K=2 + MPOBS1 | 완료 | -21724.2 | 26.48 / 4.20 | -.49 / .90 | .522 / .959 | POS offset은 조금 줄지만 전체 실패 |
| `260811_lib2_h2048_history_action_rate_k2_mpobs1_caps001_seed0` | K=2 + MPOBS1 + CAPS(.01) | 완료 | -21886.6 | 26.75 / 4.37 | -1.45 / 1.15 | .516 / .950 | 세 모듈 결합의 이점 확인 안 됨 |
| `260811_lib2_history_action_rate_seed0` | 512 carry + rate | **48 episode에서 중단** | -347682.6 | 656.48 / 18.42 | -548.83 / -15.66 | .018 / .147 | 학습 발산. 1000 episode 완료 모델과 비교 불가 |

### 검증 그림

<details>
<summary>History + action baseline</summary>

[검증 PNG](../../../pneu_rl/exp/260811_lib2_h2048_history_action_seed0/screen_eval_0813_ablation_2048_2048step/trajectory.png) · [metrics](../../../pneu_rl/exp/260811_lib2_h2048_history_action_seed0/screen_eval_0813_ablation_2048_2048step/metrics.yaml) · [cfg](../../../pneu_rl/models/260811_lib2_h2048_history_action_seed0/cfg.yaml)

POS reference jump 뒤에 actual pressure가 느리게 따라가며 큰 offset이 남는다. NEG는 비교적 reference 주변에 있지만 작은 vibration과 jump response 지연이 있다.

</details>

<details>
<summary>History + action + pressure rate</summary>

[검증 PNG](../../../pneu_rl/exp/260811_lib2_h2048_history_action_rate_seed0/screen_eval_0813_ablation_2048_2048step/trajectory.png) · [metrics](../../../pneu_rl/exp/260811_lib2_h2048_history_action_rate_seed0/screen_eval_0813_ablation_2048_2048step/metrics.yaml) · [cfg](../../../pneu_rl/models/260811_lib2_h2048_history_action_rate_seed0/cfg.yaml)

rate 정보가 있어도 reference jump 자체가 너무 빠르기 때문에 POS chamber/actuator가 필요한 시점에 따라가지 못한다.

</details>

<details>
<summary>History + rate + K=2</summary>

[검증 PNG](../../../pneu_rl/exp/260811_lib2_h2048_history_action_rate_k2_seed0/screen_eval_0813_ablation_2048_2048step/trajectory.png) · [metrics](../../../pneu_rl/exp/260811_lib2_h2048_history_action_rate_k2_seed0/screen_eval_0813_ablation_2048_2048step/metrics.yaml) · [cfg](../../../pneu_rl/models/260811_lib2_h2048_history_action_rate_k2_seed0/cfg.yaml)

K=2는 command switching을 줄이는 대신 2-step action을 미리 결정한다. 이 legacy reference에서는 POS jump를 해결하지 못하고, NEG만 소폭 개선된다.

</details>

<details>
<summary>History + rate + K=2 + CAPS(.01)</summary>

[검증 PNG](../../../pneu_rl/exp/260811_lib2_h2048_history_action_rate_k2_caps001_seed0/screen_eval_0813_ablation_2048_2048step/trajectory.png) · [metrics](../../../pneu_rl/exp/260811_lib2_h2048_history_action_rate_k2_caps001_seed0/screen_eval_0813_ablation_2048_2048step/metrics.yaml) · [cfg](../../../pneu_rl/models/260811_lib2_h2048_history_action_rate_k2_caps001_seed0/cfg.yaml)

CAPS가 action을 부드럽게 만드는 효과는 있어도, 불가능에 가까운 reference jump를 추적할 수 있게 만들지는 않는다.

</details>

<details>
<summary>History + rate + K=2 + MPOBS1</summary>

[검증 PNG](../../../pneu_rl/exp/260811_lib2_h2048_history_action_rate_k2_mpobs1_seed0/screen_eval_0813_ablation_2048_2048step/trajectory.png) · [metrics](../../../pneu_rl/exp/260811_lib2_h2048_history_action_rate_k2_mpobs1_seed0/screen_eval_0813_ablation_2048_2048step/metrics.yaml) · [cfg](../../../pneu_rl/models/260811_lib2_h2048_history_action_rate_k2_mpobs1_seed0/cfg.yaml)

MPOBS1을 추가해도 POS error가 지속된다. predictor 정보를 넣는 것보다 reference의 dynamic feasibility가 먼저 맞아야 한다.

</details>

<details>
<summary>History + rate + K=2 + MPOBS1 + CAPS(.01)</summary>

[검증 PNG](../../../pneu_rl/exp/260811_lib2_h2048_history_action_rate_k2_mpobs1_caps001_seed0/screen_eval_0813_ablation_2048_2048step/trajectory.png) · [metrics](../../../pneu_rl/exp/260811_lib2_h2048_history_action_rate_k2_mpobs1_caps001_seed0/screen_eval_0813_ablation_2048_2048step/metrics.yaml) · [cfg](../../../pneu_rl/models/260811_lib2_h2048_history_action_rate_k2_mpobs1_caps001_seed0/cfg.yaml)

MPOBS1과 작은 CAPS의 결합도 legacy jump 상황에서는 POS offset과 pressure rate 문제를 해결하지 못한다.

</details>

<details>
<summary>512 carry + rate, 중단 모델</summary>

[검증 PNG](../../../pneu_rl/exp/260811_lib2_history_action_rate_seed0/screen_eval_0813_ablation_2048_2048step/trajectory.png) · [metrics](../../../pneu_rl/exp/260811_lib2_history_action_rate_seed0/screen_eval_0813_ablation_2048_2048step/metrics.yaml) · [cfg](../../../pneu_rl/models/260811_lib2_history_action_rate_seed0/cfg.yaml)

이 모델은 1000 episode를 끝낸 모델이 아니다. episode 48에서 reward가 약 `-656k`까지 내려갔고 validation도 POS pressure가 정상 영역에 머물지 않았다. carry가 원인이라고 단정하기보다, 512 carry + legacy jump + rate feature 조합의 학습 안정성이 깨진 사례로 기록한다.

</details>

## 과거 기준 모델과의 비교

과거의 주요 lib2 모델은 서로 다른 reference를 사용했으므로 다음처럼 기록한다.

| 모델 | 학습 reference | 검증 POS/NEG RMSE | 결론 |
|---|---|---:|---|
| `260723_screen_lib2_h2048_reset_sac_seed0` | legacy RandomRef | 23.38 / 6.77 | legacy 기준에서는 POS 실패 |
| `260723_screen_lib2_h2048_reset_sac_mpobs_caps_seed0` | legacy RandomRef | 24.86 / 6.00 | legacy 기준에서 MPOBS+CAPS도 POS 실패 |
| `260729_lib2_final_sac_hist10_g099_seed0` | `joint_reachable_random` | 5.18 / 5.64 | reachable reference에서 양호 |
| `260729_lib2_final_sac_mpobs_caps_hist10_g099_seed0` | `joint_reachable_random` | 4.13 / 5.94 | reachable reference에서 선행 구성의 양압 개선 |
| `260729_lib2_final_sac_mpobs_caps_iterm_hist10_g099_seed0` | `joint_reachable_random` | 11.97 / 7.80 | integral까지 추가하면 오히려 불안정 |

이 비교는 lib2에서 CAPS/MPOBS가 언제든 필요한지에 대한 직접 증거가 아니다. 더 강한 해석은 다음과 같다.

```text
legacy RandomRef: reference jump가 actuator capacity를 크게 초과 → 대부분 구조가 POS 실패
joint_reachable_random: pressure/rate domain이 reachable → History SAC도 성공, MPOBS+CAPS도 양호
```

즉 lib2에서는 먼저 reference protocol을 통일한 뒤 모듈 ablation을 해야 한다. 이번 260811 세트는 legacy stress test로서 의미가 있지만, 선행 연구 모듈의 순수한 효과를 측정하는 clean ablation은 아니다.

## lib2 중간 결론

1. 최근 260811 h2048 ablation의 실패는 단순히 history/rate/K=2가 나빠서가 아니다. POS reference rate가 최대 약 `3529 kPa/s`인 legacy RandomRef를 사용했기 때문에, simulator pressure capacity를 벗어난 target이 학습과 평가에 들어갔다.
2. K=2, CAPS, MPOBS1을 붙여도 reference가 도달 불가능하면 POS offset은 사라지지 않는다.
3. 512 carry 모델은 48 episode에서 발산했으므로, 이 하나의 결과로 carry 자체를 폐기할 수는 없지만 legacy jump와 carry를 동시에 사용한 protocol은 안전하지 않다.
4. 7월 말의 좋은 결과는 `joint_reachable_random`에서 나온 것이다. 따라서 lib2에서 선행 논문 모듈의 ablation을 다시 하려면 `lib2_joint_reachable.yaml`을 고정하고 Pure/History/CAPS/MPOBS를 같은 seed·horizon·reset 조건에서 비교해야 한다.
5. 현재 lib2 기록에서 확정할 수 있는 것은 “MPOBS+CAPS가 reachable reference에서 도움을 줄 가능성”이지 “항상 필요하다” 또는 “불필요하다”가 아니다.

## 재현 명령과 원자료

최신 260811 학습 명령은 [run_5_history_rate_modules_h2048_reset.sh](run_5_history_rate_modules_h2048_reset.sh)에 기록되어 있다. 검증은 각 모델에 대해 다음 output directory에 저장했다.

```text
pneu_rl/exp/<model>/screen_eval_0813_ablation_2048_2048step/
```

모든 검증은 `reference_content_match: true`, `training_pid_enabled: false`, `evaluation_pid_enabled: false`를 확인했다.
