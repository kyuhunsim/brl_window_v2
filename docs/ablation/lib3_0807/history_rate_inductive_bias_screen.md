# History와 pressure-rate inductive bias: 3-way learning screen

## 한 줄 결론

현재 결과는 **명시적 pressure rate를 추가하면, history 안에서 차분을 policy가
스스로 찾게 두는 것보다 제한된 학습 budget에서 더 빠르고 낮은 tracking error에
도달한다**는 empirical evidence다. 즉 `p_dot`은 새 sensor 정보가 아니라,
history에서 계산 가능한 정보를 control-aligned 좌표로 미리 제공하는
**inductive bias**로 해석한다.

![SAC, History SAC, and History SAC + rate learning dynamics](../../../pneu_rl/exp/lib3_0830_history_plateau/sac_history_rate_learning_dynamics.png)

그림에는 run name, lib version, seed를 넣지 않았다. 발표 본문에서는 아래의 동일
protocol을 함께 명시한다.

## 비교의 의미

| Arm | Policy 입력의 시간 정보 | 이 비교가 답하는 질문 |
|---|---|---|
| SAC | 현재 observation만 | history가 없으면 현재 pressure snapshot으로 충분한가? |
| History SAC | 10-step pressure/action history | temporal context 자체가 학습을 돕는가? |
| History SAC + rate | 동일한 10-step history + explicit \(\dot p\) | history에서 유도 가능한 rate 좌표를 명시하면 최적화가 쉬워지는가? |

세 arm은 동일한 lib3 plant, base reward, 512-step episode, 1,000 episode
budget, training seed 0, reference profile을 사용했다. History와 History+rate는
동일한 `history_action_sac`, 10-step pressure/action history를 사용하며, 후자에만
4개 chamber의 normalized pressure-rate feature가 추가된다. 따라서 **History
SAC → History SAC + rate**가 rate inductive bias를 직접 보는 핵심 비교다.

## 수치 요약

아래 수치는 각 episode의 POS/NEG RMSE를
\(\sqrt{(\mathrm{RMSE}_{\rm pos}^2+\mathrm{RMSE}_{\rm neg}^2)/2}\)로 합친 뒤,
50-episode moving mean으로 계산했다.

| Arm | 2 kPa 도달 episode | 최종 50-episode combined RMSE | 최종 50-episode return |
|---|---:|---:|---:|
| SAC | 1,000 episode 내 미도달 | 9.577 kPa | -2318.2 |
| History SAC | 1,000 episode 내 미도달 | 3.411 kPa | -847.0 |
| History SAC + rate | **367** | **1.757 kPa** | **-433.0** |

따라서 그림은 두 단계의 결과를 보인다.

1. `SAC → History SAC`: temporal context가 없을 때보다 learning/tracking이 크게 개선된다.
2. `History SAC → History SAC + rate`: 같은 history를 유지해도 explicit rate가
   2 kPa 수준에 더 이르게 도달하고 최종 error도 낮다.

## 발표용 해석

> `p_dot`은 history가 없으면 만들 수 없는 새 물리량이 아닙니다. 고정 길이
> history MLP는 원리적으로 pressure 차분을 표현할 수 있습니다. 다만 finite
> replay data와 finite update budget 안에서 차분을 찾아내고 이를 damping 방향의
> action과 결합하는 것은 별도 optimization 문제입니다. \(\dot p\)를 직접
> 제공하면 “현재 압력이 reference보다 높은가/낮은가”뿐 아니라 “어느 방향과
> 속도로 움직이고 있는가”가 분리되어, 이 조건에서는 sample-efficient learning
> coordinate로 작동했습니다.

이 문장은 **`p_dot`이 수학적으로 필수다** 또는 **POMDP를 해결했다**는 주장이
아니다. Pressure/action history로 hidden valve/pump dynamics를 근사 추정하는
policy에 대해, rate coordinate가 그 추정을 더 쉽게 최적화하도록 만든다는 주장이다.

## 주장 범위와 다음 사용법

이것은 seed 0의 matched-budget screen이다. 따라서 논문/발표에서는
“sample efficiency를 증명했다”보다 다음 표현이 정확하다.

> 동일 protocol의 seed-0 screen에서 History SAC + rate는 더 빠른 error 감소와
> 더 낮은 final tracking RMSE를 보였으며, pressure-rate coordinate가 finite-budget
> learning의 sample-efficiency bias로 작동함을 지지한다.

최종 논문 표에는 rate finalist를 최소 3 seeds와 held-out reference/OOD trajectory로
재검증해 mean ± standard deviation 및 success rate를 제시한다. 그 뒤 observation
feature sweep을 늘리기보다, validated simulator, reference envelope, sim-to-real,
OOD evaluation으로 결론을 확장하는 것이 더 강한 연구 흐름이다.

원자료: [learning summary](../../../pneu_rl/exp/lib3_0830_history_plateau/sac_history_rate_learning_summary.csv),
[plot script](../../../pneu_rl/scripts/plot_lib3_observation_reward_comparison.py).

## 다음 실험과의 관계

이 3-way screen은 `History SAC`와 `History SAC + rate`의 matched-reward 비교를
이미 완료한 결과다. 따라서 같은 조건으로 rate 실험을 다시 반복하지 않는다. 다음
관측 설계의 새 축은 real에서 측정 가능한 6-channel flow
(`pump→chamber` 2개, `chamber→actuator` 2개, `actuator→atmosphere` 2개)이며,
기존 HR 결과는 그 flow 실험의 baseline으로 사용한다. simulator 내부 `q`나
`q_dot`을 real flow measurement로 간주하지 않고, 실제 sensor map과 delay/filter/noise를
반영한 `flow6` observation을 별도로 검증해야 한다.
