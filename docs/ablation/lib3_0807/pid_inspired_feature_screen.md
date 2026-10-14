# PID-inspired structured feature screen (lib3, seed 0)

> 발표와 논문의 **주 비교는** 이제 [SAC / History SAC / History SAC + rate
> 3-way screen](history_rate_inductive_bias_screen.md)으로 한정한다. 이 문서는
> PIDE/GPIDE-lite를 왜 보조 탐색으로만 남기는지 기록한 supporting analysis다.

## 이 실험의 질문

압력/action history를 이미 쓰는 SAC에서, history 안에 선형적으로 들어 있는
`error derivative`를 명시적으로 꺼내 주는 것이 실제로 유의미한가를 확인했다.
여기서 확인하려는 것은 **새 정보를 추가했는가**가 아니라, MLP-SAC가 시간
차분·누적을 직접 학습하지 않아도 되게 하는 **coordinate / optimization
inductive bias**가 있는가이다.

비교는 동일한 lib3 plant, reward, history length, 1,000 episode budget,
training seed 0, 그리고 16개 fixed reference-bank 평가에서 수행했다. Error bar는
reference member의 표준편차이며, training seed 신뢰구간이 아니다.

## NeurIPS 2023 논문과의 관계

Ian Char와 Jeff Schneider의 [PID-Inspired Inductive Biases for Deep
Reinforcement Learning in Partially Observable Control Tasks (NeurIPS
2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/ba1c5356d9164bb64c446a4b690226b0-Paper-Conference.pdf)는 tracking task에서
`error`, `integral(error)`, `error derivative`를 PIDE로 제공하고, 더 일반적인
문제를 위해 learned projection과 여러 temporal aggregation head를 쓰는 GPIDE를
제안한다. 이는 정규 NeurIPS 2023 conference paper다.

우리의 구현은 논문의 주장 전체를 재현한 것이 아니라, pneumatic tracking에 맞춘
**첫 번째 feature screen**이다.

| Arm | 이번에 policy에 추가한 feature | 논문과의 정확한 관계 |
|---|---|---|
| `e_dot` | \(\dot e_t=[(r_t-P_{act,t})-(r_{t-1}-P_{act,t-1})]/\Delta t\) (2 ch.) | D-term만 쓴 최소 baseline |
| `PIDE` | \(e_t,\;\sum_{i\in\text{fixed history}}e_i\Delta t,\;\dot e_t\) (6 ch.) | PIDE idea에 가깝다. 단 integral은 full episode가 아닌 현재 fixed history window이다. |
| `GPIDE-lite` | \(\Delta o_t\) (4), current physical action \(u_t\) (6), \(\Delta u_t\) (6) | 논문의 full GPIDE가 아니다. learned projection, multi-head temporal aggregation, learned decoder가 없는 fixed 16-D difference/action baseline이다. |

따라서 `GPIDE-lite`가 좋지 않았다는 결과를 “NeurIPS GPIDE가 실패했다”라고
말하면 안 된다. 반대로 이 실험에서 말할 수 있는 것은 *우리의 fixed
difference/action shortcut*이 이 plant와 budget에서는 좋은 coordinate가 아니었다는
것뿐이다.

## HR / HRR를 포함한 학습 reward 추이

![HR, HRR, and PID-inspired feature reward](../../../pneu_rl/exp/lib3_0906_structured_history_analysis/feature_screen_with_hr_hrr_training_reward.png)

- 실선은 이번 structured screen의 `e_dot`, `PIDE`, `GPIDE-lite`이고, 점선은
  기존 HR (history+rate) / HRR (history+rate+reserve) archive다.
- 다섯 조건은 같은 reference profile, history length, horizon, carry protocol,
  training seed 0을 썼다. 그러나 **HR/HRR archive의 chamber-bound coefficient는
  0.0, 새 screen은 0.1**이다.
- 따라서 이 그림은 “어느 policy가 reward가 더 높으니 더 좋다”는 성능표가 아니라
  **학습이 어느 시점에 plateau/stability 문제를 보이는지 확인하는 convergence
  diagnostic**이다. Feature 효과의 결론은 아래 fixed-bank RMSE와 motion 지표로만
  낸다.

## feature별 결과: 한 조건씩 분리

### 1. `e_dot`: 가장 작은 D-coordinate

![e_dot 40-s result](../../../pneu_rl/exp/lib3_0906_structured_history_analysis/e_dot_standard_40s.png)

| 추가 feature | Combined RMSE | RMSE 2.0 / 1.5 kPa 최초 도달 | Plateau rate RMS (POS / NEG) | Action-Δ RMS | Tolerance |
|---|---:|---:|---:|---:|---:|
| \(\dot e\) (2 ch.) | **1.104 +/- 0.073 kPa** | **227 / 424 epi** | 19.623 / 5.105 kPa/s | 0.0105 | **0.9996** |

이번 screen의 tracker 기준점이다. POS trace에는 plateau ripple이 남아 있으므로
“진동이 완전히 해결됐다”는 결론은 아니다. 다만 동일 budget에서 tracking을 가장
빨리, 가장 낮은 reference-bank RMSE로 만들었다.

### 2. `PIDE`: PID tracking summary

![PIDE 40-s result](../../../pneu_rl/exp/lib3_0906_structured_history_analysis/pide_standard_40s.png)

| 추가 feature | Combined RMSE | RMSE 2.0 / 1.5 kPa 최초 도달 | Plateau rate RMS (POS / NEG) | Action-Δ RMS | Tolerance |
|---|---:|---:|---:|---:|---:|
| \(e,\int e,\dot e\) (6 ch.) | 1.289 +/- 0.284 kPa | 233 / 461 epi | **14.398 / 7.107 kPa/s** | 0.0224 | 0.9936 |

PIDE는 `e_dot`보다 pressure motion은 낮지만 tracking은 약 17% 나쁘고 command
variation은 더 크다. 따라서 이 조건에서는 **PIDE가 우월하다**가 아니라
tracking–motion trade-off를 만든다고 정리하는 것이 맞다. Integral window나 scale을
별도로 최적화하지 않았으므로 PID architecture 자체에 대한 부정적 결론도 아직 낼 수
없다.

### 3. `GPIDE-lite`: fixed difference/action shortcut

![GPIDE-lite 40-s result](../../../pneu_rl/exp/lib3_0906_structured_history_analysis/gpide_lite_standard_40s.png)

| 추가 feature | Combined RMSE | RMSE 2.0 / 1.5 kPa 최초 도달 | Plateau rate RMS (POS / NEG) | Action-Δ RMS | Tolerance |
|---|---:|---:|---:|---:|---:|
| \(\Delta o,u,\Delta u\) (16 ch.) | 3.738 +/- 0.460 kPa | 364 / 미도달 epi | 11.413 / **3.867 kPa/s** | **0.0053** | 0.7639 |

낮은 pressure motion과 command variation만으로 이 arm을 진동 억제 성공으로 볼 수
없다. Reference를 충분히 따라가지 못하면 pressure 자체를 적게 움직여도 되기
때문이다. 이 arm은 **under-tracking negative baseline**이다.

## 한 장 표와 결론

| Arm | Combined RMSE (kPa) | Plateau pressure-rate RMS (kPa/s) | Plateau action-Δ RMS | 해석 |
|---|---:|---:|---:|---|
| `e_dot` | **1.104 +/- 0.073** | 14.476 | 0.0105 | 이번 budget의 best tracker |
| `PIDE` | 1.289 +/- 0.284 | 11.404 | 0.0224 | tracking–motion trade-off |
| `GPIDE-lite` | 3.738 +/- 0.460 | **8.572** | **0.0053** | under-tracking; smoothness 성공 아님 |

교수님께는 다음처럼 말하는 것이 가장 정확하다.

> NeurIPS의 full GPIDE를 그대로 구현한 결과는 아닙니다. 먼저 history에 대해
> D-only, PID summary, fixed difference/action summary라는 세 coordinate를 같은
> SAC budget에서 screening했습니다. 그 결과 error derivative가 tracking을 가장
> 빨리 안정화했고, PIDE는 일부 pressure motion을 낮추는 대신 tracking과 action
> variation trade-off를 보였으며, fixed GPIDE-lite는 under-tracking했습니다.
> 따라서 현재의 지지는 “physics/tracking-aligned derivative coordinate가 이
> pneumatic plant의 finite-budget learning에 강한 bias로 작동한다”까지입니다.

## 이 결과가 말하지 않는 것과 다음 판단

이 seed-0 screen만으로 `p_dot`/`e_dot`이 수학적으로 필수라고 증명할 수는 없다.
고정 길이 history에서 차분은 선형적으로 계산 가능하므로, representational capacity가
충분하고 최적화가 잘 되면 history-only MLP도 같은 함수를 표현할 수 있다. 이번
결과는 그 함수가 **주어진 finite data/optimization budget에서 쉽게 학습되었는가**를
보는 empirical evidence다.

다음 단계는 두 갈래 중 하나를 명확히 선택하면 된다.

1. **논문 재현/architecture 비교가 목적이면**, official GPIDE처럼 learned
   projection + summation/exponential-smoothing/attention aggregation head를 구현하고,
   `e_dot` 및 GRU와 같은 interaction budget에서 비교한다. 이때 `GPIDE-lite`는
   full GPIDE의 proxy로 쓰지 않는다.
2. **pneumatic controller가 목적이면**, 현 시점에서는 새 feature sweep을 늘리기보다
   `e_dot` 또는 rate-based finalist를 seed 1--2와 40-s command/pressure motion
   acceptance rule로 재검증한 뒤 lib4/real로 옮기는 편이 더 강한 연구 흐름이다.

원자료: [full-bank metrics](../../../pneu_rl/exp/lib3_0906_structured_history_analysis/fullbank_metrics.csv),
[learning dynamics](../../../pneu_rl/exp/lib3_0906_structured_history_analysis/learning_dynamics.csv),
[per-member paired metrics](../../../pneu_rl/exp/lib3_0906_structured_history_analysis/per_member_metrics.csv).
