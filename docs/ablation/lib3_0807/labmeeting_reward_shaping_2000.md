# 랩미팅 1-page: reward shaping이 history-only 진동의 원인인가?

## 지난 피드백과 검증 질문

지난 피드백은 “history-only SAC가 잘 안 된 이유가 observation 부족이 아니라,
기존 reward가 plateau의 oscillation을 직접 벌주지 않았기 때문일 수 있지 않은가?”였다.
이를 분리하기 위해 observation과 reward를 독립적으로 바꾼 2x2를 2,000 episode까지
학습했다.

| Observation | 기존 tracking reward | reference-rate mismatch reward 추가 |
|---|---|---|
| H: pressure/action history | H-base | H-smooth |
| HR: H + explicit pressure rate | HR-base | HR-smooth |

\[
r_{\mathrm{base}}=-0.3\left(|e_+|+|e_-|\right)-0.10b_{\mathrm{ch}},
\qquad
r_{\mathrm{smooth}}=r_{\mathrm{base}}-0.025\sum_{c\in\{+,-\}}
\left|\frac{\dot P_{\mathrm{act},c}-\dot r_c}{s_c}\right|.
\]

중요한 점은 \(|\dot P|\) 자체가 아니라 **reference와의 rate mismatch**를 벌준다는
것이다. 따라서 reference ramp에서 필요한 pressure motion은 불필요하게 억제하지
않는다. 모든 arm은 같은 chamber-bound penalty, seed 0, 2,000 episode,
16-member fixed reference-bank final evaluation을 사용했다.

## 결과

| Arm | Combined RMSE (kPa) | POS / NEG plateau rate RMS (kPa/s) | Plateau action-Δ RMS |
|---|---:|---:|---:|
| H-base | 7.836 | 30.290 / 15.897 | 0.00542 |
| H-smooth | 5.834 | 11.506 / 3.372 | 0.00174 |
| HR-base | 1.047 | 1.459 / 12.410 | 0.03380 |
| HR-smooth | **1.025** | 9.806 / **3.340** | 0.02699 |

## Training reward 추이

![reward shaping training reward](../../../pneu_rl/exp/lib3_0831_histreward2x2_analysis/history_rate_reward_2x2_training_reward.png)

H와 HR을 panel로 분리했다. 각 panel에서 reward가 올라가는 것은 학습 진행을 보여
주지만, base와 smooth는 **서로 다른 scalar objective**이므로 raw reward 값 자체를
공통 성능 점수로 비교하면 안 된다. 이 그래프는 H arm이 약 1,200--1,300 episode
이후 안정적으로 개선되지 않는 현상과 HR arm의 상대적으로 안정적인 수렴을 보여 주는
진단용이다. 결론은 위의 fixed-bank RMSE, plateau pressure-rate RMS, action variation으로
판단한다.

### 발표할 결론

> **Reward 누락은 H-base의 oscillation/under-damped behavior에 실제로 기여했다.
> 그러나 그것만으로 history-only의 finite-budget tracking gap은 설명되지 않는다.**

- H에서 smooth reward는 combined RMSE를 **7.836 -> 5.834 kPa**로, POS/NEG plateau
  pressure motion을 62%/79% 줄였다. 즉 “reward를 바꿔도 아무 변화가 없었다”는
  표현은 틀리다.
- 하지만 H-smooth는 여전히 reference를 attenuate/bias하며, 같은 smooth reward의
  HR-smooth (**1.025 kPa**)와는 큰 차이가 남는다. H-smooth -> HR-smooth의 combined
  RMSE 감소는 **82%**다.
- HR에서는 smooth reward가 combined RMSE를 1.047 -> 1.025 kPa로 거의 바꾸지
  않았다. NEG tracking/rate는 좋아졌지만 POS tracking/rate는 나빠졌다. 즉 rate
  reward는 HR에 대해 uniform improvement가 아니라 **channel-wise trade-off**를
  만들었다.
- 두 H arm은 약 episode 1,200--1,300에서 좋아진 뒤 2,000까지 안정적으로 유지되지
  않았다. post-hoc best-era full-bank checkpoint도 H-base 4.207, H-smooth 4.736 kPa로
  HR final보다 훨씬 나빴다. 따라서 “마지막 checkpoint를 잘못 골라서 H가 진 것”도
  주된 설명은 아니다.

## 교수님께 말할 문장

> 지난주 피드백대로 rate-mismatch reward를 추가해 history-only도 2,000 episode까지
> 다시 확인했습니다. Reward는 history-only의 plateau motion과 RMSE를 분명히
> 개선했지만, 같은 objective 아래 explicit rate observation을 준 controller와는
> 여전히 큰 tracking gap이 남았습니다. 따라서 현재 결과는 “history-only는 원리적으로
> 불가능하다”가 아니라, **reward misalignment는 부분 원인이고, explicit rate는 이
> finite-budget MLP-SAC에서 별도의 strong coordinate/optimization bias**라는
> 결론입니다.

## 이 실험의 범위와 다음 결정

이것은 training seed 0의 원인 분리 screen이다. Reference member의 error bar는
training seed uncertainty가 아니다. 따라서 논문에서 stochastic superiority를 주장하려면
finalist pair에 대해서만 추가 seed가 필요하다.

지금은 reward coefficient를 더 sweep하기보다, reward 축은 이 결과로 닫고
representation/command-bandwidth 축을 정리하는 편이 낫다. 특히 K=2 archive는
command total variation이 커서 smooth actuation의 근거가 아니므로, 같은 physical
interaction budget의 H+rate K=2 및 command-hold K=2를 분리해야 한다.

## Controller별 40-s 결과: 모든 그림의 공통 형식

모든 그림은 같은 member 01의 약 40-s rollout이다. 위부터 chamber POS/NEG,
actuator POS, actuator NEG 순서이며, **red = POS, blue = NEG, black = reference**로
고정했다. 따라서 controller마다 색의 의미가 바뀌지 않는다.

### H-base: history-only + 기존 reward

![H-base 40-s result](../../../pneu_rl/exp/lib3_0831_histreward2x2_analysis/h_base_standard_40s.png)

| Observation / reward | Combined RMSE | Plateau rate RMS (POS / NEG) | Action-Δ RMS |
|---|---:|---:|---:|
| H / base | 7.836 kPa | 30.290 / 15.897 kPa/s | 0.00542 |

### H-smooth: history-only + rate-mismatch reward

![H-smooth 40-s result](../../../pneu_rl/exp/lib3_0831_histreward2x2_analysis/h_smooth_standard_40s.png)

| Observation / reward | Combined RMSE | Plateau rate RMS (POS / NEG) | Action-Δ RMS |
|---|---:|---:|---:|
| H / smooth | 5.834 kPa | 11.506 / 3.372 kPa/s | **0.00174** |

H-smooth는 H-base보다 visible oscillation과 pressure motion을 줄인다. 그러나
reference plateau를 충분히 따라가지 못하는 attenuation/bias가 남아 있으므로,
smoothness만으로 성공이라고 판단하면 안 된다.

### HR-base: history + explicit pressure rate + 기존 reward

![HR-base 40-s result](../../../pneu_rl/exp/lib3_0831_histreward2x2_analysis/hr_base_standard_40s.png)

| Observation / reward | Combined RMSE | Plateau rate RMS (POS / NEG) | Action-Δ RMS |
|---|---:|---:|---:|
| HR / base | 1.047 kPa | **1.459** / 12.410 kPa/s | 0.03380 |

### HR-smooth: history + explicit pressure rate + rate-mismatch reward

![HR-smooth 40-s result](../../../pneu_rl/exp/lib3_0831_histreward2x2_analysis/hr_smooth_standard_40s.png)

| Observation / reward | Combined RMSE | Plateau rate RMS (POS / NEG) | Action-Δ RMS |
|---|---:|---:|---:|
| HR / smooth | **1.025 kPa** | 9.806 / **3.340** kPa/s | 0.02699 |

HR-smooth는 NEG rate/motion은 개선하지만 POS rate/motion은 나빠진다. 즉 smooth
reward는 HR에서 uniform improvement가 아니라 POS/NEG trade-off를 바꾼다.

상세 protocol과 paired effect는 [history_rate_reward_2x2.md](history_rate_reward_2x2.md),
원자료는 [final metrics](../../../pneu_rl/exp/lib3_0831_histreward2x2_analysis/final_checkpoint_metrics.csv)와
[paired effects](../../../pneu_rl/exp/lib3_0831_histreward2x2_analysis/paired_effects.csv)에 있다.
