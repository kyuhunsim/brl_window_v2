# `pneu_rl/models` 모델 분석 및 reward 시각화

분석 대상은 `pneu_rl/models` 아래의 15개 모델이다. 각 모델의 설정은 `cfg.yaml`을 기준으로 분류했고, 학습 곡선과 성능 평가는 `train_diagnostics.csv`의 episode별 기록을 우선 사용했다. `Ours000_v09`처럼 진단 CSV가 없는 모델은 `infos.pkl`의 reward만 별도로 시각화했다.

## 결과 바로가기

- [모델별 카드](models/README.md)
- [모델 비교 그림](../img/model_diagnostics/model_comparison.png)
- [PID/CAPS/MPOBS feature 비교](../img/model_diagnostics/feature_comparison.png)
- [재현 가능한 시각화 스크립트](../pneu_rl/viz/model_diagnostics.py)

진단 CSV가 있는 모델의 상세 그림은 `img/model_diagnostics/model_<model_name>.png`에 있다. 첫 패널은 reward이고, 나머지는 tracking RMSE, bias, valve conflict, chamber pressure, action delta이다. `Ours000_v09`는 진단 CSV가 없어 reward-only 그림으로 저장했다. `0627ㄱㄷ`, `0703_retrain1`, `0706`, `Ours000_v09`는 1000 episode에 빨간 점선을 표시했다.

## 분류 기준

이 저장소에는 별도의 `success` boolean이 없으므로 다음을 분석용 휴리스틱으로 사용했다. 마지막 100 episode의 양/음 actuator RMSE가 모두 5 kPa 미만이면 `success`, 둘 다 7 kPa 미만이면 `partial`, 그 외는 `failure`로 표시했다. 이 라벨은 학습 결과를 빠르게 읽기 위한 기준이며, 실제 task-level 성공 판정의 대체값은 아니다.

retrain 모델은 1000 episode 이전을 pre-retrain, 이후 마지막 100 episode를 post-retrain으로 따로 계산했다. retrain 전후에는 reward 항과 실행 환경이 바뀔 수 있으므로 reward의 절대값만으로 성공 여부를 판단하지 않고 RMSE, bias, conflict를 함께 봐야 한다.

## 용어 해석

| 항목 | 판정 방법 | 의미 |
|---|---|---|
| PID | `cfg.yaml`의 `pid`가 `null`이 아닌가 | actuator pressure 보정 루프가 환경 안에서 켜져 있는가 |
| CAPS | `temporal_weight` 또는 `spatial_weight`가 0이 아닌가 | `sac_with_loss.py`의 temporal/spatial policy regularization이 적용되었는가 |
| MPOBS | `pred`가 있고 `num_prev > 0`, `num_pred > 1`인가 | 과거 관측과 미래 pressure prediction을 policy 입력에 사용했는가 |
| conflict | `pos_conflict_mean`, `neg_conflict_mean` | 같은 방향의 in/out valve가 동시에 열리는 정도. 낮을수록 해석 가능한 valve allocation에 가깝다 |
| reward | episode reward | 높을수록 좋지만 reward 항이 바뀐 실험끼리는 절대 비교에 주의해야 한다 |

CAPS와 MPOBS는 서로 다른 축이다. MPOBS는 시간 문맥/예측 입력이고, CAPS는 policy loss regularization이다. 따라서 `Ours` 계열은 보통 둘 다 켜져 있고, `SAC + MPObs`는 MPOBS만, `SAC + CAPS`는 CAPS만 켠 비교군이다.

## 전체 비교

아래 수치는 일반 모델은 마지막 100 episode, retrain 모델은 post-retrain 마지막 100 episode의 평균이다. RMSE와 bias의 단위는 kPa다.

| 모델 | PID | CAPS | MPOBS | 판정 | reward | pos RMSE | neg RMSE | pos bias | neg bias |
|---|:---:|:---:|:---:|---|---:|---:|---:|---:|---:|
| `0624_lib4_Ours` | ✓ | ✓ | ✓ | success | -17,066 | 4.56 | 2.71 | 3.07 | 0.99 |
| `0624_lib4_sac` |  |  |  | failure | -17,240 | 8.98 | 5.91 | 7.20 | -4.44 |
| `0624_sac_mpobs` |  |  | ✓ | failure | -40,437 | 12.32 | 4.44 | 11.08 | -2.75 |
| `0625_lib4_Ours` | ✓ | ✓ | ✓ | success | -20,982 | 4.51 | 4.43 | 2.87 | -0.23 |
| `0626_lib4_Ours` | ✓ | ✓ | ✓ | success | -15,968 | 4.30 | 2.59 | 2.47 | 0.61 |
| `0626_lib4_Ours_no_vel_action_penalty` | ✓ | ✓ | ✓ | success | -18,328 | 4.29 | 2.55 | 3.25 | 0.05 |
| `0626_lib4_Ours_no_vel_action_penalty_high` | ✓ | ✓ | ✓ | success | -16,351 | 3.44 | 2.62 | 2.38 | 0.13 |
| `0627ㄱㄷ` | ✓ | ✓ | ✓ | partial | -50,062 | 6.36 | 4.64 | 3.14 | 2.87 |
| `062lib4_Ours_all` | ✓ | ✓ | ✓ | failure | -51,403 | 16.03 | 2.41 | 15.37 | -1.19 |
| `062lib4_Ours_no_vel` | ✓ | ✓ | ✓ | success | -15,258 | 3.39 | 2.42 | 2.20 | 0.23 |
| `062lib4_sac_caps` |  | ✓ |  | failure | -31,958 | 12.92 | 9.97 | 12.04 | -9.77 |
| `0703_retrain1` | ✓ | ✓ | ✓ | failure | -38,283 | 11.73 | 3.71 | 10.94 | 2.47 |
| `0706` | ✓ | ✓ | ✓ | failure | -9,149 | 10.59 | 4.63 | 9.51 | 2.80 |
| `0707` |  | ✓ | ✓ | failure | -88,359 | 14.87 | 17.28 | 14.55 | -17.06 |
| `Ours000_v09` | ✓ | ✓ | ✓ | reward-only | -2,353 | — | — | — | — |

## PID 유무에 따른 관찰

저장된 비교군에서는 PID가 켜진 모델들이 대체로 양/음 actuator RMSE를 5 kPa 근처까지 낮췄고, PID가 없는 네 모델(`0624_lib4_sac`, `0624_sac_mpobs`, `062lib4_sac_caps`, `0707`)은 모두 positive bias 또는 양쪽 tracking 실패가 남았다. 특히 `0624_sac_mpobs`는 negative RMSE는 4.44 kPa까지 낮췄지만 positive bias가 11.08 kPa였고, `0707`은 양/음 bias가 각각 14.55/-17.06 kPa였다.

다만 PID 효과와 CAPS/MPOBS 효과가 완전히 분리된 factorial 실험은 아니다. PID 유무와 함께 action dimension, prediction, reference 범위, reward, horizon도 바뀌므로 “PID가 원인이다”라고 단정하기보다 현재 저장된 configuration에서 PID-assisted formulation이 더 안정적이었다고 해석해야 한다.

## CAPS와 MPOBS 관찰

- `062lib4_sac_caps`는 CAPS만 적용하고 PID/MPOBS를 사용하지 않았다. 마지막 100 episode에서 pos/neg RMSE가 12.92/9.97 kPa로 남아 CAPS regularization만으로 tracking 문제가 해결되지는 않았다.
- `0624_sac_mpobs`는 MPOBS만 적용했다. negative 방향은 어느 정도 개선됐지만 positive pressure에 큰 bias가 남았다. temporal information이 있어도 low-level valve allocation과 chamber regulation 문제를 단독으로 해결하지 못한 결과로 읽힌다.
- `Ours` 계열은 PID + MPOBS + CAPS를 함께 사용한다. 성공 모델들은 보통 pos RMSE 3.39–4.56 kPa, neg RMSE 2.42–4.43 kPa 범위에 들어왔다.
- `062lib4_Ours_all`은 feature 조합은 성공한 `Ours`와 거의 같지만 positive RMSE가 16.03 kPa로 실패했다. 이 차이는 설정 feature보다 seed/optimization trajectory 또는 학습 변동성의 영향으로 보는 것이 안전하다.

## 성공 모델과 실패 모델의 원인

### 성공 패턴

`062lib4_Ours_no_vel`, `0626_lib4_Ours_no_vel_action_penalty_high`, `0626_lib4_Ours`, `0624_lib4_Ours`가 대표적인 성공 패턴이다. 공통적으로 PID가 actuator pressure의 residual error를 보정하고, MPOBS가 delay가 있는 plant의 temporal context를 제공하며, CAPS가 policy output을 regularize한다. 결과적으로 bias가 작고, chamber end pressure도 대체로 positive 135–154 kPa, negative 18–27 kPa 범위에 머문다.

단, valve conflict는 성공 모델에서도 대략 0.84–0.88 수준이다. 즉 tracking은 성공했지만 valve in/out allocation은 아직 동시에 열리는 동작이 많아, “성공 = 깨끗한 제어”는 아니다.

### 실패 패턴

- PID 없는 plain SAC는 관측 history와 pressure prediction이 없고 1D action baseline이라 delay/coupling에 취약하다. positive bias가 커진다.
- MPOBS만 사용한 모델은 관측/예측은 늘었지만 regularization과 PID가 없어 positive chamber/actuator 방향의 drift가 남는다.
- CAPS만 사용한 모델은 action smoothness 쪽 inductive bias는 얻지만, low-level state/action representation이 부족해 tracking 자체가 수렴하지 않았다.
- `0707`은 PID가 없고 action-delta penalty가 0.5로 강하다. smoothness를 강하게 요구하는 것이 tracking보다 앞서면서 양쪽 pressure bias가 모두 커진 실패 사례로 볼 수 있다.
- `062lib4_Ours_all`은 같은 feature class에서도 실패할 수 있음을 보여준다. 이 모델은 feature ablation의 반례라기보다 학습 안정성/seed 분산을 확인하는 사례다.
- `Ours000_v09`는 `infos.pkl`만 있고 `train_diagnostics.csv`가 없어 RMSE/bias/conflict를 계산할 수 없다. post-retrain reward 마지막 100 episode 평균은 `-2,353`이지만, 이는 reward-only 결과이므로 tracking success로 분류하지 않았다.

## Retrain 전후

세 retrain 모델 모두 `parent_model: 0626_lib4_Ours_no_vel_action_penalty`이고 1000 episode 경계에 retrain이 시작된다. 그림에서 빨간 점선 오른쪽이 post-retrain이다.

| 모델 | pre-retrain pos/neg RMSE | post-retrain pos/neg RMSE | 관찰 |
|---|---:|---:|---|
| `0627ㄱㄷ` | 4.29 / 2.55 | 6.36 / 4.64 | reward가 -18,328에서 -50,062로 악화되고 conflict도 증가 |
| `0703_retrain1` | 4.29 / 2.55 | 11.73 / 3.71 | negative는 유지됐지만 positive bias가 10.94 kPa로 커짐 |
| `0706` | 4.29 / 2.55 | 10.59 / 4.63 | reward는 -9,149로 좋아 보이지만 tracking RMSE와 bias는 악화 |

세 결과 모두 retrain_cfg에 `type: real`이 기록되어 있다. 따라서 post-retrain reward는 pre-retrain simulation reward와 동일한 측정 조건으로 보기 어렵다. 특히 `0706`은 reward만 보면 개선처럼 보이지만 pos RMSE가 4.29→10.59 kPa로 악화되어, retrain 평가에서는 reward 단독 결론을 피해야 한다.

## 산출물

```text
img/model_diagnostics/
├── model_<model>.png        # 모델별 6-panel plot
├── model_comparison.png     # 전체 모델 final-phase 비교
├── feature_comparison.png   # PID/CAPS 기준 RMSE scatter
└── model_summary.csv        # plot에 사용한 수치
```
