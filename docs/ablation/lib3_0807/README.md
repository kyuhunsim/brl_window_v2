# lib3 강화학습 절제 실험 기록

> LaTeX 수식을 지원하지 않는 Markdown viewer에서는
> [plain 수식 요약](../../formula_reference_plain.md)을 참고한다.

> Observation 구성의 물리적 의미, 교수님 PINN 피드백과 predictor 근거는 [lib3 observation의 물리적·예측적 근거 검증](observation_predictive_evidence.md)에, History/rate scale confound와 5-step history compression을 검사하는 신규 모델은 [lib3 observation coordinate controls](observation_coordinate_control_experiment.md)에 정리했다.

> 최신 `e_dot`/PIDE/GPIDE-lite 1,000-episode screen의 동일 40.96 s
> reference 추종, six-valve command 그림, 결론 경계와 다음 실험 순서는
> [structured-history screen 결과](structured_history_screen_results.md)에 정리했다.

이 문서는 lib3 공압 전용 6밸브 시뮬레이터에서 수행한 실험을 시간순으로 정리한 것이다. 기존 기록에는 여러 중간 기준 궤적이 섞여 있었으므로, 여기서는 다음 두 기준 궤적을 사용한 모델만 포함한다.

1. 최신 도달가능성 확인 기준 궤적
   - `pneu_rl/config/reference/lib3_capacity_random_medium_16x4x512_seed61098.yaml`
   - 시뮬레이터의 압력 궤적 묶음에서 생성된 기준 궤적
   - 압력 범위, 변화율, chamber reserve를 사전에 확인한 최신 기준
2. 예전 무작위 기준 궤적
   - 매번 offset/amplitude/period를 즉시 바꾸는 예전 `RandomRef`
   - parameter jump 때문에 높은 변화율이 발생하는 강건성 시험용 기준

`rate_limited_random`, `constant`, 중간 점검용 실험은 모델 발전을 이해하는 데 필요한 경우에만 설명하고, 전체 모델 표에서는 제외했다. 최종적으로 포함한 모델은 총 47개다.

## 공통 실험 조건

- lib3 공압 전용 6밸브 모델
- 시작 상태: 양압·음압 actuator와 chamber 모두 약 `101.325 kPa`
- 실제 제어 명령: `u = 0.8 + 0.2a`
- 제어 주파수: 50 Hz
- 시뮬레이터 지연: 0.1 s
- PID: 학습 및 검증에서 사용하지 않음
- 외부 탐색 noise: `0`
- 기본 action history: 10 step
- 기본 horizon: 512 step
- 기본 carry: 4 block마다 상태 유지
- gamma: 0.99
- 검증: 1 episode, 2048 실제 시뮬레이터 step
- K=2 검증: 1024 policy step을 실행해 2048 실제 시뮬레이터 step으로 맞춤

평가 지표는 RMSE만 사용하지 않았다.

- 압력 RMSE/MAE
- 허용오차 안에 머문 비율
- 기준 궤적과 실제 궤적의 방향 일치율 및 상관
- 기준 변화율과 실제 변화율
- 압력 궤적의 최적 지연
- chamber reserve 및 chamber 과충전
- 실제 제어 명령의 `mean |Δu|`, `mean |Δ²u|`
- episode reward
- 압력 궤적과 제어 명령 궤적의 시각적 개형

모든 검증 결과는 [lib3_0809_ablation_validation_2048](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048)에 저장되어 있다.

## 전체 연구 흐름: 무엇을 추가했고 무엇이 달라졌는가

이번 lib3 실험의 흐름은 단순히 모델을 많이 바꿔 본 것이 아니라, 각 실패 양상에 맞춰 관측값과 시간 처리 방식을 하나씩 추가한 과정이다.

### 1. 처음의 직접형 SAC: 현재 압력과 기준 궤적만 입력

가장 단순한 직접형 Pure SAC는 현재의 네 압력

```text
P_ch,pos, P_ch,neg, P_act,pos, P_act,neg
```

과 현재·다음 기준 압력만 관측한다. 이 구조는 관측 길이가 8이고, policy는 6개 밸브 명령을 한 번에 출력한다. 즉 밸브 내부의 `z`, 유량, chamber reserve, 이전 명령, 지연 전의 압력 상태를 직접 알 수 없다.

이 구조에서 나타난 문제는 다음과 같다.

- 현재 압력 오차는 알지만, 지금 올린 밸브 명령의 효과가 몇 step 뒤에 나타날지 알기 어렵다.
- 같은 현재 압력이라도 내부 밸브 상태와 chamber 압력에 따라 다음 압력 변화가 달라진다.
- 여섯 밸브의 명령이 동시에 바뀌므로 action switching과 압력 진동이 발생한다.

따라서 직접형 SAC의 실패는 단순히 탐색 noise가 부족해서라기보다, 현재 관측만으로는 lib3의 상태가 Markov 상태가 되지 않는 데서 시작했다고 해석했다.

<details>
<summary><code>260806_lib3_capref_512_pure_sac_seed0</code> 근거 결과</summary>

- 기준 궤적: 최신 기준
- 관측 길이: 8
- policy 출력: 6개 밸브 명령
- POS/NEG RMSE: `19.66 / 8.99 kPa`
- POS/NEG 방향 일치율: `.52 / .54`
- 평균/최대 제어 명령 변화량: `.0113 / .1519`

![직접형 Pure SAC 검증 그림](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_pure_sac_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_pure_sac_seed0/episode_01.csv)

이 결과는 큰 압력 오차와 낮은 방향 일치율, 큰 제어 명령 변화가 함께 나타난 직접형 SAC의 대표 사례다. 현재 압력과 기준 압력만으로는 밸브의 지연 효과와 chamber reserve를 분리하지 못했다는 해석을 뒷받침한다.

</details>

### 2. 압력 history 추가: 지연을 과거 압력으로 추정

다음 단계에서는 최근 10 step 압력과 15 step의 기준 궤적을 넣었다. 압력 history만 사용하는 구조의 관측 길이는 96이다.

여기서 기준 궤적의 종류는 여전히 2개뿐이다.

```text
reference = [P_act,pos_ref, P_act,neg_ref]
```

관측 길이에서 reference 부분이 52가 되는 이유는 2개 reference를 26개 시간 지점에 펼쳤기 때문이다.

```text
과거 10 step + 현재 1 step + 미래 15 step = 26개 시간 지점
26개 시간 지점 × 2개 actuator reference = 52차원
```

즉 52는 chamber reference 52개가 아니다. chamber pressure는 reference가 아니라 시뮬레이터에서 관측되는 `P_ch,pos`, `P_ch,neg` 상태에 포함된다. 현재 설정의 `chamber_controller`도 `none`이므로 chamber reference를 policy에 별도로 제공하지 않는다.

```text
최근 압력 11개 × 4 = 44
현재·과거·미래 actuator 기준 압력 26개 × 2 = 52
전체 = 96
```

이 변경으로 policy는 압력이 상승 중인지 하강 중인지, 기준이 곧 바뀌는지, 최근 변화가 일시적인지 지속적인지를 볼 수 있게 됐다. 큰 압력 개형은 이전보다 잘 따라갔지만, 밸브 명령이 실제로 어떤 방향의 유량을 만들었는지는 여전히 간접적으로만 추정했다.

<details>
<summary><code>260806_lib3_capref_512_mpobs1_caps_seed0</code> 근거 결과</summary>

- 기준 궤적: 최신 기준
- 관측 구성: 압력 history 96 + 1-step predictor 4 = 100차원
- policy 출력: 6개 밸브 명령
- POS/NEG RMSE: `5.44 / 4.40 kPa`
- POS/NEG 방향 일치율: `.51 / .54`
- 평균/최대 제어 명령 변화량: `.0027 / .0868`

![압력 history 계열 MPOBS1+CAPS 검증 그림](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_mpobs1_caps_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_mpobs1_caps_seed0/episode_01.csv)

압력 history를 사용해 큰 개형은 어느 정도 따라갔지만 방향 일치율이 낮고, 최신 기준에서 양압·음압의 세부 변화는 충분히 따라가지 못했다. 이 모델은 순수한 96차원 압력 history 절제가 아니라 predictor와 CAPS가 함께 포함된 100차원 변형이라는 점에 주의해야 한다.

</details>

<details>
<summary><code>260806_lib3_capref_1024_mpobs1_caps_seed1</code> 근거 결과</summary>

- 기준 궤적: 최신 기준
- 관측 구성: 압력 history 96 + 1-step predictor 4 = 100차원
- policy 출력: 6개 밸브 명령
- POS/NEG RMSE: `7.00 / 8.36 kPa`
- POS/NEG 방향 일치율: `.51 / .53`
- 평균/최대 제어 명령 변화량: `.0061 / .0688`

![압력 history 계열 MPOBS1+CAPS 1024 검증 그림](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_1024_mpobs1_caps_seed1/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_1024_mpobs1_caps_seed1/episode_01.csv)

같은 압력 history 계열에서도 학습 조건과 seed에 따라 결과가 달라졌다. 따라서 압력 history는 필요한 시간 정보를 보완하지만, 그것만으로 밸브 방향성과 chamber reserve 문제를 해결한다고 보기는 어렵다.

</details>

### 3. 이전 밸브 명령 history 추가: 지연과 숨은 밸브 상태를 관측

압력 history만으로는 동일한 압력에서 명령의 과거가 다른 경우를 구별하기 어렵기 때문에 최근 10개의 6밸브 명령을 추가했다. 이때 관측 길이는 156이다.

```text
압력 history = 44
actuator 기준 궤적 = 52
이전 6밸브 명령 10개 = 60
전체 = 156
```

History SAC는 직접형 SAC보다 안정적으로 큰 개형을 따라갔고, 특히 예전 무작위 기준에서는 POS/NEG RMSE `3.997/3.893 kPa`를 기록했다. 이것은 이전 명령이 밸브 지연과 hysteresis를 간접적으로 복원하는 데 실제로 도움이 된다는 증거다.

다만 history는 “무슨 명령을 냈는가”를 알려줄 뿐 “그 명령이 현재 chamber reserve에서 어떤 효과를 낼 것인가”를 직접 알려주지 않는다. 그래서 offset과 방향성 실패가 남았다.

<details>
<summary><code>260808_lib3_clean_profile_history_sac_seed0</code> 근거 결과</summary>

- 기준 궤적: 최신 기준
- 관측 구성: 압력 history 44 + 기준 궤적 52 + 이전 밸브 명령 60 = 156차원
- policy 출력: 6개 밸브 명령
- POS/NEG RMSE: `4.94 / 7.50 kPa`
- POS/NEG 방향 일치율: `.69 / .60`
- 평균/최대 제어 명령 변화량: `.0024 / .0556`

![이전 밸브 명령 history SAC 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_sac_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_sac_seed0/episode_01.csv)

직접형 SAC보다 큰 개형은 안정적으로 따라가지만, 음압 offset과 방향성 실패가 남는다. 이전 명령 history는 지연을 보완하지만 chamber reserve와 실제 압력 변화율을 직접 알려주지는 않는다.

</details>

### 4. CAPS와 MPOBS 추가: 진동 억제와 미래 효과 예측 시도

선행 연구의 핵심 구성인 CAPS와 MPOBS를 lib3에 추가했다.

- CAPS: 인접한 policy 출력이 급격히 달라지지 않도록 제어 명령 변화에 penalty를 적용
- MPOBS: predictor가 짧은 미래 압력 변화를 policy observation에 제공

이론적으로는 지연 시스템에서 합리적인 추가다. 그러나 lib3에서는 CAPS가 명령 변화 자체를 줄이는 대신 필요한 chamber 충전·방전 변화까지 억제할 수 있었다. MPOBS도 현재 구현에서는 plant 내부 상태를 그대로 이어받는 완전한 predictor가 아니며, predictor 지연과 plant 지연이 다르다.

최신 기준 궤적에서 History SAC의 POS/NEG RMSE는 `4.937/7.497`, History SAC+CAPS는 `8.989/5.449`, History SAC+MPOBS1+CAPS는 `13.425/5.739 kPa`였다. 따라서 lib3에서는 “CAPS/MPOBS가 항상 더 좋다”가 아니라, 물리 관측값과 reward 연결이 맞지 않으면 진동을 줄이는 대신 tracking과 chamber reserve를 망칠 수 있다는 결론을 얻었다.

<details>
<summary><code>260808_lib3_clean_profile_history_caps_seed0</code> 근거 결과</summary>

- 기준 궤적: 최신 기준
- 관측 구성: 이전 밸브 명령 history 156차원
- 추가 구성: CAPS
- POS/NEG RMSE: `8.99 / 5.45 kPa`
- POS/NEG 방향 일치율: `.54 / .52`
- 평균/최대 제어 명령 변화량: `.0075 / .0801`

![History SAC+CAPS 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_caps_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_caps_seed0/episode_01.csv)

CAPS를 추가했지만 최신 기준에서 양압 RMSE와 방향 일치율이 History SAC보다 나빠졌다. 명령 변화 억제가 필요한 밸브 변화까지 제한했을 가능성이 있다.

</details>

<details>
<summary><code>260808_lib3_clean_profile_history_mpobs1_caps_seed0</code> 근거 결과</summary>

- 기준 궤적: 최신 기준
- 관측 구성: 이전 밸브 명령 history 156 + MPOBS1 predictor 4 = 160차원
- 추가 구성: MPOBS1 + CAPS
- POS/NEG RMSE: `13.42 / 5.74 kPa`
- POS/NEG 방향 일치율: `.55 / .52`
- 평균/최대 제어 명령 변화량: `.0049 / .0756`

![History SAC+MPOBS1+CAPS 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_mpobs1_caps_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_mpobs1_caps_seed0/episode_01.csv)

MPOBS1을 더했음에도 양압 chamber가 과충전되고 양압 추적이 크게 악화됐다. 현재 predictor가 lib3의 비선형 지연 상태를 충분히 설명하지 못할 가능성을 보여준다.

</details>

### 5. 물리 관측값 추가: policy에게 압력 변화의 방향과 여유를 직접 제공

물리 관측값은 다음 18개다.

```text
압력 변화율 4개
chamber-actuator reserve 2개
현재 실제 밸브 명령 6개
이전 실제 밸브 명령 6개
전체 = 18
```

따라서 기존 History SAC의 156차원 관측에 물리 관측값을 추가하면 174차원이 된다. 이 정보는 `z`, `q_pred`, `mdot`를 그대로 policy에 넣은 것이 아니라, 현재 simulator에서 측정 가능한 압력 변화율·reserve·실제 명령을 명시적으로 넣은 것이다.

다만 기존 History SAC가 이미 최근 10개의 6밸브 명령을 관측하고 있으므로, 18개가 전부 새로운 정보인 것은 아니다.

```text
실질적으로 새로 추가되는 정보:
압력 변화율 4개 + chamber-actuator reserve 2개 = 6개

기존 action history와 의미가 겹치는 정보:
현재 실제 명령 6개 + 이전 실제 명령 6개 = 12개
```

차이는 표현 방식에도 있다. action history는 최근 10개의 정규화된 policy 명령을 길게 제공하고, physics feature의 현재·이전 명령은 시뮬레이터에 실제로 적용된 physical command `u = 0.8 + 0.2a`로 변환해 가장 최근 두 명령만 제공한다. 따라서 관측 차원은 156에서 174로 늘어나지만, History SAC 기준으로 physics observation의 핵심 효과를 “새로운 압력 변화율 4개와 reserve 2개를 추가한 것”으로 해석하는 것이 더 정확하다.

이 변경 후 policy는 단순히 “오차가 양수인가”만 보는 것이 아니라 다음을 구별할 수 있다.

- chamber가 actuator보다 충분히 높은가
- 현재 압력이 실제로 상승 중인가 하강 중인가
- 이전 명령이 아직 효과를 내고 있는가
- 지금 actuator 밸브를 더 열면 reserve가 고갈되는가

가장 중요한 결과는 최신 기준 궤적의 `K=2 + 물리 관측값`이었다. POS/NEG RMSE `1.208/.981 kPa`, 방향 일치율 `.998/.996`, 평균 `|Δu|` `.0020`을 기록했고 POS chamber 최대값도 `143.0 kPa`로 유지됐다. 따라서 현재까지는 물리 관측값이 lib3의 실패 원인을 가장 직접적으로 완화했다.

<details>
<summary><code>260807_lib3_physics_obs_seed0</code> 근거 결과</summary>

- 기준 궤적: 최신 기준
- 관측 구성: 이전 밸브 명령 history 156 + 물리 관측값 18 = 174차원
- policy 출력: 6개 밸브 명령
- POS/NEG RMSE: `2.86 / 2.30 kPa`
- POS/NEG 방향 일치율: `.71 / .59`
- 평균/최대 제어 명령 변화량: `.0086 / .1492`

![K=1 물리 관측값 SAC 검증 그림](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_physics_obs_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_physics_obs_seed0/episode_01.csv)

K=1에서도 물리 관측값을 추가하면 직접형·history-only 구조보다 압력 개형과 chamber reserve를 더 잘 관리한다. 이 결과는 K=2가 아니라 물리 관측값 자체가 주요 개선 원인일 수 있다는 근거다.

</details>

### 6. 적분 오차 추가: 지속적인 정상상태 offset 보정

적분 오차는 actuator 양압·음압 두 채널에 대해 누적된 tracking error를 2개 추가한다. 그러므로 물리 관측값과 적분 오차를 함께 사용하면 관측 길이는 176이다.

적분항의 장점은 일정한 offset을 계속 보정할 수 있다는 점이다. 반면 chamber reserve가 부족한데도 오차를 계속 누적하면 wind-up이 발생할 수 있고, policy가 그 오차를 줄이기 위해 명령을 과도하게 흔들 수 있다. 실제로 `K=2 + 물리 관측값 + 적분항`은 seed 1에서 양압 RMSE `.946 kPa`를 기록했지만 seed 0 물리 관측값 모델보다 명령의 매끄러움과 일부 방향 지표가 낮았다.

따라서 적분항은 무조건 기본 구성으로 넣기보다 정상상태 offset이 반복될 때 선택적으로 사용하는 보정항으로 보는 것이 타당하다.

<details>
<summary><code>260807_lib3_integral_error_seed0</code> 근거 결과</summary>

- 기준 궤적: 최신 기준
- 관측 구성: 이전 밸브 명령 history 156 + actuator 적분 오차 2 = 158차원
- policy 출력: 6개 밸브 명령
- POS/NEG RMSE: `7.45 / 4.18 kPa`
- POS/NEG 방향 일치율: `.78 / .55`
- 평균/최대 제어 명령 변화량: `.0050 / .1032`

![K=1 적분 오차 SAC 검증 그림](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_error_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_error_seed0/episode_01.csv)

적분항은 음압 RMSE와 양압 방향성 일부를 개선했지만, 양압 offset을 해결하지 못했고 전체 성능은 history-only보다 일관되게 좋지 않았다. 적분항은 물리 관측값 없이 단독으로 넣기보다 reserve와 함께 해석해야 한다.

</details>

### 7. K=2 action chunking: observation이 아니라 policy 결정 간격을 변경

K=2는 관측값을 추가하는 기능이 아니다. 관측 길이는 K=1과 동일하고, policy의 출력만 다음처럼 바뀐다.

```text
K=1: [a0의 밸브 6개]       → action 차원 6
K=2: [a0의 밸브 6개, a1의 밸브 6개] → action 차원 12
```

시뮬레이터에는 `a0`를 먼저 적용하고 바로 다음 실제 20 ms step에 `a1`을 적용한다. action hold가 아니며, 실제 밸브에는 두 개의 서로 다른 physical command가 순서대로 들어간다. 환경이 반환하는 하나의 chunk reward는

\[
R = r_0 + 0.99 r_1
\]

이고, SAC critic의 다음 상태 할인도 `gamma^2`를 사용한다. 따라서 K=2는 policy 호출을 50 Hz에서 25 Hz로 줄이는 대신 simulator의 physical control은 계속 50 Hz로 유지하는 구조다.

K=2의 장점은 다음과 같다.

- policy가 매 20 ms마다 한 번만 보고 크게 반응하는 현상을 줄인다.
- 짧은 미래 명령 두 개를 함께 결정하므로 지연 시스템에 더 긴 시간 문맥을 제공한다.
- action switching과 pressure vibration이 줄어들 가능성이 있다.

단점도 분명하다.

- 출력 차원이 6에서 12로 늘어나 critic이 학습해야 할 action 공간이 커진다.
- 두 번째 명령 `a1`은 현재 상태에서 실행 결과를 보지 않고 미리 정한다.
- 첫 번째 명령이 예상과 다르게 작동해도 두 번째 명령은 이미 정해져 있다.
- reward가 두 physical step의 합으로 묶여 credit assignment가 더 거칠어질 수 있다.

따라서 K=2가 항상 좋은 것은 아니다. 현재 결과에서도 K=1 + 물리 관측값 + 적분항은 양압 RMSE `1.756 kPa`로 경쟁력 있는 결과를 보였고, 예전 무작위 기준의 K=1 + 물리 관측값도 큰 개형을 잘 따라갔다. K=2의 우위는 모든 지표의 절대 우위라기보다, 최신 기준에서 `K=2 + 물리 관측값`이 압력 오차·방향성·chamber reserve·명령 매끄러움을 동시에 만족한 데 있다. 앞으로는 K=1과 K=2를 같은 seed와 같은 기준 궤적에서 반드시 쌍으로 비교해야 한다.

<details>
<summary><code>260808_lib3_ablation_profile_k2_baseline_seed0</code> 근거 결과</summary>

- 기준 궤적: 최신 기준
- 관측 구성: 이전 밸브 명령 history 156차원
- K=2 policy 출력: 12개, 실제 simulator 적용: 2 step × 6개
- POS/NEG RMSE: `2.54 / 1.85 kPa`
- POS/NEG 방향 일치율: `.64 / .78`
- 평균/최대 제어 명령 변화량: `.0072 / .0846`

![K=2 기준선 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_baseline_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_baseline_seed0/episode_01.csv)

K=2만 추가해도 음압 추적은 좋아졌지만 양압 chamber가 `248.9 kPa`까지 누적됐다. 따라서 K=2는 시간 처리 문제를 완화할 수 있어도 chamber reserve를 스스로 알게 해주지는 않는다.

</details>

<details>
<summary><code>260808_lib3_ablation_profile_k2_physics_seed0</code> 근거 결과</summary>

- 기준 궤적: 최신 기준
- 관측 구성: 이전 밸브 명령 history 156 + 물리 관측값 18 = 174차원
- K=2 policy 출력: 12개, 실제 simulator 적용: 2 step × 6개
- POS/NEG RMSE: `1.21 / .98 kPa`
- POS/NEG 방향 일치율: `.998 / .996`
- 평균/최대 제어 명령 변화량: `.0020 / .1792`

![K=2 물리 관측값 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_seed0/episode_01.csv)

K=2와 물리 관측값을 함께 사용했을 때 pressure tracking, 방향성, chamber reserve가 동시에 가장 안정적이었다. 다만 K=2만의 효과와 물리 관측값의 효과는 아직 완전히 분리되지 않았으므로 K=1 물리 관측값과 동일 조건 비교가 필요하다.

</details>

## 8월 10일: pressure-rate/reserve 분리 절제 실험

앞선 실험에서는 `physics observation`을 18개 전체로 한 번에 추가했다. 그러나 History SAC는 이미 최근 10개의 6밸브 명령을 관측하므로, 물리 관측값의 핵심적인 새 정보는 다음 6개다.

```text
압력 변화율 4개: dP_ch,pos/dt, dP_ch,neg/dt, dP_act,pos/dt, dP_act,neg/dt
chamber-actuator reserve 2개: P_ch,pos-P_act,pos, P_act,neg-P_ch,neg
```

이번 9개는 이 두 정보가 각각 offset과 방향성에 어떤 영향을 주는지 분리하기 위한 절제 실험이다. 기존의 18개 physics feature를 그대로 쓰지 않고, `rate`, `reserve`, `rate_reserve` 모드를 새로 분리했다. 따라서 현재 action history와 겹치는 physical current/previous action 12개는 이번 비교에서 제외한다.

### 실험 구성

| ID | 모델 | 추가 정보/기능 | policy obs | action | 확인 목적 |
|---:|---|---|---:|---:|---|
| 1 | `260810_lib3_ablation_profile_hist_base_seed0` | History SAC baseline | 156 | 6 | physics feature가 없는 기준선 |
| 2 | `260810_lib3_ablation_profile_hist_rate_seed0` | pressure rate 4개 | 160 | 6 | 압력 변화 방향 정보의 효과 |
| 3 | `260810_lib3_ablation_profile_hist_reserve_seed0` | chamber reserve 2개 | 158 | 6 | chamber 여유 정보의 효과 |
| 4 | `260810_lib3_ablation_profile_hist_rate_reserve_seed0` | rate 4개 + reserve 2개 | 162 | 6 | 두 물리 정보의 결합 효과 |
| 5 | `260810_lib3_ablation_profile_hist_rate_reserve_integral_seed0` | 4번 + integral error 2개 | 164 | 6 | 지속적인 steady-state offset 보정 효과 |
| 6 | `260810_lib3_ablation_profile_hist_rate_reserve_k2_seed0` | 4번 + K=2 | 162 | 12 | 물리 정보와 action chunking의 결합 효과 |
| 7 | `260810_lib3_ablation_profile_hist_rate_reserve_caps001_seed0` | 4번 + mean-action CAPS | 162 | 6 | 작은 CAPS의 진동 억제 효과 |
| 8 | `260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_seed0` | 4번 + MPOBS1 | 166 | 6 | predictor가 추가 정보를 제공하는지 확인 |
| 9 | `260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_caps001_seed0` | 4번 + MPOBS1 + CAPS | 166 | 6 | 선행 연구 모듈을 물리 기준선 위에 재적용 |

여기서 `K=2`의 policy action은 12차원이지만 simulator에는 6차원 명령 두 개가 20 ms 간격으로 순서대로 적용된다. 나머지 모델은 매 policy step마다 6개 밸브 명령 하나를 출력한다.

### 공통 조건

9개 모두 아래 조건을 고정한다. 이 조건을 통일해야 rate와 reserve 중 어떤 정보가 실제로 기여했는지를 해석할 수 있다.

- 기준 궤적: 최신 도달가능성 확인 profile
  - `pneu_rl/config/reference/lib3_capacity_random_medium_16x4x512_seed61098.yaml`
- `ref_seed=61098`, train seed `0`
- `horizon=512`, `carry`, `carry_reset_blocks=4`
- 총 `1000 episode`, `start_epi=10`, `gamma=0.99`
- 실제 command scale: `u=0.8+0.2a`
- action history: 최근 10 step
- observation normalization: 사용
- buffer size: `500000`
- 외부 exploration noise: 없음
- 초기 warm-up: physical command `0.95` 고정
- warm-up 중 update 보류 후 reset하고 policy 학습 시작
- tracking loss: 기존 absolute error
- rate/action-change/conflict/reserve reward penalty: 모두 `0`
- CAPS 실험: `mean_action`, temporal `0.01`, spatial `0.01`, replay noise `0`
- 실행 thread 제한: `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`

이번 세트의 핵심 비교는 다음 순서다.

```text
1 → 2, 3 → 4       rate와 reserve의 단독/결합 효과
4 → 5               integral error의 steady-state offset 효과
4 → 6               K=2의 decision-frequency 효과
4 → 7               작은 CAPS의 추가 효과
4 → 8 → 9           MPOBS1과 CAPS의 추가 효과
```

이 9개는 2026-08-10에 학습을 완료했고, 2026-08-13에 모두 동일한 2048-step 단일 episode로 재검증했다. 결과 그림과 summary는 [lib3_0813_ablation_validation_2048](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048)에 저장되어 있다. 아래 결과는 RMSE뿐 아니라 방향 일치율, reference/actual rate, chamber reserve, 실제 command 변화량, steady-state offset을 함께 사용해 해석한다.

### 실행 스크립트

한 터미널에서 세 모델이 순서대로 실행된다. 세 터미널에서 그룹 1, 2, 3을 각각 실행하면 총 9개가 병렬 그룹으로 진행된다.

- [9개 rate/reserve ablation 실행 스크립트](run_9_rate_reserve_ablation_groups.sh)

```bash
# 터미널 1
cd /home/rise/test/brl_window_v2
bash docs/ablation/lib3_0807/run_9_rate_reserve_ablation_groups.sh 1

# 터미널 2
cd /home/rise/test/brl_window_v2
bash docs/ablation/lib3_0807/run_9_rate_reserve_ablation_groups.sh 2

# 터미널 3
cd /home/rise/test/brl_window_v2
bash docs/ablation/lib3_0807/run_9_rate_reserve_ablation_groups.sh 3
```

이 실험은 기존 기록에 추가된 9개 완료 모델이며, 각 모델별 검증 그림과 CSV는 아래 결과 절에 연결했다.

## 2026-08-13: 8월 10일 rate/reserve ablation 검증 결과

8월 10일에 완료한 9개를 2026-08-13에 다시 검증했다. 모든 모델은 동일한 최신 도달가능성 확인 profile, `seed=61098`, 초기 대기압, `u=0.8+0.2a`, PID off, 2048 simulator step 조건으로 평가했다. profile은 16개 simulator trajectory bank를 사용하지만 profile 내부 seed가 같기 때문에 각 모델의 첫 검증 trajectory는 동일하게 선택된다.

2026-08-23부터 lib3 검증 그림은 `*_main.png`와 `*_ctrl.png`로 분리한다. main은 reference 없는 chamber POS/NEG, REF와 비교한 actuator POS, REF와 비교한 actuator NEG의 세 축만 사용하며 displacement는 넣지 않는다. ctrl은 여섯 physical valve command를 별도 축으로 표시한다. 모든 축에는 y축 이름·단위와 `Time [sec]`를 표시하고 기존 색상·선 굵기를 유지한다.

| 모델 | 추가 구성 | obs | POS/NEG RMSE | POS/NEG 방향 | ±5 kPa 비율 | mean |Δu| | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| `260810_lib3_ablation_profile_hist_base_seed0` | History SAC | 156 | 4.57 / 6.00 | .565 / .562 | .705 / .536 | .0074 | 기준선. 두 압력 모두 진동하며 음압 offset이 남음 |
| `260810_lib3_ablation_profile_hist_rate_seed0` | rate 4 | 160 | 4.89 / 3.69 | .908 / .675 | .501 / .787 | .0053 | 방향 정보는 늘었지만 양압 offset이 큼 |
| `260810_lib3_ablation_profile_hist_reserve_seed0` | reserve 2 | 158 | 9.10 / 2.89 | .581 / .627 | .270 / .941 | .0055 | reserve만으로는 양압 제어 방향이 해결되지 않음 |
| `260810_lib3_ablation_profile_hist_rate_reserve_seed0` | rate + reserve | 162 | 1.38 / 1.25 | .815 / .851 | 1.000 / 1.000 | .0039 | 두 물리 정보의 결합이 큰 개선을 만듦 |
| `260810_lib3_ablation_profile_hist_rate_reserve_integral_seed0` | rate + reserve + integral | 164 | 2.20 / 1.45 | .923 / .909 | 1.000 / 1.000 | .0048 | 방향성과 offset은 좋지만 baseline보다 RMSE는 악화 |
| `260810_lib3_ablation_profile_hist_rate_reserve_k2_seed0` | rate + reserve + K=2 | 162, action 12 | **0.54 / 0.81** | **1.000 / .968** | 1.000 / .999 | .0079 | 가장 낮은 RMSE. 다만 command 변화량은 rate+reserve보다 큼 |
| `260810_lib3_ablation_profile_hist_rate_reserve_caps001_seed0` | rate + reserve + CAPS(.01) | 162 | 5.94 / 3.57 | .890 / .798 | .304 / .823 | .0064 | CAPS가 필요한 동작까지 억제해 양압 추적 악화 |
| `260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_seed0` | rate + reserve + MPOBS1 | 166 | 3.20 / 2.84 | .881 / .747 | .978 / .982 | .0021 | command는 가장 부드럽지만 tracking은 중간 수준 |
| `260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_caps001_seed0` | rate + reserve + MPOBS1 + CAPS(.01) | 166 | 0.91 / 2.40 | .836 / .589 | 1.000 / .991 | .0070 | 양압 RMSE는 좋지만 음압 방향성과 reserve가 불안정 |

### 각 모델 검증 그림

<details>
<summary>History SAC baseline</summary>

[압력 PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_base_seed0/episode_01_main.png) · [6-control PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_base_seed0/episode_01_ctrl.png) · [CSV](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_base_seed0/episode_01.csv) · [학습 cfg](../../../pneu_rl/models/260810_lib3_ablation_profile_hist_base_seed0/cfg.yaml)

기준선은 큰 reference 개형을 대략 따라가지만 POS/NEG actual pressure에 고주파 진동이 남고, direction agreement가 `.565/.562`로 낮다.

</details>

<details>
<summary>Pressure-rate only</summary>

[압력 PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_seed0/episode_01_main.png) · [6-control PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_seed0/episode_01_ctrl.png) · [CSV](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_seed0/episode_01.csv) · [학습 cfg](../../../pneu_rl/models/260810_lib3_ablation_profile_hist_rate_seed0/cfg.yaml)

rate를 넣으면 방향 일치가 양압 `.908`까지 상승하지만, chamber reserve를 직접 보지 않으므로 양압 offset과 tolerance 실패가 남는다.

</details>

<details>
<summary>Chamber-reserve only</summary>

[압력 PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_reserve_seed0/episode_01_main.png) · [6-control PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_reserve_seed0/episode_01_ctrl.png) · [CSV](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_reserve_seed0/episode_01.csv) · [학습 cfg](../../../pneu_rl/models/260810_lib3_ablation_profile_hist_reserve_seed0/cfg.yaml)

reserve만 추가하면 음압은 비교적 안정되지만 POS RMSE가 `9.10 kPa`로 악화된다. “chamber가 얼마나 남았나”만으로는 압력이 실제 상승 중인지 알 수 없다.

</details>

<details>
<summary>Rate + reserve</summary>

[압력 PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_seed0/episode_01_main.png) · [6-control PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_seed0/episode_01_ctrl.png) · [CSV](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_seed0/episode_01.csv) · [학습 cfg](../../../pneu_rl/models/260810_lib3_ablation_profile_hist_rate_reserve_seed0/cfg.yaml)

rate와 reserve를 함께 넣은 모델은 양압·음압 모두 5 kPa tolerance를 100% 만족했고, 평균 command 변화량도 `.0039`로 낮았다. 이번 ablation에서 가장 균형 잡힌 K=1 결과다.

</details>

<details>
<summary>Rate + reserve + integral</summary>

[압력 PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_integral_seed0/episode_01_main.png) · [6-control PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_integral_seed0/episode_01_ctrl.png) · [CSV](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_integral_seed0/episode_01.csv) · [학습 cfg](../../../pneu_rl/models/260810_lib3_ablation_profile_hist_rate_reserve_integral_seed0/cfg.yaml)

integral은 persistent error를 줄이는 데 도움이 되지만, 이 동일 평가에서는 rate+reserve보다 RMSE가 높았다. integral은 기본 필수 관측이라기보다 offset이 확인된 경우의 보정 옵션으로 남긴다.

</details>

<details>
<summary>Rate + reserve + K=2</summary>

[압력 PNG](../../../pneu_rl/exp/lib3_0821_k2_rate_reserve_validation_actual2048/260810_lib3_ablation_profile_hist_rate_reserve_k2_seed0/episode_01_main.png) · [6-control PNG](../../../pneu_rl/exp/lib3_0821_k2_rate_reserve_validation_actual2048/260810_lib3_ablation_profile_hist_rate_reserve_k2_seed0/episode_01_ctrl.png) · [CSV](../../../pneu_rl/exp/lib3_0821_k2_rate_reserve_validation_actual2048/260810_lib3_ablation_profile_hist_rate_reserve_k2_seed0/episode_01.csv) · [학습 cfg](../../../pneu_rl/models/260810_lib3_ablation_profile_hist_rate_reserve_k2_seed0/cfg.yaml)

K=2는 policy가 12개 action을 출력하고 simulator에는 두 개의 6-valve command를 연속 적용한다. 이번 평가에서 POS/NEG RMSE `.54/.81 kPa`, 방향 `.1.000/.968`로 가장 좋았다. 단, K=2 자체가 physics information을 대체한 것은 아니며, rate+reserve가 이미 제공하는 상태 정보와 함께 해석해야 한다.

</details>

<details>
<summary>Rate + reserve + CAPS(.01)</summary>

[검증 PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_caps001_seed0/episode_01.png) · [CSV](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_caps001_seed0/episode_01.csv) · [학습 cfg](../../../pneu_rl/models/260810_lib3_ablation_profile_hist_rate_reserve_caps001_seed0/cfg.yaml)

작은 mean-action CAPS도 이번 조건에서는 개선을 만들지 못했다. 제어 명령이 매끄러워지는 것과 pressure tracking이 좋아지는 것은 분리해서 봐야 한다.

</details>

<details>
<summary>Rate + reserve + MPOBS1</summary>

[검증 PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_seed0/episode_01.png) · [CSV](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_seed0/episode_01.csv) · [학습 cfg](../../../pneu_rl/models/260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_seed0/cfg.yaml)

MPOBS1 단독 모델은 평균 command 변화량 `.0021`로 가장 부드럽지만, K=1 rate+reserve보다 압력 tracking과 direction agreement가 낮다. 부드러운 command가 곧 좋은 control은 아니다.

</details>

<details>
<summary>Rate + reserve + MPOBS1 + CAPS(.01)</summary>

[검증 PNG](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_caps001_seed0/episode_01.png) · [CSV](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_caps001_seed0/episode_01.csv) · [학습 cfg](../../../pneu_rl/models/260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_caps001_seed0/cfg.yaml)

양압은 잘 맞지만 음압 direction agreement가 `.589`이고 lag가 `0.30 s`까지 늘었다. 현재 MPOBS predictor와 CAPS 조합은 lib3에서 기본 구조로 채택할 근거가 부족하다.

</details>

### 학습 reward 수렴과 검증 결과의 관계

[9개 reward/diagnostic plot](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/reward_analysis/reward_convergence_and_post_convergence.png) · [수치 요약 CSV](../../../pneu_rl/exp/lib3_0813_ablation_validation_2048/reward_analysis/reward_convergence_summary.csv)

학습 후반 reward는 모든 완주 모델에서 개선됐지만, reward가 좋아지는 것과 검증 trajectory의 방향성이 항상 일치하지는 않았다. 예를 들어 `rate_reserve`는 tail reward 평균이 약 `-364`이고 검증 RMSE가 `1.38/1.25`였으며, `K=2`는 tail reward가 약 `-457`인데 검증 RMSE는 `0.54/0.81`이었다. 따라서 reward curve는 수렴 여부를 확인하는 자료이고, 최종 구조 선택은 동일 reference에서의 pressure 개형·방향성·reserve를 함께 봐야 한다.

## lib3 중간 결론

현재까지 가장 강한 결론은 “lib3의 실패 원인은 탐색 noise 하나가 아니라, 압력 history만으로는 비선형 밸브와 chamber-to-actuator 지연 상태를 충분히 구분하지 못한다”는 것이다.

1. History SAC baseline은 큰 개형을 일부 따라가지만 고주파 pressure vibration과 offset이 남는다.
2. `dP/dt`만 추가하면 방향성은 좋아지지만 reserve를 모르면 chamber를 과충전하거나 필요한 공급을 늦춘다.
3. reserve만 추가하면 chamber 여유는 알 수 있지만 현재 압력의 진행 방향을 모른다.
4. rate와 reserve를 함께 넣으면 K=1에서도 tolerance와 방향성이 크게 개선된다. 이번 검증에서 가장 균형 잡힌 K=1 기준선은 `hist + rate + reserve`다.
5. K=2를 rate+reserve 위에 추가하면 이번 단일 2048-step 검증에서 가장 낮은 RMSE와 높은 방향 일치율을 얻었다. 다만 policy action 차원이 6→12로 늘고 두 번째 action을 미리 결정하므로, 별도 seed와 receding-horizon 비교가 필요하다.
6. CAPS와 MPOBS1은 lib2에서 유효했던 모듈이지만 lib3에 자동으로 이식되지 않았다. 현재 small CAPS와 MPOBS1은 command smoothness를 만들 수 있어도 tracking 또는 direction을 희생할 수 있다.
7. integral error는 steady-state offset 보정 후보지만, 이번 9개 결과만으로 기본 구조에 넣을 필수 요소라고 판단할 수 없다.

따라서 lib3의 현재 중간 기준 구조는 다음처럼 두는 것이 가장 해석 가능하다.

```text
History SAC
→ pressure-rate 4개 + chamber-reserve 2개 추가
→ K=1을 안정 기준선으로 유지
→ K=2는 성능 후보로 별도 비교
→ CAPS/MPOBS는 이 기준선 위에서 독립 재검증
```

이번 결과만으로 “lib3를 완전히 정복했다”고 말할 수는 없다. 한 trajectory의 2048-step 검증이며, seed 0 중심이고, 최신 profile bank의 모든 member에 대한 평균 성능과 real-to-sim transfer는 아직 남아 있다. 그러나 현재까지는 실패 원인을 `rate 정보 부족`, `reserve 정보 부족`, `decision frequency`, `CAPS/MPOBS의 상태·reward 연결 문제`로 상당히 분리했다. 다음 단계는 새 구조를 계속 늘리는 것보다 `hist+rate+reserve` K=1/K=2를 여러 bank member와 train seed에서 재현하고, 그 뒤 CAPS와 MPOBS를 하나씩 다시 붙이는 것이 맞다.

## 관측값과 action 구조의 현재 위치

| 단계 | 주요 관측값 | 관측 길이 | policy 출력 | 해석 |
|---|---|---:|---:|---|
| 직접형 Pure SAC | 현재 압력 4 + 기준 궤적 4 | 8 | 6 | 가장 단순하지만 지연·reserve를 모름 |
| 압력 history | 압력 history 44 + 기준 궤적 52 | 96 | 6 | 압력의 방향과 기준 변화를 기억 |
| 압력 history + MPOBS1 | 96 + 1-step predictor 4 | 100 | 6 | 짧은 predictor를 추가했지만 plant 내부 지연은 완전하지 않음 |
| 압력 + action history | 96 + 이전 명령 60 | 156 | 6 | 현재까지 가장 deploy 가능한 지연 보완 구조 |
| 압력 + action history + MPOBS1 | 156 + predictor 4 | 160 | 6 | 이전 명령과 predictor를 함께 사용 |
| 압력 + action history + 물리 관측값 | 156 + 물리 관측값 18 | 174 | 6 또는 12 | 현재 가장 설득력 있는 lib3 기준 구조 |
| 위 구조 + 적분 오차 | 174 + 적분 오차 2 | 176 | 6 또는 12 | 정상상태 offset 보정용, wind-up 확인 필요 |

CAPS는 관측 길이를 바꾸지 않고 policy loss에 추가되며, K=2도 관측 길이를 바꾸지 않고 action 출력만 12개로 바꾼다. 즉 현재 좋은 결과가 나온 이유를 정확히 분리하려면 “물리 관측값 효과”와 “K=2 결정 간격 효과”를 각각 K=1/K=2 쌍으로 비교해야 한다.

## 전체 연구 흐름과 구조 변화

### 1. 8월 5일: History SAC 기준선과 seed 확인

직접형 SAC가 밸브 지연과 actuator/chamber 결합을 충분히 처리하지 못했기 때문에 이전 제어 명령을 observation에 추가했다. History SAC는 이전 제어 명령을 통해 밸브 내부 상태와 지연을 간접적으로 추정할 수 있었고, 직접형 SAC보다 압력의 큰 개형을 따라가는 경우가 늘었다.

같은 History SAC를 seed 0/1/2로 학습해 재현성을 확인했다. seed에 따라 양압·음압 offset과 진동 정도가 달라졌고, 따라서 단일 seed의 높은 reward만으로 구조의 우수성을 판단하면 안 된다는 것을 확인했다.

이 단계에서 추가한 것은 다음과 같다.

- `history_action_sac`: 이전 제어 명령 10개
- 이전 제어 명령 길이 5 비교
- 변화율 penalty 0.05
- 충돌 penalty 1.0

결과적으로 이전 제어 명령은 필요하지만 충분조건은 아니었다. 이전 제어 명령은 지연을 기억하는 데 도움을 주지만, 제어 명령의 빠른 전환을 직접 억제하지는 않았다.

### 2. 8월 6일: 선행 연구 구성의 lib3 직접 적용

lib2 선행 연구의 핵심 구성인 SAC, CAPS, MPOBS를 lib3에 적용했다.

- Pure SAC
- History SAC
- History SAC + CAPS
- SAC + MPOBS1 + CAPS
- 512 carry와 1024 carry 비교
- 최신 도달가능성 확인 기준과 예전 무작위 기준 비교

lib2에서는 밸브 제어 명령과 유량의 관계가 비교적 직접적이고 제어 channel도 단순했기 때문에 CAPS와 MPOBS가 진동 억제에 효과적이었다. 그러나 lib3에서는 밸브 parameter가 비선형이고 chamber pressure를 거쳐 actuator pressure로 전달되는 결합이 강하다. 따라서 predictor나 시간 방향 평활화를 추가하는 것만으로는 잘못된 chamber reserve를 자동으로 해결하지 못했다.

특히 최신 기준 궤적에서 동일 조건으로 다시 검증했을 때:

- History SAC: POS/NEG RMSE `4.94/7.50`
- History SAC + CAPS: `8.99/5.45`
- History SAC + MPOBS1 + CAPS: `13.42/5.74`

가 나왔다. 이 결과는 lib2의 기여가 틀렸다는 뜻이 아니라, 현재 lib3의 상태, predictor target, reward, 밸브 비선형성에 그대로 이식할 수 없다는 뜻이다.

### 3. 8월 7일: CAPS 계수, 적분 오차, 2차 차분

CAPS의 시간·공간 계수를 낮추고 mean-action 방식도 비교했다. 작은 CAPS는 제어 명령 전환을 줄이는 장점이 있었지만, 추적 성능을 개선한다고 보기는 어려웠다. 특히 `mean |Δ²u|`가 작아져도 압력 offset이 커질 수 있으므로 제어 명령의 매끄러움만으로 성공을 판단하지 않았다.

정상상태 오차를 줄이기 위해 integral error를 추가했다. 적분항은 지속적인 압력 offset을 줄일 가능성이 있었지만, chamber reserve가 부족한 상태에서 계속 누적되면 wind-up 또는 chamber 과충전이 발생할 수 있었다.

2차 제어 명령 차분 penalty도 시험했다.

```text
L_vibration = ||u_t - 2u_{t-1} + u_{t-2}||²
```

이 penalty는 제어 명령 자체를 고정하지 않고 고주파 방향 전환을 줄이는 목적이다. 하지만 coefficient 50처럼 크게 설정하면 진동은 줄어도 추적 반응성이 사라졌다. 즉 제어 명령 regularization은 압력 오차와 함께 조정해야 했다.

### 4. 8월 7일: K=2 제어 명령 묶음

K=2에서는 policy가 두 개의 제어 명령을 출력하고 시뮬레이터에는 두 실제 step을 순서대로 적용한다. 이번 검증에서는 policy step이 아니라 실제 step 기준을 맞추기 위해 1024 policy step을 사용했다.

K=2의 장점은 policy가 매 step마다 크게 흔들리지 않고 짧은 미래 제어 명령을 함께 고려한다는 점이다. 그러나 K=2만으로는 충분하지 않았다. K=2 기준선은 최신 기준 궤적에서 양압 chamber가 `248.9 kPa`까지 누적되었다.

K=2에 물리 관측값을 추가하면 chamber-actuator reserve와 압력 변화율 정보를 policy가 직접 볼 수 있다. 그 결과 최신 기준 궤적에서 가장 좋은 방향성 결과를 얻었다.

### 5. 8월 8~9일: 15개 동일 조건 절제 실험

마지막으로 두 기준 궤적에서 기본 구조를 다시 맞추고 K=1/K=2와 물리 관측값/적분항을 분리했다. 이 15개는 기준 궤적, seed, 이전 제어 명령, buffer, horizon, gamma를 통일한 비교 세트다.

## 15개 동일 조건 절제 실험 결과

| 번호 | 기준 궤적 | 구성 | POS/NEG RMSE [kPa] | POS/NEG 방향 | 평균 Δu | 결론 |
|---:|---|---|---:|---:|---:|---|
| 1 | 최신 기준 | History SAC | 4.937 / 7.497 | .692 / .603 | .0024 | 음압 offset과 chamber reserve 부족 |
| 2 | 최신 기준 | History SAC + CAPS | 8.989 / 5.449 | .539 / .519 | .0075 | CAPS 단독으로 추적 성능 악화 |
| 3 | 최신 기준 | History SAC + MPOBS1 + CAPS | 13.425 / 5.739 | .554 / .517 | .0049 | 양압 chamber 과충전, 양압 결과 최악 |
| 4 | 예전 무작위 | History SAC | 3.997 / 3.893 | .592 / .532 | .0108 | 큰 개형은 따라가나 제어 명령 전환 존재 |
| 5 | 예전 무작위 | History SAC + CAPS | 5.252 / 4.264 | .550 / .568 | .0057 | 제어 명령은 부드러워졌지만 추적 저하 |
| 6 | 예전 무작위 | History SAC + MPOBS1 + CAPS | 6.796 / 6.648 | .530 / .603 | .0066 | MPOBS1+CAPS 조합의 이점 확인 안 됨 |
| 7 | 최신 기준 | K=1 + 물리 관측값 + 적분항 | 1.756 / 3.887 | .606 / .539 | .0075 | 양압 개선, 음압 offset 잔존 |
| 8 | 최신 기준 | K=2 기준선 | 2.540 / 1.852 | .637 / .783 | .0072 | 양압 chamber가 248.9 kPa까지 증가 |
| 9 | 최신 기준 | K=2 + 물리 관측값 | **1.208 / .981** | **.998 / .996** | **.0020** | 전체적으로 가장 설득력 있는 기준선 |
| 10 | 예전 무작위 | K=1 + 물리 관측값 | 2.224 / 2.663 | .573 / .597 | .0112 | RMSE는 좋지만 방향성은 제한적 |
| 11 | 예전 무작위 | K=1 + 적분항 | 4.867 / 4.781 | .539 / .521 | .0076 | 적분항 단독 효과 약함 |
| 12 | 예전 무작위 | K=1 + 물리 관측값 + 적분항 | 2.678 / 2.739 | .563 / .599 | .0117 | 물리 관측값 단독보다 일관되게 좋지 않음 |
| 13 | 예전 무작위 | K=2 + 물리 관측값 | 2.179 / 3.185 | **.802 / .815** | .0066 | 예전 무작위 강건성 시험에서 가장 좋은 방향성 |
| 14 | 예전 무작위 | K=2 + 물리 관측값 + 적분항 | 2.733 / 2.703 | .638 / .834 | .0162 | 음압은 개선되지만 제어 명령 전환 증가 |
| 15 | 최신 기준 | K=2 + 물리 관측값 + 적분항, seed 1 | .946 / 1.645 | .817 / .909 | .0052 | RMSE는 좋지만 seed 0 물리 관측값 모델보다 방향성 낮음 |

### 최신 도달가능성 확인 기준의 핵심 결과 해석

`K=2 + 물리 관측값`은 압력 개형, 방향 일치, chamber 경계, 제어 명령의 매끄러움을 동시에 만족했다. 양압 chamber 최대값은 `143.0 kPa`였고 K=2 기준선의 `248.9 kPa`보다 훨씬 안정적이었다. 따라서 현재 lib3의 가장 합리적인 기준선은 다음이다.

```text
History SAC
→ K=2 제어 명령 묶음
→ 압력 변화율 / chamber reserve / 이전 제어 명령 물리 관측값
```

K=2 + 물리 관측값 + 적분항 seed 1은 양압 RMSE `0.946 kPa`로 더 낮았지만 K=2 + 물리 관측값 seed 0보다 방향 일치와 제어 명령의 매끄러움이 낮았다. 따라서 현재는 seed 1 결과를 “최고 RMSE”로만 기록하고 구조 기준선은 K=2 + 물리 관측값으로 두는 것이 맞다.

### 예전 무작위 기준 궤적의 결과 해석

예전 무작위 기준 궤적은 일반 학습용 기준이 아니라 jump/high-rate 강건성 시험이다. K=1 물리 관측값과 K=2 물리 관측값 모두 큰 압력 개형은 따라갔지만, 예전 기준 궤적의 변화율과 불연속성 때문에 최신 기준과 같은 척도로 단순 비교할 수 없다.

예전 무작위 기준에서 K=2 + 물리 관측값은 방향 일치 `.802/.815`로 가장 좋았다. 반면 K=2 + 물리 관측값 + 적분항은 음압 RMSE가 약간 좋아졌지만 mean `|Δ²u|`가 `.0209`로 증가해 제어 명령 떨림이 커졌다. 따라서 적분항은 기본 구조에 자동으로 추가할 요소가 아니라 정상상태 오차가 확인된 조건에서 별도로 검증해야 한다.

### 선행 연구 구성의 결론

현재 동일 조건에서는 CAPS와 MPOBS1+CAPS가 최신 기준에서 개선을 만들지 못했다. 특히 MPOBS1+CAPS는 양압 chamber 최대값이 `290.7 kPa`, 양압 허용오차 비율이 `.045`였다.

이 결과만으로 CAPS/MPOBS가 불필요하다고 결론내리면 안 된다. 현재 결과에서 말할 수 있는 결론은 다음과 같다.

1. lib2에서는 CAPS와 MPOBS가 단순한 actuator mapping과 낮은 결합에서 진동 억제에 효과적이었다.
2. lib3에서는 밸브 유량이 비선형이고 chamber reserve를 거쳐 actuator pressure에 영향을 준다.
3. 현재 MPOBS1 predictor는 이 비선형 상태 변화를 충분히 설명하지 못하거나 predictor 정보를 reward/제어 명령과 연결하는 방식이 맞지 않을 수 있다.
4. 따라서 물리 관측값을 포함한 안정적인 lib3 기준선을 먼저 확정한 뒤 CAPS와 MPOBS를 각각 독립적으로 다시 붙이는 것이 올바른 순서다.

## 전체 모델 목록과 검증 링크

아래 표는 최신 도달가능성 확인 기준 또는 예전 무작위 기준을 사용한 전체 47개 모델이다. `구성`의 `K1/K2`는 제어 명령 묶음 길이, `hist`는 이전 제어 명령 길이, `phys`는 물리 관측값, `int`는 적분 오차, `d1/d2`는 제어 명령 차분 penalty를 뜻한다. 각 모델의 PNG와 CSV를 직접 열어 압력, 변화율, chamber, 실제 제어 명령을 확인할 수 있다.

| 날짜/표시명 | 기준 궤적 | 구성 | POS/NEG RMSE | POS/NEG 방향 | 평균/최대 Δu | 결과 그림 | 원자료 |
|---|---|---|---:|---:|---:|---|---|
| 260805 capref base seed0 | 최신 기준 | K1+hist10 | 8.81/3.50 | .74/.61 | .0045/.0881 | [PNG](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed0/episode_01.csv) |
| 260805 capref base seed1 | 최신 기준 | K1+hist10 | 5.38/6.14 | .55/.77 | .0030/.0450 | [PNG](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed1/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed1/episode_01.csv) |
| 260805 capref base seed2 | 최신 기준 | K1+hist10 | 4.51/3.51 | .69/.64 | .0105/.1642 | [PNG](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed2/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed2/episode_01.csv) |
| 260805 conflict1 | 최신 기준 | K1+hist10+conflict | 10.64/3.67 | .62/.54 | .0064/.1155 | [PNG](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_conflict1_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_conflict1_seed0/episode_01.csv) |
| 260805 hist5 | 최신 기준 | K1+hist5 | 12.75/10.73 | .70/.81 | .0022/.0949 | [PNG](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_hist5_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_hist5_seed0/episode_01.csv) |
| 260805 rate005 | 최신 기준 | K1+hist10+rate | 15.09/2.75 | .72/.85 | .0007/.0675 | [PNG](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_rate005_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_rate005_seed0/episode_01.csv) |
| 260806 capref 1024 History SAC | 최신 기준 | K1+hist10 | 4.94/7.50 | .69/.60 | .0024/.0556 | [PNG](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_1024_history_action_sac_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_1024_history_action_sac_seed0/episode_01.csv) |
| 260806 capref 1024 MPOBS1 CAPS | 최신 기준 | K1+MPOBS1 | 7.00/8.36 | .51/.53 | .0061/.0688 | [PNG](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_1024_mpobs1_caps_seed1/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_1024_mpobs1_caps_seed1/episode_01.csv) |
| 260806 capref History CAPS | 최신 기준 | K1+hist10+CAPS | 10.24/17.23 | .85/.70 | .0002/.0168 | [PNG](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_history_action_caps_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_history_action_caps_seed0/episode_01.csv) |
| 260806 capref History SAC | 최신 기준 | K1+hist10 | 4.94/7.50 | .69/.60 | .0024/.0556 | [PNG](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_history_action_sac_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_history_action_sac_seed0/episode_01.csv) |
| 260806 capref MPOBS1 CAPS | 최신 기준 | K1+MPOBS1 | 5.44/4.40 | .51/.54 | .0027/.0868 | [PNG](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_mpobs1_caps_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_mpobs1_caps_seed0/episode_01.csv) |
| 260806 capref Pure SAC | 최신 기준 | K1 | 19.66/8.99 | .52/.54 | .0113/.1519 | [PNG](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_pure_sac_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_pure_sac_seed0/episode_01.csv) |
| 260806 legacy History CAPS | 예전 무작위 | K1+hist10+CAPS | 10.09/5.28 | .60/.57 | .0032/.1282 | [PNG](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_legacy_random_512_history_action_caps_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_legacy_random_512_history_action_caps_seed0/episode_01.csv) |
| 260806 legacy History SAC | 예전 무작위 | K1+hist10 | 4.00/3.89 | .59/.53 | .0108/.1698 | [PNG](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_legacy_random_512_history_action_sac_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_legacy_random_512_history_action_sac_seed0/episode_01.csv) |
| 260807 CAPS .05 legacy mode | 최신 기준 | K1+hist10+CAPS | 14.80/6.38 | .56/.52 | .0110/.0941 | [PNG](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_caps005_legacy_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_caps005_legacy_seed0/episode_01.csv) |
| 260807 CAPS .05 mean action | 최신 기준 | K1+hist10+CAPS | 4.80/5.26 | .53/.53 | .0110/.1513 | [PNG](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_caps005_mean_action_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_caps005_mean_action_seed0/episode_01.csv) |
| 260807 integral chunk2 | 최신 기준 | K2+hist10+int | 1.56/1.07 | .93/.92 | .0032/.1263 | [PNG](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_chunk2_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_chunk2_seed0/episode_01.csv) |
| 260807 integral error | 최신 기준 | K1+hist10+int | 7.45/4.18 | .78/.55 | .0050/.1032 | [PNG](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_error_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_error_seed0/episode_01.csv) |
| 260807 integral + d2 | 최신 기준 | K1+hist10+int+d2 | 4.85/3.30 | .89/.87 | .0001/.0063 | [PNG](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_second_diff50_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_second_diff50_seed0/episode_01.csv) |
| 260807 K2 integral + d1 | 최신 기준 | K2+hist10+int+d1 | 1.47/1.56 | .91/.97 | .0005/.0601 | [PNG](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_d1_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_d1_seed0/episode_01.csv) |
| 260807 K2 integral + d2 | 최신 기준 | K2+hist10+int+d2 | 4.71/3.06 | .59/.62 | .0128/.1158 | [PNG](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_d2_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_d2_seed0/episode_01.csv) |
| 260807 K2 physics + bound | 최신 기준 | K2+hist10+phys+int | 1.47/1.22 | .94/.85 | .0059/.1172 | [PNG](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_bound_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_bound_seed0/episode_01.csv) |
| 260807 K2 physics + d1 | 최신 기준 | K2+hist10+phys+int+d1 | 1.87/1.20 | .81/.68 | .0072/.1106 | [PNG](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_d1_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_d1_seed0/episode_01.csv) |
| 260807 K2 physics + d2 | 최신 기준 | K2+hist10+phys+int+d2 | 1.96/1.97 | .72/.87 | .0143/.1189 | [PNG](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_d2_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_d2_seed0/episode_01.csv) |
| 260807 K2 physics + integral | 최신 기준 | K2+hist10+phys+int | 1.00/1.82 | .89/.69 | .0112/.1659 | [PNG](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_seed0/episode_01.csv) |
| 260807 K2 integral seed1 | 최신 기준 | K2+hist10+int | 2.18/1.09 | .78/.86 | .0023/.1306 | [PNG](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_seed1/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_seed1/episode_01.csv) |
| 260807 K2 integral seed2 | 최신 기준 | K2+hist10+int | 4.24/2.71 | .60/.58 | .0114/.1171 | [PNG](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_seed2/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_seed2/episode_01.csv) |
| 260807 legacy K2 baseline | 예전 무작위 | K2+hist10 | 4.12/3.69 | .55/.56 | .0161/.1735 | [PNG](../../../pneu_rl/exp/lib3_0807_k2_validation_physical2048/260807_lib3_k2_legacy_baseline_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_k2_validation_physical2048/260807_lib3_k2_legacy_baseline_seed0/episode_01.csv) |
| 260807 legacy K2 integral | 예전 무작위 | K2+hist10+int | 2.33/2.85 | .74/.72 | .0085/.1836 | [PNG](../../../pneu_rl/exp/lib3_0807_k2_validation_physical2048/260807_lib3_k2_legacy_integral_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_k2_validation_physical2048/260807_lib3_k2_legacy_integral_seed0/episode_01.csv) |
| 260807 physics + integral + d2 | 최신 기준 | K1+hist10+phys+int+d2 | 3.22/2.78 | .76/.97 | .0001/.0495 | [PNG](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_physics_integral_second_diff50_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_physics_integral_second_diff50_seed0/episode_01.csv) |
| 260807 physics observation | 최신 기준 | K1+hist10+phys | 2.86/2.30 | .71/.59 | .0086/.1492 | [PNG](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_physics_obs_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_physics_obs_seed0/episode_01.csv) |
| 260807 second difference | 최신 기준 | K1+hist10+d2 | 5.11/5.10 | .80/.80 | .0001/.0063 | [PNG](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_second_diff50_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_second_diff50_seed0/episode_01.csv) |
| 260808 legacy K1 integral | 예전 무작위 | K1+hist10+int | 4.87/4.78 | .54/.52 | .0076/.1781 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_integral_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_integral_seed0/episode_01.csv) |
| 260808 legacy K1 physics integral | 예전 무작위 | K1+hist10+phys+int | 2.68/2.74 | .56/.60 | .0117/.1955 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_physics_integral_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_physics_integral_seed0/episode_01.csv) |
| 260808 legacy K1 physics | 예전 무작위 | K1+hist10+phys | 2.22/2.66 | .57/.60 | .0112/.1751 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_physics_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_physics_seed0/episode_01.csv) |
| 260808 legacy K2 physics integral | 예전 무작위 | K2+hist10+phys+int | 2.73/2.70 | .64/.83 | .0162/.1876 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k2_physics_integral_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k2_physics_integral_seed0/episode_01.csv) |
| 260808 legacy K2 physics | 예전 무작위 | K2+hist10+phys | 2.18/3.19 | .80/.81 | .0066/.1555 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k2_physics_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k2_physics_seed0/episode_01.csv) |
| 260808 profile K1 physics integral | 최신 기준 | K1+hist10+phys+int | 1.76/3.89 | .61/.54 | .0075/.1046 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k1_physics_integral_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k1_physics_integral_seed0/episode_01.csv) |
| 260808 profile K2 baseline | 최신 기준 | K2+hist10 | 2.54/1.85 | .64/.78 | .0072/.0846 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_baseline_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_baseline_seed0/episode_01.csv) |
| 260808 profile K2 physics integral seed1 | 최신 기준 | K2+hist10+phys+int | .95/1.64 | .82/.91 | .0052/.1740 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_integral_seed1/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_integral_seed1/episode_01.csv) |
| 260808 profile K2 physics | 최신 기준 | K2+hist10+phys | 1.21/.98 | .998/.996 | .0020/.1792 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_seed0/episode_01.csv) |
| 260808 clean legacy History CAPS | 예전 무작위 | K1+hist10+CAPS(.05) | 5.25/4.26 | .55/.57 | .0057/.1609 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_caps_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_caps_seed0/episode_01.csv) |
| 260808 clean legacy History MPOBS1 CAPS | 예전 무작위 | K1+hist10+CAPS(.05)+MPOBS1 | 6.80/6.65 | .53/.60 | .0066/.1667 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_mpobs1_caps_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_mpobs1_caps_seed0/episode_01.csv) |
| 260808 clean legacy History SAC | 예전 무작위 | K1+hist10 | 4.00/3.89 | .59/.53 | .0108/.1698 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_sac_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_sac_seed0/episode_01.csv) |
| 260808 clean profile History CAPS | 최신 기준 | K1+hist10+CAPS(.05) | 8.99/5.45 | .54/.52 | .0075/.0801 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_caps_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_caps_seed0/episode_01.csv) |
| 260808 clean profile History MPOBS1 CAPS | 최신 기준 | K1+hist10+CAPS(.05)+MPOBS1 | 13.42/5.74 | .55/.52 | .0049/.0756 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_mpobs1_caps_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_mpobs1_caps_seed0/episode_01.csv) |
| 260808 clean profile History SAC | 최신 기준 | K1+hist10 | 4.94/7.50 | .69/.60 | .0024/.0556 | [PNG](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_sac_seed0/episode_01.png) | [CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_sac_seed0/episode_01.csv) |

표의 이전 모델 검증은 당시 실행 폴더에 저장된 결과를 사용했고 최신 15개는 모두 2026-08-09 동일 조건 검증 폴더에서 다시 확인했다. 따라서 날짜가 다른 모델의 절대 RMSE는 기준 궤적과 검증 조건이 완전히 같을 때만 직접 비교해야 한다.

## 전체 실행 모델 ID

표의 표시명은 읽기 쉽게 줄여 썼지만, 실제 학습에 사용한 모델 ID는 다음과 같다. 이 목록을 기준으로 모델의 실행 순서와 구조 변화가 끊기지 않도록 정리했다.

### 8월 5일

- `260805_lib3_capref4x512_history_action_base_seed0`
- `260805_lib3_capref4x512_history_action_base_seed1`
- `260805_lib3_capref4x512_history_action_base_seed2`
- `260805_lib3_capref4x512_history_action_conflict1_seed0`
- `260805_lib3_capref4x512_history_action_hist5_seed0`
- `260805_lib3_capref4x512_history_action_rate005_seed0`

### 8월 6일

- `260806_lib3_capref_1024_history_action_sac_seed0`
- `260806_lib3_capref_1024_mpobs1_caps_seed1`
- `260806_lib3_capref_512_history_action_caps_seed0`
- `260806_lib3_capref_512_history_action_sac_seed0`
- `260806_lib3_capref_512_mpobs1_caps_seed0`
- `260806_lib3_capref_512_pure_sac_seed0`
- `260806_lib3_legacy_random_512_history_action_caps_seed0`
- `260806_lib3_legacy_random_512_history_action_sac_seed0`

### 8월 7일

`260807_lib3_caps005_legacy_seed0`는 모델 ID에 `legacy`가 들어가지만 실제 설정 파일은 최신 도달가능성 확인 기준을 사용했다. 기준 궤적 판정은 모델 이름이 아니라 `cfg.yaml`의 reference 설정을 기준으로 했다.

- `260807_lib3_caps005_legacy_seed0`
- `260807_lib3_caps005_mean_action_seed0`
- `260807_lib3_integral_chunk2_seed0`
- `260807_lib3_integral_error_seed0`
- `260807_lib3_integral_second_diff50_seed0`
- `260807_lib3_k2_integral_profile_d1_seed0`
- `260807_lib3_k2_integral_profile_d2_seed0`
- `260807_lib3_k2_integral_profile_physics_bound_seed0`
- `260807_lib3_k2_integral_profile_physics_d1_seed0`
- `260807_lib3_k2_integral_profile_physics_d2_seed0`
- `260807_lib3_k2_integral_profile_physics_seed0`
- `260807_lib3_k2_integral_profile_seed1`
- `260807_lib3_k2_integral_profile_seed2`
- `260807_lib3_k2_legacy_baseline_seed0`
- `260807_lib3_k2_legacy_integral_seed0`
- `260807_lib3_physics_integral_second_diff50_seed0`
- `260807_lib3_physics_obs_seed0`
- `260807_lib3_second_diff50_seed0`

### 8월 8~9일: 15개 동일 조건 절제 실험

- `260808_lib3_ablation_legacy_k1_integral_seed0`
- `260808_lib3_ablation_legacy_k1_physics_integral_seed0`
- `260808_lib3_ablation_legacy_k1_physics_seed0`
- `260808_lib3_ablation_legacy_k2_physics_integral_seed0`
- `260808_lib3_ablation_legacy_k2_physics_seed0`
- `260808_lib3_ablation_profile_k1_physics_integral_seed0`
- `260808_lib3_ablation_profile_k2_baseline_seed0`
- `260808_lib3_ablation_profile_k2_physics_integral_seed1`
- `260808_lib3_ablation_profile_k2_physics_seed0`
- `260808_lib3_clean_legacy_history_caps_seed0`
- `260808_lib3_clean_legacy_history_mpobs1_caps_seed0`
- `260808_lib3_clean_legacy_history_sac_seed0`
- `260808_lib3_clean_profile_history_caps_seed0`
- `260808_lib3_clean_profile_history_mpobs1_caps_seed0`
- `260808_lib3_clean_profile_history_sac_seed0`

## 추가 기준 궤적 교차 검증

최신 도달가능성 확인 기준에서 학습한 모델을 예전 무작위 기준에 다시 넣어 강건성을 확인한 결과도 있다.

| 학습 모델 | 예전 무작위 POS/NEG RMSE | POS/NEG 방향 | 그림 | 해석 |
|---|---:|---:|---|
| 260807 integral chunk2 | 3.06/3.57 | .77/.71 | [PNG](../../../pneu_rl/exp/lib3_0808_legacy_crossref_validation_physical2048/260807_lib3_integral_chunk2_seed0/episode_01.png) | 기존 best의 강건성은 유지되지만 기준 jump에서 offset 증가 |
| 260807 K2 integral + d1 | 3.87/3.76 | .58/.63 | [PNG](../../../pneu_rl/exp/lib3_0808_legacy_crossref_validation_physical2048/260807_lib3_k2_integral_profile_d1_seed0/episode_01.png) | 제어 명령 regularization이 jump 대응을 제한 |
| 260807 K2 physics + bound | 2.04/2.64 | .69/.75 | [PNG](../../../pneu_rl/exp/lib3_0808_legacy_crossref_validation_physical2048/260807_lib3_k2_integral_profile_physics_bound_seed0/episode_01.png) | chamber bound가 예전 기준의 reserve 보호에 도움 |
| 260807 K2 physics + d1 | 2.05/2.59 | .68/.78 | [PNG](../../../pneu_rl/exp/lib3_0808_legacy_crossref_validation_physical2048/260807_lib3_k2_integral_profile_physics_d1_seed0/episode_01.png) | 음압 방향성은 개선되지만 완전한 안정성은 아님 |

이 결과는 물리 관측값과 chamber reserve 정보가 기준 궤적이 바뀌어도 비교적 일반화된다는 근거다. 반대로 단순 CAPS/MPOBS의 효과는 기준 궤적과 비선형 밸브 동역학에 매우 의존한다.

## 최종 결론과 다음 기준 모델

현재까지의 실험에서 가장 중요한 결론은 “SAC가 무조건 안 된다”가 아니다. 정확히는 다음과 같다.

1. 직접형 Pure SAC는 lib3 6-action 비선형 시스템에서 진동과 큰 offset을 보였다.
2. History SAC는 지연과 이전 밸브 상태를 간접적으로 보완했지만, 압력 방향과 chamber reserve를 명시적으로 알지는 못했다.
3. CAPS는 제어 명령 전환을 줄일 수 있지만, lib3에서는 매끄러운 제어 명령과 좋은 압력 추적이 동일하지 않았다.
4. MPOBS1+CAPS는 현재 predictor와 reward 연결만으로는 chamber reserve와 비선형 유량 threshold를 해결하지 못했다.
5. 적분 오차는 정상상태 offset을 줄일 수 있지만 wind-up과 제어 명령 전환을 유발할 수 있다.
6. K=2는 policy 판단을 짧게 묶어 진동을 줄이는 데 도움을 줬다.
7. K=2에 물리 관측값을 추가했을 때 chamber reserve와 압력 변화율 방향이 명시되어 가장 좋은 결과가 나왔다.

따라서 다음 실험의 기준 모델은 다음으로 고정하는 것이 합리적이다.

```text
History SAC + K=2 제어 명령 묶음 + 물리 관측값
```

다음 우선순위는 이 모델의 seed 1/2 재현성, 동일 모델의 예전 무작위 기준 교차 검증, 그 이후에 CAPS와 MPOBS를 각각 하나씩 추가하는 것이다. 현재 결과만으로 CAPS와 MPOBS를 삭제하지 않고, 먼저 lib3에서 물리 관측값 기준선을 확정한 뒤 선행 연구 기여를 통제된 절제 실험으로 다시 확인해야 한다.

## lib3 강화학습의 최종 결론

현재까지의 결과를 하나의 문장으로 정리하면 다음과 같다.

> lib3는 SAC 자체로 제어할 수 없는 시스템이 아니라, 현재 압력만 입력받아 여섯 밸브를 직접 결정하는 구조로는 지연·비선형 유량·chamber reserve·숨은 밸브 상태를 충분히 구분하기 어려운 시스템이다.

실험의 발전 순서는 다음과 같다.

```text
직접형 SAC
→ 압력 history
→ 이전 밸브 명령 history
→ CAPS/MPOBS 선행 구조 재적용
→ 물리 관측값 추가
→ 적분 오차 추가
→ K=2 action chunking
```

이 중 가장 확실한 개선은 `이전 밸브 명령 history + 물리 관측값`이었다. K=2는 이 구조에 추가했을 때 좋은 결과를 만들었지만, K=2가 독립적으로 모든 문제를 해결한 것은 아니다. K=1에서도 물리 관측값과 적분항을 사용하면 양압 추적이 상당히 개선됐고, 예전 무작위 기준에서는 K=1 물리 관측값도 방향성 있는 큰 개형을 만들었다.

그러므로 현재의 최종 기준 모델은 하나의 고정된 “K=2가 정답”이 아니라 다음 두 모델의 쌍으로 정의하는 것이 가장 정확하다.

1. K=1 + History SAC + 물리 관측값: 174차원 observation, 6차원 action
2. K=2 + History SAC + 물리 관측값: 174차원 observation, 12차원 action

두 모델을 같은 seed·reference·episode·검증 길이로 비교해야 K=2가 실제로 진동을 줄인 것인지, 아니면 물리 관측값이 대부분의 개선을 만든 것인지 분리할 수 있다.

## 앞으로 진행할 실험

### 1단계: 기준 구조 재현성

- K=1 + 물리 관측값, seed 1/2
- K=2 + 물리 관측값, seed 1/2
- 최신 기준과 예전 무작위 기준 각각 1 episode 검증
- RMSE뿐 아니라 방향 일치율, actual/reference 변화율, chamber 최대·최소, `|Δu|`, `|Δ²u|` 비교

### 2단계: K=1과 K=2의 원인 분리

- 같은 observation에서 K=1/K=2만 바꾸기
- 같은 reward에서 K=1/K=2만 바꾸기
- K=2의 `a0`, `a1` 실행 결과와 chunk reward를 별도로 저장하기
- K=2가 후반부에 일정한 명령으로 고정되는지, 실제로 pressure가 안정화된 것인지 확인하기

### 3단계: 선행 연구 구성의 재검증

물리 관측값 기준선이 안정된 뒤에만 CAPS와 MPOBS를 다시 추가한다.

- K=1 + 물리 관측값 + 작은 mean-action CAPS
- K=2 + 물리 관측값 + 작은 mean-action CAPS
- K=1 + 물리 관측값 + MPOBS1
- K=2 + 물리 관측값 + MPOBS1
- 마지막으로 필요한 경우 MPOBS1+CAPS

이 순서라야 CAPS/MPOBS가 실패했을 때 “기준 구조가 불안정해서”인지 “추가 모듈이 실제로 방해해서”인지 구별할 수 있다. lib2에서의 기여를 삭제하는 것이 아니라 lib3의 비선형·지연 조건에 맞게 다시 검증하는 과정이다.

### 4단계: 구조 확장

현재 직접형 MLP가 K=1/K=2 비교에서 충분히 안정되면 다음 구조를 검토한다.

- 압력·이전 명령 history를 평탄화하지 않고 GRU/LSTM으로 인코딩
- actor는 실제 측정 가능한 pressure/history/physics만 사용하고 critic만 simulator privileged state를 사용하는 비대칭 actor-critic
- K=2를 미리 두 명령 실행하는 방식에서, `a0`만 실행하고 새 observation으로 다시 `[a1, a2]`를 예측하는 receding-horizon 구조
- 최종적으로 real fine-tuning에 사용할 때 simulator 전용 `z`, `q_pred`, `mdot`는 직접 입력하지 않고 검증·critic·보조 loss에 한정

### 최종 판단 기준

최종 모델을 선택할 때는 RMSE 하나가 아니라 다음 조건을 동시에 만족해야 한다.

- 기준 궤적과 실제 압력의 상승·하강 방향이 일치할 것
- actual pressure가 기준 궤적을 일정 offset 없이 따라갈 것
- 실제 변화율이 기준 변화율보다 과도하게 크지 않을 것
- chamber가 reserve를 유지하면서도 과충전되지 않을 것
- 제어 명령의 고주파 전환이 줄어들 것
- seed와 두 기준 궤적에서 같은 경향이 재현될 것

이 기준으로 보면 현재 가장 유망한 방향은 `History SAC + 물리 관측값`을 핵심 구조로 두고, K=1과 K=2를 동등한 후보로 비교한 뒤, CAPS와 MPOBS를 그 위에 통제된 방식으로 재추가하는 것이다.

## 대표 최신 검증 그림

### 최신 도달가능성 확인 기준

![History SAC 기준선](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_sac_seed0/episode_01.png)

![K2 기준선](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_baseline_seed0/episode_01.png)

![K2 물리 관측값](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_seed0/episode_01.png)

![K2 물리 관측값 적분항 seed1](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_integral_seed1/episode_01.png)

### 예전 무작위 기준

![예전 무작위 기준 History SAC](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_sac_seed0/episode_01.png)

![예전 무작위 기준 K2 물리 관측값](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k2_physics_seed0/episode_01.png)

![예전 무작위 기준 K2 물리 관측값 적분항](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k2_physics_integral_seed0/episode_01.png)

## 실험 스크립트와 설정

- [6개 기본 절제 실험 전체 실행](run_6_clean_ablation.sh)
- [6개를 두 터미널로 나눠 실행](run_6_clean_ablation_groups.sh)
- [9개 rate/reserve 절제 실험을 세 터미널로 나눠 실행](run_9_rate_reserve_ablation_groups.sh)
- [추가 9개를 세 터미널로 나눠 실행](run_9_followup_ablation_groups.sh)
- [10개 실험 원본 스크립트](run_10_experiments.sh)
- 최신 도달가능성 확인 기준: `pneu_rl/config/reference/lib3_capacity_random_medium_16x4x512_seed61098.yaml`
+

## 모델별 검증 그림 토글

아래에서 모델명을 누르면 해당 모델의 검증 그림과 수치를 펼쳐볼 수 있다. 모든 그림은 2048 실제 시뮬레이터 step 검증 결과다.

<details>
<summary><code>260805_lib3_capref4x512_history_action_base_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10
- POS/NEG RMSE: 8.81/3.50
- POS/NEG 방향 일치율: .74/.61
- 평균/최대 제어 명령 변화량: .0045/.0881

![260805_lib3_capref4x512_history_action_base_seed0 검증 그림](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed0/episode_01.csv)

</details>

<details>
<summary><code>260805_lib3_capref4x512_history_action_base_seed1</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10
- POS/NEG RMSE: 5.38/6.14
- POS/NEG 방향 일치율: .55/.77
- 평균/최대 제어 명령 변화량: .0030/.0450

![260805_lib3_capref4x512_history_action_base_seed1 검증 그림](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed1/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed1/episode_01.csv)

</details>

<details>
<summary><code>260805_lib3_capref4x512_history_action_base_seed2</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10
- POS/NEG RMSE: 4.51/3.51
- POS/NEG 방향 일치율: .69/.64
- 평균/최대 제어 명령 변화량: .0105/.1642

![260805_lib3_capref4x512_history_action_base_seed2 검증 그림](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed2/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_base_seed2/episode_01.csv)

</details>

<details>
<summary><code>260805_lib3_capref4x512_history_action_conflict1_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+conflict
- POS/NEG RMSE: 10.64/3.67
- POS/NEG 방향 일치율: .62/.54
- 평균/최대 제어 명령 변화량: .0064/.1155

![260805_lib3_capref4x512_history_action_conflict1_seed0 검증 그림](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_conflict1_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_conflict1_seed0/episode_01.csv)

</details>

<details>
<summary><code>260805_lib3_capref4x512_history_action_hist5_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist5
- POS/NEG RMSE: 12.75/10.73
- POS/NEG 방향 일치율: .70/.81
- 평균/최대 제어 명령 변화량: .0022/.0949

![260805_lib3_capref4x512_history_action_hist5_seed0 검증 그림](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_hist5_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_hist5_seed0/episode_01.csv)

</details>

<details>
<summary><code>260805_lib3_capref4x512_history_action_rate005_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+rate
- POS/NEG RMSE: 15.09/2.75
- POS/NEG 방향 일치율: .72/.85
- 평균/최대 제어 명령 변화량: .0007/.0675

![260805_lib3_capref4x512_history_action_rate005_seed0 검증 그림](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_rate005_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_capref_validation_2048/260805_lib3_capref4x512_history_action_rate005_seed0/episode_01.csv)

</details>

<details>
<summary><code>260806_lib3_capref_1024_history_action_sac_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10
- POS/NEG RMSE: 4.94/7.50
- POS/NEG 방향 일치율: .69/.60
- 평균/최대 제어 명령 변화량: .0024/.0556

![260806_lib3_capref_1024_history_action_sac_seed0 검증 그림](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_1024_history_action_sac_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_1024_history_action_sac_seed0/episode_01.csv)

</details>

<details>
<summary><code>260806_lib3_capref_1024_mpobs1_caps_seed1</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+MPOBS1
- POS/NEG RMSE: 7.00/8.36
- POS/NEG 방향 일치율: .51/.53
- 평균/최대 제어 명령 변화량: .0061/.0688

![260806_lib3_capref_1024_mpobs1_caps_seed1 검증 그림](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_1024_mpobs1_caps_seed1/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_1024_mpobs1_caps_seed1/episode_01.csv)

</details>

<details>
<summary><code>260806_lib3_capref_512_history_action_caps_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+CAPS
- POS/NEG RMSE: 10.24/17.23
- POS/NEG 방향 일치율: .85/.70
- 평균/최대 제어 명령 변화량: .0002/.0168

![260806_lib3_capref_512_history_action_caps_seed0 검증 그림](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_history_action_caps_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_history_action_caps_seed0/episode_01.csv)

</details>

<details>
<summary><code>260806_lib3_capref_512_history_action_sac_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10
- POS/NEG RMSE: 4.94/7.50
- POS/NEG 방향 일치율: .69/.60
- 평균/최대 제어 명령 변화량: .0024/.0556

![260806_lib3_capref_512_history_action_sac_seed0 검증 그림](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_history_action_sac_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_history_action_sac_seed0/episode_01.csv)

</details>

<details>
<summary><code>260806_lib3_capref_512_mpobs1_caps_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+MPOBS1
- POS/NEG RMSE: 5.44/4.40
- POS/NEG 방향 일치율: .51/.54
- 평균/최대 제어 명령 변화량: .0027/.0868

![260806_lib3_capref_512_mpobs1_caps_seed0 검증 그림](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_mpobs1_caps_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_mpobs1_caps_seed0/episode_01.csv)

</details>

<details>
<summary><code>260806_lib3_capref_512_pure_sac_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1
- POS/NEG RMSE: 19.66/8.99
- POS/NEG 방향 일치율: .52/.54
- 평균/최대 제어 명령 변화량: .0113/.1519

![260806_lib3_capref_512_pure_sac_seed0 검증 그림](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_pure_sac_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_capref_512_pure_sac_seed0/episode_01.csv)

</details>

<details>
<summary><code>260806_lib3_legacy_random_512_history_action_caps_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K1+hist10+CAPS
- POS/NEG RMSE: 10.09/5.28
- POS/NEG 방향 일치율: .60/.57
- 평균/최대 제어 명령 변화량: .0032/.1282

![260806_lib3_legacy_random_512_history_action_caps_seed0 검증 그림](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_legacy_random_512_history_action_caps_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_legacy_random_512_history_action_caps_seed0/episode_01.csv)

</details>

<details>
<summary><code>260806_lib3_legacy_random_512_history_action_sac_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K1+hist10
- POS/NEG RMSE: 4.00/3.89
- POS/NEG 방향 일치율: .59/.53
- 평균/최대 제어 명령 변화량: .0108/.1698

![260806_lib3_legacy_random_512_history_action_sac_seed0 검증 그림](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_legacy_random_512_history_action_sac_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_morning_validation_2048_20260807/260806_lib3_legacy_random_512_history_action_sac_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_caps005_legacy_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+CAPS
- POS/NEG RMSE: 14.80/6.38
- POS/NEG 방향 일치율: .56/.52
- 평균/최대 제어 명령 변화량: .0110/.0941

![260807_lib3_caps005_legacy_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_caps005_legacy_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_caps005_legacy_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_caps005_mean_action_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+CAPS
- POS/NEG RMSE: 4.80/5.26
- POS/NEG 방향 일치율: .53/.53
- 평균/최대 제어 명령 변화량: .0110/.1513

![260807_lib3_caps005_mean_action_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_caps005_mean_action_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_caps005_mean_action_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_integral_chunk2_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10+int
- POS/NEG RMSE: 1.56/1.07
- POS/NEG 방향 일치율: .93/.92
- 평균/최대 제어 명령 변화량: .0032/.1263

![260807_lib3_integral_chunk2_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_chunk2_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_chunk2_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_integral_error_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+int
- POS/NEG RMSE: 7.45/4.18
- POS/NEG 방향 일치율: .78/.55
- 평균/최대 제어 명령 변화량: .0050/.1032

![260807_lib3_integral_error_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_error_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_error_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_integral_second_diff50_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+int+d2
- POS/NEG RMSE: 4.85/3.30
- POS/NEG 방향 일치율: .89/.87
- 평균/최대 제어 명령 변화량: .0001/.0063

![260807_lib3_integral_second_diff50_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_second_diff50_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_integral_second_diff50_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_k2_integral_profile_d1_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10+int+d1
- POS/NEG RMSE: 1.47/1.56
- POS/NEG 방향 일치율: .91/.97
- 평균/최대 제어 명령 변화량: .0005/.0601

![260807_lib3_k2_integral_profile_d1_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_d1_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_d1_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_k2_integral_profile_d2_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10+int+d2
- POS/NEG RMSE: 4.71/3.06
- POS/NEG 방향 일치율: .59/.62
- 평균/최대 제어 명령 변화량: .0128/.1158

![260807_lib3_k2_integral_profile_d2_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_d2_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_d2_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_k2_integral_profile_physics_bound_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10+phys+int
- POS/NEG RMSE: 1.47/1.22
- POS/NEG 방향 일치율: .94/.85
- 평균/최대 제어 명령 변화량: .0059/.1172

![260807_lib3_k2_integral_profile_physics_bound_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_bound_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_bound_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_k2_integral_profile_physics_d1_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10+phys+int+d1
- POS/NEG RMSE: 1.87/1.20
- POS/NEG 방향 일치율: .81/.68
- 평균/최대 제어 명령 변화량: .0072/.1106

![260807_lib3_k2_integral_profile_physics_d1_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_d1_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_d1_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_k2_integral_profile_physics_d2_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10+phys+int+d2
- POS/NEG RMSE: 1.96/1.97
- POS/NEG 방향 일치율: .72/.87
- 평균/최대 제어 명령 변화량: .0143/.1189

![260807_lib3_k2_integral_profile_physics_d2_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_d2_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_d2_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_k2_integral_profile_physics_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10+phys+int
- POS/NEG RMSE: 1.00/1.82
- POS/NEG 방향 일치율: .89/.69
- 평균/최대 제어 명령 변화량: .0112/.1659

![260807_lib3_k2_integral_profile_physics_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_physics_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_k2_integral_profile_seed1</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10+int
- POS/NEG RMSE: 2.18/1.09
- POS/NEG 방향 일치율: .78/.86
- 평균/최대 제어 명령 변화량: .0023/.1306

![260807_lib3_k2_integral_profile_seed1 검증 그림](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_seed1/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_seed1/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_k2_integral_profile_seed2</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10+int
- POS/NEG RMSE: 4.24/2.71
- POS/NEG 방향 일치율: .60/.58
- 평균/최대 제어 명령 변화량: .0114/.1171

![260807_lib3_k2_integral_profile_seed2 검증 그림](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_seed2/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_k2_integral_profile_validation_physical2048/260807_lib3_k2_integral_profile_seed2/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_k2_legacy_baseline_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K2+hist10
- POS/NEG RMSE: 4.12/3.69
- POS/NEG 방향 일치율: .55/.56
- 평균/최대 제어 명령 변화량: .0161/.1735

![260807_lib3_k2_legacy_baseline_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_k2_validation_physical2048/260807_lib3_k2_legacy_baseline_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_k2_validation_physical2048/260807_lib3_k2_legacy_baseline_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_k2_legacy_integral_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K2+hist10+int
- POS/NEG RMSE: 2.33/2.85
- POS/NEG 방향 일치율: .74/.72
- 평균/최대 제어 명령 변화량: .0085/.1836

![260807_lib3_k2_legacy_integral_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_k2_validation_physical2048/260807_lib3_k2_legacy_integral_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_k2_validation_physical2048/260807_lib3_k2_legacy_integral_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_physics_integral_second_diff50_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+phys+int+d2
- POS/NEG RMSE: 3.22/2.78
- POS/NEG 방향 일치율: .76/.97
- 평균/최대 제어 명령 변화량: .0001/.0495

![260807_lib3_physics_integral_second_diff50_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_physics_integral_second_diff50_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_physics_integral_second_diff50_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_physics_obs_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+phys
- POS/NEG RMSE: 2.86/2.30
- POS/NEG 방향 일치율: .71/.59
- 평균/최대 제어 명령 변화량: .0086/.1492

![260807_lib3_physics_obs_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_physics_obs_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_physics_obs_seed0/episode_01.csv)

</details>

<details>
<summary><code>260807_lib3_second_diff50_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+d2
- POS/NEG RMSE: 5.11/5.10
- POS/NEG 방향 일치율: .80/.80
- 평균/최대 제어 명령 변화량: .0001/.0063

![260807_lib3_second_diff50_seed0 검증 그림](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_second_diff50_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0807_validation_2048/260807_lib3_second_diff50_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_ablation_legacy_k1_integral_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K1+hist10+int
- POS/NEG RMSE: 4.87/4.78
- POS/NEG 방향 일치율: .54/.52
- 평균/최대 제어 명령 변화량: .0076/.1781

![260808_lib3_ablation_legacy_k1_integral_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_integral_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_integral_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_ablation_legacy_k1_physics_integral_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K1+hist10+phys+int
- POS/NEG RMSE: 2.68/2.74
- POS/NEG 방향 일치율: .56/.60
- 평균/최대 제어 명령 변화량: .0117/.1955

![260808_lib3_ablation_legacy_k1_physics_integral_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_physics_integral_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_physics_integral_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_ablation_legacy_k1_physics_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K1+hist10+phys
- POS/NEG RMSE: 2.22/2.66
- POS/NEG 방향 일치율: .57/.60
- 평균/최대 제어 명령 변화량: .0112/.1751

![260808_lib3_ablation_legacy_k1_physics_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_physics_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k1_physics_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_ablation_legacy_k2_physics_integral_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K2+hist10+phys+int
- POS/NEG RMSE: 2.73/2.70
- POS/NEG 방향 일치율: .64/.83
- 평균/최대 제어 명령 변화량: .0162/.1876

![260808_lib3_ablation_legacy_k2_physics_integral_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k2_physics_integral_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k2_physics_integral_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_ablation_legacy_k2_physics_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K2+hist10+phys
- POS/NEG RMSE: 2.18/3.19
- POS/NEG 방향 일치율: .80/.81
- 평균/최대 제어 명령 변화량: .0066/.1555

![260808_lib3_ablation_legacy_k2_physics_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k2_physics_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_legacy_k2_physics_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_ablation_profile_k1_physics_integral_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+phys+int
- POS/NEG RMSE: 1.76/3.89
- POS/NEG 방향 일치율: .61/.54
- 평균/최대 제어 명령 변화량: .0075/.1046

![260808_lib3_ablation_profile_k1_physics_integral_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k1_physics_integral_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k1_physics_integral_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_ablation_profile_k2_baseline_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10
- POS/NEG RMSE: 2.54/1.85
- POS/NEG 방향 일치율: .64/.78
- 평균/최대 제어 명령 변화량: .0072/.0846

![260808_lib3_ablation_profile_k2_baseline_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_baseline_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_baseline_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_ablation_profile_k2_physics_integral_seed1</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10+phys+int
- POS/NEG RMSE: .95/1.64
- POS/NEG 방향 일치율: .82/.91
- 평균/최대 제어 명령 변화량: .0052/.1740

![260808_lib3_ablation_profile_k2_physics_integral_seed1 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_integral_seed1/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_integral_seed1/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_ablation_profile_k2_physics_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K2+hist10+phys
- POS/NEG RMSE: 1.21/.98
- POS/NEG 방향 일치율: .998/.996
- 평균/최대 제어 명령 변화량: .0020/.1792

![260808_lib3_ablation_profile_k2_physics_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_ablation_profile_k2_physics_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_clean_legacy_history_caps_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K1+hist10+CAPS(.05)
- POS/NEG RMSE: 5.25/4.26
- POS/NEG 방향 일치율: .55/.57
- 평균/최대 제어 명령 변화량: .0057/.1609

![260808_lib3_clean_legacy_history_caps_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_caps_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_caps_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_clean_legacy_history_mpobs1_caps_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K1+hist10+CAPS(.05)+MPOBS1
- POS/NEG RMSE: 6.80/6.65
- POS/NEG 방향 일치율: .53/.60
- 평균/최대 제어 명령 변화량: .0066/.1667

![260808_lib3_clean_legacy_history_mpobs1_caps_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_mpobs1_caps_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_mpobs1_caps_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_clean_legacy_history_sac_seed0</code></summary>

- 기준 궤적: 예전 무작위
- 구성: K1+hist10
- POS/NEG RMSE: 4.00/3.89
- POS/NEG 방향 일치율: .59/.53
- 평균/최대 제어 명령 변화량: .0108/.1698

![260808_lib3_clean_legacy_history_sac_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_sac_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_legacy_history_sac_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_clean_profile_history_caps_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+CAPS(.05)
- POS/NEG RMSE: 8.99/5.45
- POS/NEG 방향 일치율: .54/.52
- 평균/최대 제어 명령 변화량: .0075/.0801

![260808_lib3_clean_profile_history_caps_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_caps_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_caps_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_clean_profile_history_mpobs1_caps_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10+CAPS(.05)+MPOBS1
- POS/NEG RMSE: 13.42/5.74
- POS/NEG 방향 일치율: .55/.52
- 평균/최대 제어 명령 변화량: .0049/.0756

![260808_lib3_clean_profile_history_mpobs1_caps_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_mpobs1_caps_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_mpobs1_caps_seed0/episode_01.csv)

</details>

<details>
<summary><code>260808_lib3_clean_profile_history_sac_seed0</code></summary>

- 기준 궤적: 최신 기준
- 구성: K1+hist10
- POS/NEG RMSE: 4.94/7.50
- POS/NEG 방향 일치율: .69/.60
- 평균/최대 제어 명령 변화량: .0024/.0556

![260808_lib3_clean_profile_history_sac_seed0 검증 그림](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_sac_seed0/episode_01.png)

[원자료 CSV](../../../pneu_rl/exp/lib3_0809_ablation_validation_2048/260808_lib3_clean_profile_history_sac_seed0/episode_01.csv)

</details>
