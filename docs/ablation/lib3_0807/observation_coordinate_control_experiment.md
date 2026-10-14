# lib3 observation coordinate controls

## 질문

Pressure rate는 11-step pressure history에서 정확히 계산된다. 따라서 이 실험은 rate가 새로운 Markov information인지 검사하지 않는다. 다음 두 가능성을 분리한다.

1. signed finite-difference 좌표를 명시한 효과
2. 기존 rate normalization이 history 차분보다 약 100배 크게 들어간 numerical emphasis 효과

## Core 4-way 모델

| ID | 모델 | normalized observation | 상태 | 목적 |
|---|---|---|---|---|
| H | `260810_lib3_ablation_profile_hist_base_seed0` | History | 기존 1000 episode 재사용 | 기준선 |
| R50 | `260810_lib3_ablation_profile_hist_rate_seed0` | History + `rate/50` | 기존 1000 episode 재사용 | 현재 rate 결과 |
| R5000 | `260825_lib3_obscoord_hist_rate_scale5000_seed0` | History + `rate/5000` | 신규 1000 episode | history 차분과 scale match |
| Z4 | `260825_lib3_obscoord_hist_zero4_seed0` | History + zero 4개 | 신규 1000 episode | dimension/fan-in placebo |

lib3는 50 Hz이므로 다음이 성립한다.

\[
\frac{rate}{50}=P_t-P_{t-1},
\qquad
\frac{rate}{5000}=\frac{P_t-P_{t-1}}{100}
\]

두 번째 값은 pressure history가 `P/100`으로 normalization된 상태에서 network가 형성하는 한 step 차분과 같은 크기다. R50과 R5000은 raw rate 값이 같고 SAC 입력 normalization denominator만 다르다.

Z4는 R50과 같은 4차원 observation slot과 scale 50을 사용하지만 값은 항상 0이다. 이는 첫 linear layer의 input dimension과 fan-in 변화만 통제한다. 정보량이나 signal amplitude의 placebo는 아니다.

## Short-history compression pair

Rate가 긴 history를 일부 압축할 수 있는지도 별도 pair로 확인한다. `num_prev=5`는 현재 sample을 포함한 pressure 6개로 `t-0.10 s`부터 `t`까지를 덮는다. Action history도 직전 5개, 즉 0.10 s로 맞춘다. 이는 lib3 sensor delay 0.1 s를 덮는 최소 temporal context라는 물리적 기준이다.

| ID | 모델 | temporal observation | 상태 | 목적 |
|---|---|---|---|---|
| H5 | `260825_lib3_obscoord_short_hist5_base_seed0` | pressure previous 5 + action previous 5 | 신규 1000 episode | short-history 기준선 |
| H5R | `260825_lib3_obscoord_short_hist5_rate_scale5000_seed0` | H5 + `rate/5000` | 신규 1000 episode | scale-matched rate의 history compression 효과 |

H5/H5R에서는 pressure와 action history뿐 아니라 reference past도 5 step으로 줄어든다. Future reference 15 step은 유지된다. 따라서 이 pair는 rate만의 효과를 H5와 H5R 차이로 분리하고, H10과 H5의 차이는 전체 temporal-context 축소 효과로 해석한다.

## 고정 조건

네 모델은 다음 조건이 같아야 한다.

- reference: `lib3_capacity_random_medium_16x4x512_seed61098.yaml`
- seed 0, 1000 episode, horizon 512
- 4-block carry
- Core 4-way: pressure previous 10 + action previous 10
- Short pair: pressure previous 5 + action previous 5
- gamma 0.99, physical action range 0.8–1.0
- observation normalization on
- fixed 0.95 warm-up, warm-up 중 update 보류, 이후 hard reset
- 외부 exploration noise와 모든 보조 reward off
- 동일한 SAC hyperparameter
- 검증 2048 actual simulator step

신규 두 모델은 100 episode마다 `model_epi_NNNN.pth`를 저장한다. Rolling `model.pth`도 종전처럼 매 episode 갱신된다.

## 실행

상태 및 dry-run:

```bash
bash docs/ablation/lib3_0807/run_4_observation_coordinate_controls.sh status
bash docs/ablation/lib3_0807/run_4_observation_coordinate_controls.sh dry-run
```

두 terminal에서 신규 모델을 병렬 학습할 수 있다.

```bash
bash docs/ablation/lib3_0807/run_4_observation_coordinate_controls.sh train-scale-matched
bash docs/ablation/lib3_0807/run_4_observation_coordinate_controls.sh train-zero4
```

Short-history pair도 두 terminal에서 따로 실행하거나 한 terminal에서 순차 실행할 수 있다.

```bash
bash docs/ablation/lib3_0807/run_4_observation_coordinate_controls.sh train-short-history
bash docs/ablation/lib3_0807/run_4_observation_coordinate_controls.sh train-short-history-rate
```

```bash
bash docs/ablation/lib3_0807/run_4_observation_coordinate_controls.sh train-short-pair
```

두 개를 한 terminal에서 순차 실행하려면 다음을 사용한다.

```bash
bash docs/ablation/lib3_0807/run_4_observation_coordinate_controls.sh train-new
```

신규 네 모델을 모두 한 terminal에서 순차 실행하려면 다음을 사용한다.

```bash
bash docs/ablation/lib3_0807/run_4_observation_coordinate_controls.sh train-all-new
```

여섯 final checkpoint의 동일 길이 검증:

```bash
bash docs/ablation/lib3_0807/run_4_observation_coordinate_controls.sh validate
```

신규 periodic checkpoint 하나를 별도로 검증하는 예시는 다음과 같다.

```bash
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/evaluate_lib3_screening.py \
  --model 260825_lib3_obscoord_hist_rate_scale5000_seed0 \
  --checkpoint-episode 800 \
  --episodes 1 \
  --steps 2048 \
  --output-root pneu_rl/exp/lib3_0825_obscoord_checkpoint_validation
```

## 사전 판정 규칙

| 결과 | 해석 |
|---|---|
| R50만 H보다 좋음 | 기존 rate 이득은 numerical emphasis가 지배적 |
| R5000도 H와 Z4보다 좋음 | scale을 맞춰도 signed finite-difference coordinate가 finite SAC 학습을 도움 |
| Z4가 H보다 좋아짐 | input dimension 또는 initialization 변화가 섞임 |
| R5000과 H가 유사함 | rate의 coordinate 효과는 작고 기존 차이는 scale shortcut일 가능성 |
| H5R이 H5보다 좋고 H10에 근접 | rate가 delay-length short history의 효율적인 보조 좌표 |
| H5R이 H5와 유사 | short history에서 explicit rate의 추가 효과가 작음 |
| H5/H5R 모두 H10보다 크게 나쁨 | 0.1 s보다 긴 temporal context가 hidden dynamics 복원에 필요 |

이 seed-0 screening 뒤에만 History 2000 또는 seed 1/2를 결정한다. R5000이 H보다 유의미하게 좋을 때 History의 추가 학습으로 catch-up 여부를 검사한다.

## 2026-08-25 1000-episode 결과와 full-bank 검증

신규 네 모델은 모두 1000 episode, 512,000 simulator step과 episode 1000 checkpoint까지 완료했다. 먼저 기존 명령으로 동일한 한 member를 2048 step 평가했고, 한 member에 종속된 결론을 피하기 위해 evaluator에 `--all-reference-members`를 추가했다. 이 mode는 16개 member를 각각 지정한 뒤 plant와 reference를 hard reset한다. 단순히 `--episodes 16`을 쓰면 carry와 `repeat_after_end=false` 때문에 첫 trajectory 이후 마지막 reference가 hold될 수 있으므로 사용하지 않는다.

```bash
bash docs/ablation/lib3_0807/run_4_observation_coordinate_controls.sh validate-fullbank
```

Full-bank 결과는 6 models × 16 members × 2048 actual simulator step이다. `Combined RMSE`는 member별 `sqrt((POS RMSE^2 + NEG RMSE^2)/2)`의 평균이다. Chamber audit은 generator가 사용한 `CH_POS <= 200 kPa`, `CH_NEG >= 15 kPa`를 기준으로 했다.

| 모델 | reward | RMSE POS/NEG | Combined RMSE | direction POS/NEG | mean abs delta-u | CH POS max / CH NEG min | bound 위반 POS/NEG |
|---|---:|---:|---:|---:|---:|---:|---:|
| H | -4319.9 | 3.706 / 4.939 | 4.370 | .623 / .591 | .00503 | 163.4 / 8.7 | 0 / 4.06% |
| R50 | -3973.5 | 4.252 / 3.555 | 3.922 | .906 / .705 | .00404 | 169.9 / 17.3 | 0 / 0 |
| R5000 | -3941.8 | 3.120 / 4.210 | 3.707 | .965 / .762 | .00257 | 389.0 / 10.1 | 81.53 / 14.09% |
| Z4 | **-3597.3** | 5.204 / **2.239** | 4.035 | .657 / .674 | .00301 | 197.7 / 12.4 | 0 / 1.02% |
| H5 | -10296.0 | 12.135 / 7.951 | 10.400 | .534 / .554 | .01083 | 410.1 / 12.6 | 27.09 / .54% |
| H5R | -5892.8 | 7.480 / 4.039 | 6.084 | .564 / .537 | .01093 | 144.9 / 9.8 | 0 / 48.98% |

[Full-bank H summary](../../../pneu_rl/exp/lib3_0825_obscoord_6model_fullbank_actual2048/260810_lib3_ablation_profile_hist_base_seed0/summary.yaml) · [R50 summary](../../../pneu_rl/exp/lib3_0825_obscoord_6model_fullbank_actual2048/260810_lib3_ablation_profile_hist_rate_seed0/summary.yaml) · [R5000 summary](../../../pneu_rl/exp/lib3_0825_obscoord_6model_fullbank_actual2048/260825_lib3_obscoord_hist_rate_scale5000_seed0/summary.yaml) · [Z4 summary](../../../pneu_rl/exp/lib3_0825_obscoord_6model_fullbank_actual2048/260825_lib3_obscoord_hist_zero4_seed0/summary.yaml) · [H5 summary](../../../pneu_rl/exp/lib3_0825_obscoord_6model_fullbank_actual2048/260825_lib3_obscoord_short_hist5_base_seed0/summary.yaml) · [H5R summary](../../../pneu_rl/exp/lib3_0825_obscoord_6model_fullbank_actual2048/260825_lib3_obscoord_short_hist5_rate_scale5000_seed0/summary.yaml)

### Member-paired 결과

16개 member를 paired sample로 bootstrap한 95% interval이다. 이 interval은 reference-member 변동만 나타내며 policy training seed 변동을 포함하지 않는다.

| 비교 | Combined RMSE 차이 | member win | direction 평균 차이 | 해석 |
|---|---:|---:|---:|---|
| R50 - H | **-.448 [-.693, -.192]** | 12/16 | **+.199 [.171, .221]** | current-scale rate의 tracking/direction 개선 |
| R5000 - H | **-.663 [-1.036, -.237]** | 12/16 | **+.257 [.225, .286]** | 수치상 개선이나 chamber constraint 위반으로 무효 |
| Z4 - H | -.335 [-1.222, +.664] | 11/16 | +.059 [.005, .116] | placebo 자체도 큰 seed-0 변동을 만듦 |
| R5000 - Z4 | -.328 [-.984, +.259] | 9/16 | **+.198 [.146, .243]** | direction 외 tracking/return의 rate 고유효과 미확정 |
| H5R - H5 | **-4.316 [-5.861, -2.806]** | 14/16 | +.007 [-.022, .034] | short-history 학습은 크게 구제하지만 direction은 못 복원 |
| H5R - H | **+1.714 [.785, 2.701]** | 4/16 | -.056 [-.099, -.016] | rate가 10-step history를 대체하지 못함 |

### Plateau와 transition

Reference absolute rate `<= 0.1 kPa/s`를 plateau로 나누었다.

| 모델 | POS plateau / transition RMSE | NEG plateau / transition RMSE | plateau signed bias POS/NEG |
|---|---:|---:|---:|
| H | 3.839 / 3.715 | 4.967 / 5.071 | +1.004 / +.224 |
| R50 | 4.452 / 4.260 | 3.608 / 3.578 | -3.485 / +2.813 |
| R5000 | 3.170 / 3.107 | 4.309 / 4.174 | -2.786 / +3.698 |
| Z4 | 5.773 / 5.464 | 2.198 / 2.293 | -2.019 / +.620 |
| H5 | 12.559 / 12.438 | 7.909 / 8.930 | -1.337 / -4.255 |
| H5R | 8.398 / 7.729 | 3.781 / 4.223 | +4.108 / -.967 |

Rate 모델의 높은 direction agreement가 steady-state bias 제거를 뜻하지 않는다. R50도 plateau에서 POS `-3.49 kPa`, NEG `+2.81 kPa` bias가 남는다. 이는 기존 8월 10–11일 결론인 `rate는 방향/동특성, steady-state offset은 별도 문제`와 일치한다.

### RMSE 밖의 진폭·진동 분석

RMSE만으로는 Z4의 낮은 NEG error와 rate 모델의 damping 효과를 구별하기 어렵다. Stable plateau는 `abs(reference rate) <= 0.1 kPa/s` 구간을 transition 양쪽에서 0.5 s씩 침식해 정의했다. Pressure high-frequency ripple은 tracking error에서 0.5 s centered moving average를 뺀 residual의 표준편차다. 이 값은 정식 주파수응답 식별치가 아니라 2 Hz 부근 이상의 ripple을 비교하는 공통 진단 지표다.

| 모델 | response gain POS/NEG | amplitude ratio POS/NEG | plateau HF ripple POS/NEG [kPa] | plateau pressure TV POS/NEG | joint-plateau action HF RMS |
|---|---:|---:|---:|---:|---:|
| H | .843 / 1.112 | 1.011 / 1.510 | 3.012 / 3.672 | .622 / .743 | .02411 |
| R50 | 1.123 / .983 | 1.185 / 1.076 | **.277 / .901** | **.062 / .193** | **.00953** |
| R5000 | 1.058 / .937 | **1.089 / 1.026** | **.222 / .488** | **.042 / .088** | **.00745** |
| Z4 | .929 / 1.024 | 1.212 / 1.102 | 3.757 / 1.216 | .756 / .225 | .01531 |

R50-Z4의 member-paired 차이는 POS HF ripple `-3.480 kPa [-4.429, -2.447]`, action HF RMS `-.0058 [-.0101, -.0010]`였다. NEG ripple 차이는 `-.314 [-.644, +.041]`로 방향은 좋지만 interval이 0을 포함했다. R5000-Z4는 POS `-3.535 [-4.523, -2.463]`, NEG `-.728 [-1.073, -.352]`, action HF `-.0079 [-.0127, -.0028]`로 두 branch 모두 강한 damping을 보였다. 단, R5000의 chamber constraint 위반 때문에 이를 feasible-control 성능으로 인정하지 않는다.

따라서 Z4가 일부 RMSE와 reward에서 좋아도 rate의 제어적 특징은 분명히 남는다.

1. 같은 tracking error에서도 rising/falling state를 구별해 direction agreement를 높인다.
2. Reference amplitude에 대한 actuator amplitude ratio를 1에 가깝게 만든다. 특히 R5000은 POS/NEG `1.089/1.026`이었다.
3. Plateau pressure ripple과 action의 high-frequency component를 줄여 PD-like damping 역할을 한다.

Rate의 주효과를 `평균 RMSE 감소`로 쓰기보다 **directional response and oscillation damping**으로 정의하는 편이 현재 데이터에 더 정확하다.

### 학습곡선 해석

- R50은 episode 101–300 평균 reward가 `-596/-598`로 H의 `-2010/-1286`보다 훨씬 빨리 학습했다. 현재 100배 강조 scale에는 강한 sample-efficiency 효과가 있다.
- R5000도 episode 201–300에서 `-751`로 Z4의 `-1274`보다 빨랐다. 따라서 scale을 맞춘 rate도 early optimization을 돕는 경향은 있다.
- 하지만 final full-bank reward는 Z4가 R5000보다 좋고 paired return/RMSE interval도 0을 포함한다. rate의 고유한 final-performance 효과는 seed 0에서 확정되지 않는다.
- H5는 episode 601–700에서 평균 reward `-10628`로 붕괴했다. H5R은 같은 구간 `-699`로 안정화되었다. Rate는 짧은 history의 optimization을 돕지만 H10 수준의 state reconstruction을 복원하지 못했다.

### 현재 결론

1. **R50은 이 여섯 모델 중 유일하게 full-bank chamber audit을 모두 통과하면서 H보다 direction과 combined tracking을 개선한 practical seed-0 winner다.** 다만 current scale의 numerical emphasis가 포함된다.
2. **R5000은 유효한 winner가 아니다.** 좋은 tracking과 direction의 대가로 81.5% step에서 POS chamber 200 kPa를 넘고 최대 389 kPa까지 충전했다. Chamber reward가 꺼진 상태에서 생긴 policy exploit이다.
3. **Z4 결과 때문에 rate의 정보적 필수성이나 final-performance 고유효과를 주장할 수 없다.** Placebo도 seed-0에서 reward와 NEG RMSE를 크게 바꿨다.
4. **Rate의 가장 일관된 효과는 방향성과 학습 속도다.** Steady-state bias 제거 효과는 확인되지 않았다.
5. **0.1 s의 H5는 부족하다.** H5R이 H5를 크게 구제하지만 H10보다 나쁘므로 rate는 긴 temporal context의 대체재가 아니다.

다음 단계는 모든 여섯 모델을 그대로 multi-seed 확장하는 것이 아니다. 먼저 chamber constraint를 평가 protocol이 아니라 training objective/termination에 동일하게 적용한 뒤, `H / R50 / R5000 / Z4` 네 모델만 seed 0/1/2로 다시 비교해야 한다. 이 조건에서도 R5000이 Z4보다 learning AUC, direction과 unseen-reference 성능을 반복적으로 개선할 때만 `scale-matched rate is a useful inductive bias`라고 주장한다. Predictor는 이 결론의 보조 진단으로만 둔다.
