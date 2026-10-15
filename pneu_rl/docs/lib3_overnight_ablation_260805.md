# lib3 overnight ablation plan — 2026-08-05

## 결론

현재 lib3 결과만으로는 `SAC+MPObs+CAPS가 된다`고 결론낼 수 없다. 완료된 lib3 모델은 20개이고 설정만 생성된 미완료 모델이 1개지만, 대부분이 같은 네 가지 프로토콜 문제를 공유한다.

1. 첫 10 episode가 random exploration이 아니라 모든 valve에 physical `0.95`를 반복한다.
2. replay가 128개를 넘는 첫 episode부터 actor/critic update가 시작되어, policy를 쓰기도 전에 동일 action 데이터로 약 5,000회 update된다.
3. 512-step carry가 무제한이라 한 번 발생한 chamber runaway가 다음 episode로 계속 전파된다.
4. 기존 CAPS는 독립 stochastic action sample의 차이를 batch 전체 norm으로 벌점화하므로 SAC entropy objective와 충돌한다.

따라서 오늘 밤에는 reference와 주요 SAC hyperparameter를 고정하고, 위 네 문제를 공통으로 수정한 뒤 observation 구조 세 가지와 CAPS 유무를 교차한 6개만 실행한다.

## 기존 결과에서 확정된 사실

### 동일 action warm-up

기존 모든 최근 lib3 config의 값은 다음과 같다.

- `start_epi=10`
- `warmup_action_raw=0.75`
- `scale_low=0.8`, `scale_high=1.0`
- simulator 입력: `0.8 + 0.2 × 0.75 = 0.95`

즉 첫 `10 × 512 = 5,120 step` 동안 6개 valve가 모두 0.95이다. 이것은 random exploration이 아니다. 또한 기존 학습 loop는 replay 크기가 128을 넘으면 warm-up 중에도 매 step update한다.

### unlimited carry와 reward loophole

기존 512-carry 학습의 전체 training trajectory에서 확인된 chamber 극값 예시는 다음과 같다.

| 모델 | CH POS max [kPa] | CH NEG min [kPa] |
|---|---:|---:|
| Pure SAC + reserve | 373.5 | 2.8 |
| MPObs1+CAPS + reserve | 2796.5 | 5.4 |
| MPObs5+CAPS+z + reserve | 4040.8 | 2.6 |
| 6-action physics SAC | 1964.4 | 4.4 |
| 6-action physics MPObs1+CAPS | 1416.4 | 2.5 |

기존 reserve reward는 positive chamber가 부족하거나 negative chamber가 충분히 낮지 않을 때만 벌점을 준다. `CH POS`가 지나치게 높거나 `CH NEG`가 지나치게 낮은 상태에는 벌점이 없어서 tracking reward를 위한 저장압으로 악용할 수 있었다.

### 기존 2048-step one-shot 결과

모두 대기압 시작, 현재 rate-limited reference의 단일 2048-step deterministic validation이다.

| 설정 | POS RMSE | NEG RMSE | POS max rate | NEG max rate | 해석 |
|---|---:|---:|---:|---:|---|
| Pure SAC | 15.11 | 12.14 | 295.0 | 147.4 | 큰 진동 |
| MPObs1+CAPS | 18.18 | 6.14 | 30.5 | 113.3 | POS가 거의 움직이지 않음 |
| MPObs1+CAPS + Huber/rate | 10.25 | 6.69 | 182.5 | 143.1 | RMSE는 낮아졌지만 진동 실패 |
| Pure SAC + reserve | 14.34 | 14.47 | 258.8 | 134.4 | reserve만으로 개선되지 않음 |
| MPObs1+CAPS + reserve | 10.74 | 14.31 | 143.4 | 99.1 | flat/poor tracking |
| MPObs1+z + reserve | 34.64 | 7.42 | 89.1 | 74.1 | POS 방향·크기 실패 |
| MPObs1+CAPS+z + reserve | 15.74 | 5.42 | 176.5 | 120.8 | transition 진동 |
| MPObs5+CAPS+z + reserve | 24.90 | 30.23 | 253.1 | 152.4 | 완전 실패 |

따라서 Huber/rate reward, z privileged state, MPObs horizon 증가, reserve reward는 현재 프로토콜 아래에서 해결책이 아니었다. 이 ablation들이 무의미한 것은 아니며, 해당 수정만으로는 부족하다는 것을 보여준다.

### 기존 CAPS 구현 문제

기존 구현은 다음과 같은 형태다.

\[
L_T=\lambda_T\lVert\tilde a(s_t)-\tilde a(s_{t+1})\rVert_2,
\qquad
L_S=\lambda_S\lVert\tilde a(s_t)-\tilde a(s_t+\epsilon)\rVert_2
\]

여기서 각 \(\tilde a\)는 서로 독립적으로 sampling한 SAC action이다. 따라서 state가 같아도 exploration sample 차이가 CAPS loss에 들어간다. norm도 sample별 평균이 아니라 batch×action 전체 global norm이다.

실제로 final alpha는 Pure SAC 약 `0.23–0.26`, CAPS 모델 약 `0.74–0.85`였다. 이것은 CAPS가 action stochasticity를 줄이고 entropy tuner가 alpha를 올리는 상충 관계와 일치한다. 인과관계의 최종 증명은 새 pair ablation으로 한다.

새 구현은 deterministic policy mean을 사용한다.

\[
L_T=\lambda_T\,\mathbb E_i
\lVert\mu(s_i)-\mu(s'_i)\rVert_2
\]

\[
L_S=\lambda_S\,\mathbb E_i
\lVert\mu(s_i)-\mu(s_i+\epsilon_i)\rVert_2
\]

CAPS는 warm-up 후 100 episode에 걸쳐 ramp한다. SAC stochastic sampling과 automatic entropy tuning은 그대로 유지한다.

### MPObs의 한계

현재 MPObs1은 plant와 같은 완전한 내부 상태를 이어받는 predictor가 아니다.

- 매 step predictor를 현재 4개 pressure로 다시 초기화한다.
- predictor의 six valve `z/x1/x2`, pump phase, delay buffer는 이어지지 않는다.
- plant delay는 0.1 s지만 predictor delay는 0이다.
- MPObs1의 실제 prediction horizon은 1 step, 즉 20 ms다.
- `num_pred=15`는 future reference 길이이며 model prediction 15 step을 의미하지 않는다.

그러므로 MPObs는 제거하지 않고 새 matrix에 남기되, deploy 가능한 pressure+past-action history와 직접 비교한다.

### 발견된 history-only 설정 오류

기존 `history_sac`는 `num_act=5`로 구성되어 30차원 action을 출력했다. predictor가 없으므로 simulator에는 첫 6개만 적용되고 나머지 24개는 사용되지 않았다. 새 실험에서는 모든 variant를 `num_act=1`, policy action 6개로 통일한다.

### alpha 초기화 오류

config에는 `alpha=0.5`가 저장됐지만 automatic entropy tuning이 켜지면 constructor가 `log_alpha=0`, 즉 실제 `alpha=1.0`으로 시작했다. 새 구현은 `log_alpha=log(0.5)`에서 시작한다.

## lib2는 되고 lib3는 어려운 이유

lib2에서는 두 action이 두 tracked pressure에 비교적 직접적이고 단조롭게 영향을 줬다. valve command에 따른 flow도 1차 선형에 가까워 action 효과가 바로 다음 reward에 나타났다. 이 조건에서는 MPObs가 짧은 delay를 보완하고 CAPS가 두 action의 진동을 줄이는 역할을 하기 쉽다.

lib3는 다음 관계를 동시에 학습해야 한다.

\[
u_{1:2}
\rightarrow P_{ch,+/-}
\rightarrow (u_{3:6},\,z,\,\Delta t)
\rightarrow P_{act,+/-}
\rightarrow r_{t+k}
\]

- 6개 action이 chamber 생성과 actuator 전달로 역할이 나뉜다.
- valve opening은 command에 선형적이지 않고 cracking threshold와 hysteresis state `z`에 의존한다.
- 같은 action이라도 chamber-actuator pressure difference에 따라 효과와 방향이 바뀐다.
- 0.1 s delay 때문에 현재 pressure만으로는 Markov state가 아니다.
- chamber action의 효과가 actuator tracking reward에 지연되어 나타나 credit assignment가 길어진다.

즉 6-action 자체가 불가능한 것이 아니라, 기존 current-pressure MLP와 잘못된 초기 replay/CAPS/carry 조건이 이 시스템의 부분관측성과 delayed credit assignment를 처리하지 못한 것이다.

## reference 진단

오늘 밤에는 reference를 바꾸지 않는다.

- kind: `rate_limited_random`
- seed: `61098`
- configured range: POS `105–150`, NEG `55–85` kPa
- configured rate: POS `±12`, NEG `±6` kPa/s
- first 2048-step measured maximum rate: POS `11.9997`, NEG `5.9996` kPa/s
- discontinuity 없음

동일 reference의 tuned PID 4×512 carry 결과는 POS RMSE `4.82`, NEG RMSE `3.37` kPa였다. 따라서 적어도 첫 2048 step은 폐루프에서 대체로 추종 가능한 과제다. 다만 이 reference는 simulator command replay로 생성된 oracle trajectory가 아니므로 모든 waypoint의 global feasibility를 수학적으로 증명한 것은 아니다.

reference를 함께 바꾸면 구조 ablation의 원인을 해석할 수 없으므로 이번 6개에서는 고정한다. 6개가 모두 방향성조차 실패한 경우에만 다음 단계에서 rate/envelope curriculum을 별도 축으로 둔다.

## 오늘 밤 6개 matrix

세 구조 모두 최근 pressure 10 step(0.2 s)과 future reference 15 step(0.3 s)을 본다. 0.2 s history는 plant delay 0.1 s보다 길다.

| ID | 구조 | Predictor | Past full action | CAPS | state dim | action dim |
|---:|---|---|---:|---|---:|---:|
| 1 | Pressure history SAC | 없음 | 0 | 없음 | 96 | 6 |
| 2 | Pressure history SAC | 없음 | 0 | mean-action CAPS | 96 | 6 |
| 3 | MPObs1 SAC | 1-step | 0 | 없음 | 100 | 6 |
| 4 | MPObs1 SAC | 1-step | 0 | mean-action CAPS | 100 | 6 |
| 5 | Pressure + action history SAC | 없음 | 10×6 | 없음 | 156 | 6 |
| 6 | Pressure + action history SAC | 없음 | 10×6 | mean-action CAPS | 156 | 6 |

관측 길이 계산은 다음과 같다.

- pressure history: `(10+1) × 4 = 44`
- future/current reference: `(10+15+1) × 2 = 52`
- MPObs1 prediction: `1 × 4 = 4`
- action history: `10 × 6 = 60`
- 따라서 `44+52=96`, `44+4+52=100`, `44+60+52=156`

이 matrix가 답하는 질문은 명확하다.

- ID 1 vs 2: pressure history에서 CAPS가 진동을 줄이는가?
- ID 3 vs 4: 현재 MPObs1에서 CAPS가 도움이 되는가?
- ID 5 vs 6: delay를 past action으로 복원할 때 CAPS가 도움이 되는가?
- ID 1 vs 3 vs 5: predictor보다 측정 가능한 action history가 더 유효한가?

## 공통 학습 조건

- lib3 full 6-action, 매 20 ms마다 새 action 출력; action hold 없음
- physical action: `u = 0.8 + 0.2a`
- reference seed: `61098`
- 50 Hz, plant delay 0.1 s
- episode: 1000
- horizon: 512
- physical carry: 4 block까지만 유지 후 hard reset, 즉 2048-step rollout
- episode 11 직전 warm-up state를 버리고 대기압 hard reset
- initial state: 모든 pressure 101.325 kPa
- first 10 episode: physical 0.95 중심 pink-noise, physical `[0.90, 1.00]`
- first 10 episode 동안 actor/critic update 없음
- episode 11 이후 external noise 없음; SAC stochastic sampling은 유지
- observation fixed-scale normalization 사용
- tracking: absolute error, POS/NEG 각각 0.3
- chamber reserve coefficient: 0.05
- chamber envelope coefficient: 0.10
- chamber envelope: POS `[101.325, 220]`, NEG `[10, 101.325]` kPa
- action-delta/rate/conflict/Huber reward: 사용하지 않음
- `gamma=0.99`, `lr=3e-4`, `tau=0.005`
- hidden 256×2, batch 128, replay 500,000
- update 1회/step, target update `tau=0.005`
- automatic entropy tuning, initial `alpha=0.5`, target entropy `-6`
- log std range `[-10, 1]`
- CAPS: temporal 0.5, spatial 0.1, normalized state noise 0.01, 100-episode ramp
- CPU thread: `OMP/BLAS/MKL/NUMEXPR=2`

## preflight 결과

학습은 실행하지 않고 다음을 확인했다.

- 6개 모두 policy action 6, simulator action 6, `num_act=1`
- state dim `96/100/156` 일치
- initial policy mean raw `0.75`, physical `0.95`
- randomized warm-up 5120-step simulator replay 완료
- warm-up physical command range `0.90–1.00`, std `0.0327`
- warm-up CH POS range `102.80–184.50` kPa
- warm-up CH NEG range `15.56–95.43` kPa
- NaN/Inf 없음
- warm-up 후 대기압 reset 확인
- 4-block carry 뒤 hard reset 확인
- reference first 2048-step range/rate 검증 통과

reference plot: `pneu_rl/exp/lib3_reference_preflight/night_260805_seed61098/reference.png`

## 실행 명령

먼저 한 번만 preflight를 실행한다.

```bash
OMP_NUM_THREADS=2 \
/home/rise/miniconda3/envs/env_1/bin/python \
pneu_rl/scripts/preflight_lib3_night_ablation.py \
--seed 0 \
--ref-seed 61098
```

그다음 서로 다른 터미널에서 ID 하나씩 실행한다. launcher 내부에서도 각 BLAS/OMP thread를 2로 제한한다.

```bash
pneu_rl/scripts/run_lib3_night_ablation_0805.sh 1
```

```bash
pneu_rl/scripts/run_lib3_night_ablation_0805.sh 2
```

```bash
pneu_rl/scripts/run_lib3_night_ablation_0805.sh 3
```

```bash
pneu_rl/scripts/run_lib3_night_ablation_0805.sh 4
```

```bash
pneu_rl/scripts/run_lib3_night_ablation_0805.sh 5
```

```bash
pneu_rl/scripts/run_lib3_night_ablation_0805.sh 6
```

학습이 모두 끝난 뒤 단 한 번의 2048-step rollout로 각각 검증한다. 여러 evaluation episode를 carry하는 방식은 사용하지 않는다.

```bash
pneu_rl/scripts/evaluate_lib3_night_ablation_0805.sh
```

## 판정 순서

training reward나 RMSE 하나로 고르지 않는다.

1. **안전성**: chamber runaway 또는 NaN이 있으면 즉시 실패다.
2. **경향성**: `improvement_over_atmosphere > 0`, POS/NEG direction agreement가 모두 0.6 이상인지 본다.
3. **동적 반응**: actual pressure span이 reference 변화에 반응하고 best lag가 0–0.3 s 안에 있는지 본다.
4. **reserve**: positive/negative reserve 10 kPa 충족 비율을 본다.
5. **진동**: pressure max rate와 physical action delta를 PID 및 CAPS-off pair와 비교한다.
6. **tracking**: 위 조건을 통과한 모델끼리 RMSE/MAE/tolerance를 비교한다.

CAPS 성공 기준은 같은 observation의 CAPS-off 모델보다 action/pressure 진동이 줄고, POS/NEG RMSE가 10% 이상 악화되지 않는 것이다. flat pressure로 rate만 낮추면 실패다.

## 결과별 다음 단계

### ID 5/6이 가장 좋음

past action이 hidden valve/delay state 복원에 중요하다는 증거다. 다음에는 MLP 10-step flatten 대신 GRU/LSTM encoder로 pressure+action history를 압축하고, 동일 protocol에서 recurrent SAC를 비교한다.

### ID 3/4가 가장 좋음

model prediction이 유효하다는 뜻이다. 다음에는 predictor의 valve state와 delay buffer를 plant로부터 이어받는 stateful MPObs, 또는 predictor ensemble/uncertainty를 연구한다.

### CAPS-on만 악화

MPOBS를 제거할 이유는 아니고 CAPS objective/weight 문제다. spatial/temporal loss, alpha, policy std 곡선을 확인하고 `0.5/0.1`을 더 낮추거나 ramp를 늘린다.

### 세 구조 모두 방향성 실패

reward tuning을 더 반복하기 전에 구조를 바꾼다.

- GRU/LSTM recurrent SAC
- asymmetric actor-critic: actor는 deploy 가능한 pressure/action history, critic은 simulator의 z/q/mdot를 privileged state로 사용
- 4-pressure target: actuator reference에서 chamber reserve reference를 파생해 auxiliary value/reward 또는 multi-task critic으로 사용
- reference rate/envelope curriculum: 쉬운 dynamic reference에서 시작해 현재 seed로 확장

allocator나 action hold는 이번 단계의 우선안이 아니다. policy는 계속 50 Hz에서 6개 valve action을 직접 출력한다.

### 방향성은 있으나 exploration이 부족함

먼저 `alpha`, log-std, saturation, replay action coverage를 확인한다. 그 증거가 있을 때만 episode 중간 pink-noise burst 또는 reference-conditioned exploration을 별도 ablation으로 추가한다. SAC가 이미 stochastic policy이므로 근거 없이 random action을 중간에 섞는 것을 표준이라고 부를 수는 없다.

## 연구 해석

이번 수정은 MPObs와 CAPS를 버리는 실험이 아니다. Pure SAC pair는 각 구성요소의 기여를 분리하기 위한 control group이다. 핵심 연구 질문은 다음과 같다.

> Nonlinear valve hysteresis와 transport delay가 있는 two-stage pneumatic system에서, model-predicted observation과 measurable action-history state 중 어느 것이 delayed action effectiveness를 더 잘 복원하며, entropy-compatible CAPS가 그 정책의 진동을 억제할 수 있는가?

이 질문은 lib1/lib2의 선형·직접 pressure control 연구에 대한 자연스러운 후속 연구가 된다.
