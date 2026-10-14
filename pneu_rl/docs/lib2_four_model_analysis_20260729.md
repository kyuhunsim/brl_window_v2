# lib2 4개 모델 결과 분석 — 2026-07-29

## 평가 조건

- Reference: `pneu_rl/config/reference/lib2_joint_reachable.yaml`
- 초기 압력: POS/NEG 모두 대기압 101.325 kPa
- Action range: 0.8–1.0
- 평가 horizon: 2048 step = 약 40.96 s
- `screen`과 `quick viz` 모두 동일 reference를 사용
- I-term 모델은 평가에서도 I-term을 켬

## Screen 결과

| 모델 | POS RMSE | NEG RMSE | POS 성공률 | NEG 성공률 | POS bias | NEG bias |
|---|---:|---:|---:|---:|---:|---:|
| SAC, not carry | 5.01 | 5.89 | 87.1% | 95.6% | -1.37 | +0.95 |
| SAC + I-term, not carry | 6.92 | 6.43 | 75.0% | 87.9% | -1.84 | -3.23 |
| SAC, carry | 5.07 | 6.15 | 87.1% | 90.2% | +1.31 | +1.08 |
| SAC + I-term, carry | 7.27 | 6.48 | 61.2% | 93.7% | -1.73 | -3.25 |

Reference의 joint rate violation은 네 모델 모두 0이었다. 따라서 이번 screen에서는 reference 자체가 simulator의 rate limit을 직접 위반해서 실패한 것은 아니다.

## 개형 및 제어명령 해석

### Pure SAC

두 pure SAC 모두 POS/NEG가 reference의 상승·하강 방향을 따라간다. 작은 진동은 남아 있지만, 제어명령이 변화하면서 압력도 실제로 움직인다. POS와 NEG 사이의 균형도 유지된다.

carry pure SAC은 단일 40.96초 평가에서는 not-carry와 거의 같은 수준이었다. 따라서 현재 결과만 보면 `carry=False` 자체가 pure SAC을 망가뜨린 증거는 없다.

### SAC + I-term

I-term 모델은 제어명령과 압력 변화가 훨씬 매끄럽다. 그러나 이번 실험에서는 이것이 장점으로 이어지지 않았다. POS 제어가 0.97–1.0 부근에 몰리면서 reference 변화에 필요한 조작을 충분히 하지 못했고, POS가 reference보다 높거나 낮은 상태로 머무는 구간이 커졌다.

즉 이번 I-term 결과는 “진동 제거 성공”이라기보다 “반응성 감소 및 steady-state bias 증가”에 가깝다. 특히 carry I-term은 POS action saturation 비율이 최근 100개 학습 진단에서 약 26.5%였고, 제어명령이 사실상 상단에 붙는 구간이 많았다.

## Carry 연속 평가

단일 screen은 매번 대기압에서 시작하므로 carry 학습의 실제 상태 분포를 완전히 재현하지 못한다. 그래서 최종 모델을 carry 방식으로 512 step씩 10개 episode 연속 평가했다.

- Carry pure SAC: 첫 episode는 POS RMSE 9.06, NEG RMSE 11.21로 흔들렸지만 이후 대부분 POS 1.2–4.0, NEG 1.6–4.1 수준으로 안정화됐다.
- Carry I-term: POS RMSE가 episode별로 2.18–9.07, POS bias가 대부분 +3–8 kPa로 나타났다. NEG는 비교적 양호하지만 POS가 reference보다 높은 방향으로 치우쳤다.

따라서 carry pure SAC은 상태가 이어지는 조건에서도 회복 가능성이 보이지만, 첫 episode와 초기 상태 민감도가 있다. carry I-term은 안정적으로 보이는 대신 잘못된 압력 operating point에 머무르는 경향이 더 강하다.

## 학습 진단

최종 학습 episode의 단일 값만 보면 carry 모델은 POS RMSE가 약 40 kPa까지 악화된 episode가 있었다. 반면 fresh-reset screen은 양호했다. 이것은 모순이라기보다 평가 조건이 다르다는 뜻이다.

- not-carry: 매 episode가 같은 초기 조건이라 train diagnostic과 screen이 비교적 일치
- carry: 이전 episode의 chamber pressure가 다음 episode의 초기 조건이므로 episode별 상태 분포가 달라짐
- 따라서 carry 모델은 마지막 episode 한 개의 reward나 fresh-reset screen 하나만으로 판단하면 안 됨

Reward는 horizon과 step 수에 비례하므로 모델 간 비교의 1차 지표로 쓰지 않고, RMSE·MAE·bias·성공률·압력 개형·제어명령·flow를 함께 봐야 한다.

## Exploration / alpha 진단

- not-carry pure SAC: 최근 alpha 평균 약 0.96, action saturation 약 2.9%
- carry pure SAC: 최근 alpha 평균 약 0.63, action saturation 약 11.2%
- not-carry I-term: 최근 alpha 평균 약 0.31
- carry I-term: 최근 alpha 평균 약 0.16, action saturation 약 26.5%

I-term 모델은 alpha가 낮고 action이 상단에 붙는 경향이 있어, 탐색 부족 또는 I-term/feed-forward가 policy action의 유효한 영향력을 압도하는 현상이 의심된다. 다만 alpha 하나만으로 원인을 확정할 수는 없다.

## 현재 결론

1. 이번 4개 결과에서 가장 좋은 기준 모델은 pure SAC이다.
2. I-term은 진동은 줄였지만 POS 반응성과 균형 추종을 악화시켰다.
3. 현재 joint reference는 rate feasibility는 만족한다. 다만 rate feasible이 곧 모든 초기 상태에서 쉽게 제어된다는 뜻은 아니다.
4. carry는 시간 절약 옵션이 아니라 초기 상태 분포를 바꾸는 학습 조건이다.
5. carry 모델은 반드시 연속 episode 평가를 해야 한다.
6. 다음 ablation의 우선순위는 `reference/초기상태 → action response/saturation → CAPS 크기 → MPOBS` 순서가 맞다. 지금 단계에서 MPOBS를 먼저 의심할 근거는 부족하다.

## 산출물

각 모델 폴더의 `screen_eval_*`에 `metrics.yaml`, `trajectory.png`, `trajectory.csv`가 저장되어 있다. 같은 폴더의 timestamp가 붙은 `*_Simulation/*.png`가 quick viz 결과다.

## 추가 ablation 결과 — CAPS/MPOBS

추가 세 모델은 모두 `reference_reset_to_observation=false`, `carry=false`, horizon 2048, action dimension 2 조건으로 다시 평가했다. CAPS는 temporal 0.1, spatial 0.08을 사용했다.

| 모델 | POS RMSE | NEG RMSE | POS 성공률 | NEG 성공률 |
|---|---:|---:|---:|---:|
| SAC + CAPS | 5.58 | 5.80 | 82.7% | 97.8% |
| SAC + MPOBS1 + CAPS | 5.53 | 6.64 | 81.6% | 79.3% |
| SAC + MPOBS1 + CAPS + I-term | 7.81 | 6.71 | 50.7% | 70.8% |

### 해석

- 낮춘 CAPS는 pure SAC의 성능을 붕괴시키지 않았다. POS는 약간 나빠졌지만 NEG는 개선되었고, 압력 진동도 소폭 감소했다. 현재 CAPS 세기는 주된 실패 원인으로 보이지 않는다.
- MPOBS1은 action dimension을 2로 유지했음에도 NEG 성공률을 95.6%에서 79.3%로 낮췄다. MPObs의 prediction/observation 처리 또는 delay alignment가 lib2에서는 이득을 주지 못했다.
- MPOBS1+CAPS+I-term은 제어명령이 0.97–1.0 근처에 몰리고 POS가 reference보다 높게 남는 구간이 커졌다. 매끄러움은 증가했지만 tracking 성능은 가장 나빴다.
- 세 모델 모두 reference joint rate violation은 0이었다. 따라서 이번 ablation에서 성능 차이는 reference rate 위반보다는 policy 구조와 보조항의 영향으로 해석하는 것이 타당하다.

추가 quick viz의 30–90초 구간 RMSE는 각각 SAC+CAPS `POS 5.47 / NEG 1.29`, MPOBS1+CAPS `POS 6.01 / NEG 1.58`, MPOBS1+CAPS+I-term `POS 10.00 / NEG 4.47`이었다. 따라서 전체 RMSE뿐 아니라 실제 개형에서도 MPOBS와 I-term 조합의 열화가 확인된다.
