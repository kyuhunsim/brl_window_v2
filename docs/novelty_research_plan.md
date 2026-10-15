# lib4 논문 연구 방향과 다음 실험

작성일: 2026-09-01  
상태: lib3 reward × observation seed-0 2×2 완료; lib4 real validation 전

> 수식이 `\sigma`, `\dot`, `\[`처럼 그대로 보이면 현재 Markdown viewer가
> MathJax/KaTeX를 지원하지 않는 것입니다. 렌더러와 무관하게 읽을 수 있는
> [plain 수식 요약](formula_reference_plain.md)을 먼저 참고하세요.

이 문서는 현재 연구를 “RL을 성공시켰다”에서 끝내지 않고, 어떤 질문과
실험으로 논문 기여를 만들 수 있는지 정리한 실행용 문서다. 기존 상세
분석은 [research novelty review](research_novelty_review.md),
[교수님 설명용 observation argument](professor_observation_argument.md),
[lib4 ablation 계획](ablation/lib4/README.md)을 함께 참조한다.

## 1. 현재 질문에 대한 결론

### History-only가 reward 수정으로 해결될 수 있는가?

현재 결과는 **부분적으로만 그렇다**.

같은 H observation에서 smooth reward를 넣으면 다음 변화가 있었다.

- POS/NEG RMSE: `8.238/7.394 → 6.576/4.920 kPa`
- plateau pressure-rate RMS: `30.290/15.897 → 11.506/3.372 kPa/s`
- plateau action-Δ RMS: `.00542 → .00174`

그러나 H-smooth의 combined RMSE는 `5.834 kPa`로, 같은 smooth reward의
HR-smooth `1.025 kPa`보다 여전히 훨씬 크다. 대표 궤적에서도 H-smooth는
움직임은 부드럽지만 reference amplitude를 충분히 따라가지 못한다.

따라서 현재 2×2가 지지하는 문장은 다음이다.

> 기존 reward는 H policy의 pressure oscillation을 허용하는 failure mode를
> 만들었다. 그러나 reward만 수정해도 finite-budget MLP-SAC가 history에서
> rate-relevant control law를 안정적으로 학습하는 문제까지 해결되지는
> 않는다.

동일 reward에서 HR-smooth가 H-smooth보다 combined RMSE가 약 82.4% 낮다는
점은 특히 중요하다. reward misalignment가 전부라면 두 policy의 격차가
사라졌어야 한다. 따라서 explicit rate는 새 정보가 아니더라도 학습·조건화·
수렴 안정성에 강한 coordinate bias를 제공한다고 해석하는 것이 현재 가장
정확하다.

### Offset은 reward에 없어서 생긴 문제인가?

steady-state offset에 대해서는 단순히 다음 항을 추가하는 것만으로는
새로운 정보가 생기지 않는다.

\[
r_{\rm base}=-w_e\sum_c|e_c|,
\qquad e_c=P_{{act},c}-r_c.
\]

상수 offset (e_c\neq0)는 이미 매 step마다 base reward를 계속 감소시킨다.
따라서 `-lambda_offset * |e|`를 전체 구간에 더하는 것은 offset 문제를
새로 관측하는 것이 아니라 tracking weight를 키우는 것뿐이다. 이것은
transition 응답을 희생할 수 있다.

offset을 정말 별도로 검사하려면 reference ramp와 hold를 분리한 plateau
term을 사용해야 한다.

\[
\chi_t=\mathbf 1\{ |\dot r_{+,t}|<\epsilon_+,\ |\dot r_{-,t}|<\epsilon_-\},
\]

\[
r_{\rm plateau\text{-}bias}
=-\lambda_b\chi_t\sum_c\frac{|e_{c,t}|}{s_{e,c}}.
\]

이 항은 ramp에서 필요한 tracking을 방해하지 않고 hold 구간의 offset만
상대적으로 강조한다. 단, 이것도 정보 추가가 아니라 objective weighting과
credit assignment를 바꾸는 실험이다.

### 진동은 어떤 reward 항으로 검사하는가?

현재 rate reward는 이미 reference-aware 진동 항이다.

\[
r_{\rm rate}
=-\lambda_r\sum_c
\left|\frac{\dot P_{{act},c}-\dot r_c}{s_{r,c}}\right|.
\]

raw \(|\dot P|\)를 모든 구간에 벌하면 ramp 자체를 늦추므로 사용하지 않는다.
plateau에서는 \,\(\dot r=0\)이어서 pressure ripple을 직접 줄이는 효과가
있다.

action chatter는 pressure ripple과 다른 현상이므로 별도 항이다.

\[
r_{\Delta u}=-\lambda_u\|u_t-u_{t-1}\|_1,
\qquad
r_{\Delta^2u}=-\lambda_{u2}\|u_t-2u_{t-1}+u_{t-2}\|_1.
\]

첫 번째는 valve command 변화량, 두 번째는 command jerk를 줄인다. 이 항을
동시에 넣으면 reward intervention이 세 개가 되므로 rate/offset 실험과
분리한다.

## 2. 권장 reward 실험: 한 번만, 작게

이미 H/HR × base/smooth를 완료했으므로 reward만 계속 sweep하지 않는다.
offset과 진동의 원인을 확인하는 **H-only 2×2 diagnostic**만 추가한다.

| | rate term 없음 | phase-aware rate term |
|---|---|---|
| plateau bias 없음 | H-base | H-rate |
| plateau bias 있음 | H-bias | H-rate+bias |

공통 조건은 기존 2×2와 동일하게 하고, 다음만 추가한다.

\[
r_t=r_{\rm base}
-\lambda_r\sum_c\left|\frac{\dot P_{act,c}-\dot r_c}{s_{r,c}}\right|
-\lambda_b\chi_t\sum_c\frac{|e_{c,t}|}{s_{e,c}}.
\]

실험 순서는 `seed 0, 2000 episode`로 충분하다. (lambda_b)는 먼저
plateau base penalty의 5–10%가 되도록 scale audit 후 고정한다. 기존
(lambda_r=0.025)는 유지한다.

판정은 reward return이 아니라 다음 Pareto 조건으로 한다.

- plateau rate RMS 감소
- plateau bias/MAE 감소
- 전체 및 transition RMSE 악화 없음
- reserve violation 증가 없음

예상 가능한 결과와 해석은 다음과 같다.

| 결과 | 결론 |
|---|---|
| bias term만 offset을 줄이고 RMSE는 유지 | 기존 reward의 plateau credit assignment 부족 |
| rate term만 진동을 줄임 | oscillation은 rate objective 문제 |
| 둘 다 H를 HR 수준으로 못 올림 | observation/optimization 문제가 남음; reward 가설 종료 |
| bias가 RMSE를 악화 | offset term 폐기; base tracking으로 충분 |

이 실험 이후에는 reward 계수 탐색을 중단한다. H가 계속 HR과 큰 차이를
보이면 더 많은 reward 항을 붙이는 대신 representation과 dynamics 쪽으로
이동한다.

## 3. 현재까지 확보한 인과적 관찰

### 3.1 Rate는 정보가 아니라 coordinate bias다

pressure history를 (H_t), rate transform을 (D_t=LH_t)라 하면

\[
\sigma(H_t,D_t)=\sigma(H_t).
\]

따라서 explicit rate가 원리적으로 표현력을 늘리거나 Markov state를
만들었다고 주장할 수 없다. 첫 MLP layer에서도

\[
W_HH_t+W_DD_t=(W_H+W_DL)H_t
\]

로 정확히 folding할 수 있다. 현재 실험에서 남는 해석은 finite network,
normalization, critic learning, exploration이 pressure trend coordinate를
직접 받을 때 더 안정적으로 학습한다는 것이다.

### 3.2 Reward가 달라도 rate coordinate 효과는 남았다

HR-smooth와 H-smooth는 동일한 reward를 사용했는데도 combined RMSE가
`1.025` 대 `5.834 kPa`였다. 따라서 “H 실패는 reward가 진동을 보상하지
않았기 때문”이라는 설명은 일부만 맞고 충분조건이 아니다.

### 3.3 HR smooth의 채널 trade-off

HR-base → HR-smooth는 NEG RMSE와 NEG plateau rate를 개선했지만 POS RMSE와
POS plateau rate는 악화시켰다. combined RMSE는 `1.047 → 1.025 kPa`로
거의 변하지 않았다. 이는 rate reward가 전역적으로 모든 채널을 개선했다기
보다 shared source의 allocation을 재조정했음을 뜻한다.

### 3.4 학습 안정성

H 계열은 약 1200–1300 episode에서 좋아졌다가 2000 episode에서 악화했다.
HR 계열은 마지막까지 낮은 training RMSE를 유지했다. 그러므로 rate의
현재 역할은 “필수 센서”보다 **finite-budget convergence/stability bias**에
가깝다.

## 4. 다음에 시도할 후보와 우선순위

### P0 — 바로 할 것

#### A. (\dot e) coordinate

\[
\dot e=\dot r-\dot P.
\]

plateau에서는 (-\dot P)와 같지만 ramp에서는 desired motion과 actual
motion을 분리한다. 현재 reward가 바로 \,\(|\dot e|\)를 벌하므로 가장
자연스러운 추가 비교다.

비교는 `H+rate` 대 `H+e_dot`로 하고, 둘을 동시에 넣지 않는다. 같은
normalization과 parameter budget을 사용한다.

#### B. GRU-SAC 하나

LSTM과 GRU를 모두 할 필요는 없다. GRU는 history를 policy가 직접 압축할 수
있는 강한 baseline이다. 결과가 H+rate보다 좋으면 fixed feature claim을
약화하고, GRU가 비슷하면 explicit coordinate의 저비용 장점을 강조한다.

#### C. PIDE 또는 GPIDE baseline

NeurIPS 2023 PIDE/GPIDE는 history에서 PID-like sum/difference coordinate를
만드는 선행법이다. 이것은 네 method가 아니라 직접 비교해야 할 baseline이다.
먼저 tracking-oriented PIDE 하나를 구현하고, GPIDE는 여유가 있을 때만
추가한다.

### P1 — lib4 main experiment

최종 representation 후보를 다음과 같이 좁힌다.

1. current/short history baseline
2. GRU
3. PIDE
4. history + \(\dot e\) + reserve
5. predecessor-style MPObs + CAPS

각 모델은 nominal sim만이 아니라 valve time constant, hysteresis, sensor
delay, actuator volume, payload를 바꾼 mismatch simulation과 real zero-shot,
동일 budget fine-tuning에서 평가한다.

### P2 — replacement claim

네 모델이 정말 MPObs/CAPS를 대체한다고 쓰려면 동일한 HRR base 위에서 다음
four-way를 추가한다.

| 모델 | 목적 |
|---|---|
| HRR | proposed minimal representation |
| HRR + MPObs | future prediction의 추가 이득 |
| HRR + CAPS | explicit policy smoothing의 추가 이득 |
| HRR + MPObs + CAPS | predecessor-style full stack |

nominal에서 비슷하고 mismatch/real에서 HRR이 더 좋으면 “model-free realized
dynamics가 model-predicted observation보다 robust”라는 강한 결과가 된다.

### P3 — lambda discrepancy

lambda discrepancy는 새 controller가 아니라 value-relevant Markovness
diagnostic이다. Current, H, GRU, PIDE, HRR, privileged representation의
동일 fresh rollout에 적용한다.

- discrepancy 감소 + control 개선: approximate information state 근거
- discrepancy 동일 + HRR control 개선: optimization bias 근거
- discrepancy 감소 + control 동일: representation은 좋아졌지만 practical
  gain 없음

SAC auxiliary loss로 바로 이식하지 말고, 먼저 offline 분석으로 사용한다.

### P4 — 가장 강한 novelty 후보: authority

최종 후보가 안정화된 후에만 short-horizon action sweep으로

\[
\mathcal A_K(x_t)=
\left\{\frac{y_{t+K}-y_t}{K\Delta t}\right\}
\]

를 추정한다. 같은 tracking error라도 reserve, velocity, pressure level과
valve history에 따라 가능한 output-rate가 달라지는지 검사한다.

raw reserve보다 authority estimate가 held-out failure와 real violation을
더 잘 예측하고, 그 estimate를 reference certification 또는 policy
conditioning에 실제 사용했을 때만 method-level novelty로 승격한다.

## 5. 하지 말아야 할 것

- LSTM과 GRU를 동시에 돌려 표만 늘리기
- offset term을 base error와 똑같이 전체 구간에 추가하기
- rate, action penalty, integral, CAPS를 한 번에 넣기
- reward return만 보고 smooth arm을 성공으로 부르기
- 한 seed 결과로 “필수 observation”이라고 단정하기
- K=2를 displacement RMSE만 보고 재채택하기
- BiPneu와 겹치는 “bipolar + variable volume + real control”만을 novelty로
  주장하기

BiPneu는 bipolar-pressure hardware, hybrid model, DM-SMC와 real demonstration을
이미 종합한 T-Mech 논문이다. 따라서 네 논문은 시스템 복잡성보다 `왜 이
system에서 이 representation이 충분한가`를 중심으로 써야 한다.

## 6. 논문용 중심 결론 후보

현재 결과가 lib4/real에서도 반복되면 다음 문장을 사용할 수 있다.

> We show that a real-measurable history and pressure-authority representation
> can replace model-predicted observations and explicit policy-smoothing
> regularization for closed-loop control of a shared dual-pressure,
> variable-volume soft actuator. Pressure-error rate provides the dynamic
> tracking coordinate, while chamber reserve constrains the learned policy to
> feasible pneumatic authority. The benefit is not additional information—rate
> is derivable from history—but a physics-structured coordinate bias that
> improves finite-budget learning and robustness under simulator mismatch.

이 문장은 다음 조건을 만족할 때만 사용한다.

1. lib4 variable-volume simulator의 pressure/length/velocity replay가 real과
   control-relevant하게 일치한다.
2. HRR이 GRU/PIDE 및 predecessor-style MPObs+CAPS와 nominal/mismatch/real에서
   비교된다.
3. rate와 reserve의 tracking/safety 역할이 ablation으로 분리된다.
4. real zero-shot 결과와 동일 budget fine-tuning 결과가 모두 기록된다.
5. rate가 정보적으로 필수라고 주장하지 않고, value/policy 결과에 대한
   finite optimization bias로 제한한다.

현재 결과만으로는 “H가 reward 수정으로 해결된다”가 아니라,
**reward는 진동의 일부를 설명하고 rate coordinate는 학습 안정성을 설명한다**가
가장 방어 가능한 결론이다.

## 7. 실험 코드 세팅 (2026-09-02, lib3 우선)

실험 순서는 lib3에서 먼저 고정한다. lib3가 현재 observation/reward ablation과
이미 축적된 seed 결과를 가진 기준 시스템이기 때문이다. lib4의 동일 feature mode는
후속 이식 확인용이며, lib4 결과를 lib3 결론의 근거로 섞지 않는다.

lib3 matched run은 다음으로 준비한다.

이 설정은 `episode horizon=512`, `carry=true`, `carry_reset_blocks=4`로 기존
`260831` H/HR reward 실험과 맞춘다. 따라서 episode 하나당 512 policy step,
1000 episode당 512,000 step이다.

```bash
DRY_RUN=1 bash pneu_rl/scripts/run_lib3_structured_history.sh
# 실제 학습: EPISODES=2000 bash ...
```

첫 screening을 두 터미널로 나누려면 다음처럼 실행한다. 각 터미널은 두 job을
순차 실행하므로 CPU simulator contention을 줄이면서 전체 wall time을 줄일 수
있다.

```bash
# terminal 1
EPISODES=1000 MODES=hrr,pide bash pneu_rl/scripts/run_lib3_structured_history.sh

# terminal 2
EPISODES=1000 MODES=gpide,edot bash pneu_rl/scripts/run_lib3_structured_history.sh
```

`rate_reserve_edot`은 첫 2×2의 필수 arm이 아니라 HRR과 error-derivative의
interaction을 확인하는 후속 arm으로 둔다.

현재 lib3의 GPIDE 고정 feature는 generalized learned encoder가 아니다. pressure
observation만 있는 lib3에서는 GPIDE의 `delta observation` 4개가 사실상
`pressure difference`, 즉 `p_dot`와 같은 방향이고, current/Delta action은 이미
`history_action_sac`의 action history와 겹친다. 따라서 GPIDE는 primary novelty
arm이 아니라 optional supplementary baseline으로 둔다.

### 네 feature의 실제 차이

| feature | lib3 코드에서 실제로 들어가는 값 | 주 질문 |
|---|---|---|
| HRR | 4개 pressure rate + 2개 chamber reserve | realized plant trend와 pneumatic headroom이 도움이 되는가 |
| e_dot | 2개 tracking-error derivative | 목표 대비 error가 줄어드는 방향인지 직접 주면 좋은가 |
| PIDE | 2개 error + 2개 finite-window integral + 2개 e_dot | PID식 tracking summary가 history보다 유리한가 |
| GPIDE-lite | 4개 observation difference + 6개 current action + 6개 action difference | generalized difference/action 좌표가 추가 이득을 주는가 |

따라서 `GPIDE-lite`의 첫 4개는 `p_dot`와 본질적으로 같은 difference다.
차이는 sample-frequency와 normalization, 그리고 action 좌표가 추가된다는 점뿐이다.
현재 raw action history도 함께 사용하므로 action 부분도 중복된다. 결론적으로
GPIDE는 HRR과 경쟁하는 독립적인 물리 가설이 아니라, `history + p_dot + 중복
action 좌표`를 확인하는 보조 대조군이다.

lib3에서의 차원은 HRR=6, PIDE=6, GPIDE=16이다. GPIDE의 16개는 4개 pressure
difference + 6개 current valve command + 6개 command difference로 구성된다.
lib4는 displacement/velocity까지 포함하므로 GPIDE가 18차원이지만, 두 library의
차원 차이는 센서 정의 차이이지 feature 의미의 차이가 아니다.

기존 lib4 script도 남겨두었지만, lib3 결과가 정리되기 전에는 lib4 structured
history를 논문 ablation 표의 최종 결과로 사용하지 않는다.

lib4 runner에 구조화 history feature를 추가했다. 기존 `history_rate_reserve`와
동일한 계산을 짧게 부르는 `hrr` alias를 제공하며, 세 baseline은 다음 명령으로
동일 simulator/reference/reward 조건에서 실행한다.

```bash
bash pneu_rl/scripts/run_lib4_structured_history.sh
```

실제 학습 전 shape와 finite reward만 확인하려면:

```bash
DRY_RUN=1 bash pneu_rl/scripts/run_lib4_structured_history.sh
```

runner가 지원하는 feature map은 다음과 같다.

| variant | feature (z_t) | 차원 | 해석 |
|---|---|---:|---|
| `hrr` | ([dot P_{ch},dot P_{act},P_{ch+}-P_{act+},P_{act-}-P_{ch-}]) | 6 | proposed history-rate-reserve |
| `pide` | ([e_t,sum_i e_iDelta t,dot e_t]) | 6 | tracking P/I/D summary |
| `gpide` | ([Delta o_t,u_t,Delta u_t]) | 18 | generalized difference/action summary |

여기서 (e_t=r_t-P_{act,t}), (dot e_t=(e_t-e_{t-1})f_s)이다. GPIDE의
reward 입력은 transition 뒤에 계산되어 현재 policy state에 넣으면 leakage가
되므로 제외했다. 세 feature 모두 simulator hidden state를 읽지 않고 현재
환경이 가진 observation/action history에서 결정적으로 계산된다. 따라서 이
코드는 “추가 센서 정보”가 아니라 representation/optimization bias 비교다.

PIDE/GPIDE는 논문의 모든 learned recurrent encoder를 복제한 것이 아니라, 해당
논문의 핵심 sum/difference inductive bias를 고정 feature로 검증하는 첫 단계다.
공정한 비교를 위해 history 길이, normalization, hidden dimension, reward,
reference bank, seed를 고정하고 `cfg.yaml`의 `structured_history` 필드를 결과와
함께 보관한다.

평가 시 `evaluate_lib4_screening.py`는 각 validation row에 deterministic twin-Q
value를 `value` 열로 기록한다. 그 trace로 TD(\(\lambda\)) discrepancy를 계산한다.

```bash
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/analyze_lambda_discrepancy.py \
  pneu_rl/exp/<evaluation-root>/260902_lib4_hrr_seed0 \
  --output pneu_rl/exp/<evaluation-root>/lambda_discrepancy_hrr.json
```

기본값은 TD(0)와 TD(0.95)의 차이다. `mean_abs`와 `normalized_mean_abs`가 낮다는
것은 해당 policy/reward 분포에서 representation이 value prediction에 더
Markov-like하다는 뜻이지, 그 policy의 폐루프 RMSE가 더 낮다는 뜻은 아니다.
서로 다른 policy의 validation CSV를 직접 순위화하지 말고, 가능하면 하나의
frozen policy와 동일 rollout에서 state representation만 바꿔 비교한다. 현재
evaluator의 K=1 trace에서는 정확하고, K>1에서는 policy-decision value가 physical
sub-step에 반복 기록되므로 별도 K 분석으로 표시한다.
