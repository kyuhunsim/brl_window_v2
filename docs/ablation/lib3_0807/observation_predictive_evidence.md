# lib3 observation의 물리적·예측적 근거 검증

작성일: 2026-08-24

## 먼저 답할 질문: predictor는 무엇인가?

이 문서에서 사용하는 predictor는 **lib3 simulator도 아니고 학습된 RL policy도 아니다.** 기존 RL 학습에서 저장된 transition을 이용해 별도로 학습하는 작은 supervised neural network다.

| 구성요소 | 입력과 출력 | 역할 |
|---|---|---|
| lib3 simulator | valve action과 내부 상태 → 다음 실제 압력 | 데이터를 만드는 plant/environment |
| SAC policy | policy observation → 6개 valve action | 제어기 |
| 진단용 predictor | 선택한 observation과 실제 action sequence → 이후 네 압력 | observation이 미래 dynamics를 얼마나 설명하는지 검사 |

진단용 predictor는 제어 명령을 만들지 않으며 simulator 식을 대체하지도 않는다. 다음 supervised regression만 수행한다.

\[
\hat f_\theta(x_t,u_{t:t+h-1})
\rightarrow
\Delta \hat P_{t,h}=\hat P_{t+h}-P_t
\]

여기서 \(x_t\)의 구성만 `current`, `history`, `action history`, `rate/reserve`로 바꾸고, dataset·target·future action sequence·network width·optimizer를 같게 유지한다. 따라서 RL return보다 직접적으로 다음 질문을 검사한다.

> 이 observation 구성은 동일한 control input 아래에서 나타나는 미래 압력 차이를 구별하는 데 도움이 되는가?

## 교수님 피드백을 연구 질문으로 번역

교수님 피드백은 다음과 같이 정리한다.

> Orifice model을 만족하도록 PINN을 구성하는 것처럼, 지금 추가한 observation이 우연히 성능을 높인 것인지 실제 pneumatic dynamics를 표현하는지 확인할 필요가 있다. 각 observation이 시스템 이해와 제어에 왜 필요한지 어떻게 검사할 것인가?

이를 네 개의 반증 가능한 질문으로 나눈다.

1. 현재 네 압력만으로는 미래 압력을 충분히 예측하지 못하고 pressure/action history가 필요한가?
2. `rate`와 `reserve`는 새로운 정보를 제공하는가, 아니면 이미 있는 정보를 학습하기 쉬운 좌표로 제공하는가?
3. 이 좌표가 dynamics prediction뿐 아니라 multi-seed RL 성능과 unseen reference 일반화도 개선하는가?
4. 같은 apparent actuator state와 valve pulse에서 reserve 또는 진행 방향을 실제로 바꾸면 future response가 달라지는가?

필요한 증거의 연결은 다음과 같다.

```text
공압식과 valve delay
→ matched-state/pulse response
→ future-pressure prediction
→ multi-seed RL ablation
→ lib4 및 real-system 재현
```

Predictor 결과 하나만으로 물리적 인과성이나 수학적 observability가 증명되지는 않는다. 반대로 RL return 하나만으로도 observation의 물리적 필요성을 증명할 수 없다. 두 결과와 intervention을 함께 사용해야 한다.

## 현재 lib3 observation에 대한 코드 audit

2026-08-24에 다음 모델의 저장 replay buffer 500,000 transition을 검사했다.

```text
pneu_rl/models/260810_lib3_ablation_profile_hist_rate_reserve_seed0
```

현재 observation 구성은 다음과 같다.

| 부분 | 차원 | 정의 |
|---|---:|---|
| pressure history | 44 | 11 step × `[CH_POS, CH_NEG, ACT_POS, ACT_NEG]` |
| action history | 60 | 10 step × 6 valve actions |
| rate | 4 | 최근 두 pressure sample의 50 Hz finite difference |
| reserve | 2 | `CH_POS-ACT_POS`, `ACT_NEG-CH_NEG` |
| reference preview | 52 | 26 step × POS/NEG actuator reference |
| 합계 | 162 | 44+60+4+2+52 |

실제 feature 정의는 `pneu_env/src/pneu_env/env3.py`의 `_get_physics_features`에 있다.

\[
rate_t=50(P_t-P_{t-1})
\]

\[
reserve_t=
\begin{bmatrix}
P_{ch,+}-P_{act,+}\\
P_{act,-}-P_{ch,-}
\end{bmatrix}
\]

replay에서 균등하게 선택한 2,000개 sample을 재계산한 결과는 다음과 같다.

| Audit | 최대 재구성 오차 |
|---|---:|
| pressure history → stored rate | 0.0 |
| current pressure → stored reserve | 0.0 |

따라서 lib3 History SAC에서는 `rate`와 `reserve`가 information-theoretically 새로운 변수가 아니다.

- rate는 pressure history의 deterministic transformation이다.
- reserve는 현재 네 pressure의 deterministic linear transformation이다.
- `History+rate+reserve`가 더 잘 됐다는 사실만으로 partial observability가 새롭게 해소됐다고 주장하면 안 된다.

정확한 가설은 다음과 같다.

> Pressure/action history는 delayed valve dynamics에서 누락된 temporal context를 제공하고, explicit rate/reserve는 그 정보에서 realized pressure response와 pressure headroom을 바로 읽을 수 있도록 만든 physics-informed inductive bias다.

## 왜 pressure/action history에는 Markov 근거가 있는가?

lib3 simulator에는 policy가 직접 관측하지 않는 valve dynamic state, flow와 0.1 s delay가 있다. `pneu_env/src/pneu_env/lib3/pneumatic_CT.cpp`에서는 다음 관계가 구현되어 있다.

\[
\dot P=-\frac{P\dot V}{V}+\frac{RT}{V}\dot m
\]

mass flow는 inlet/outlet pressure, valve command뿐 아니라 내부 valve state가 결정하는 effective area와 predicted flow에 의존한다. 구현상 `force_net`, valve state `z`, `area_eff`, `q_pred`, `mdot`이 순서대로 사용된다. 따라서 같은 현재 압력과 현재 command라도 최근 command와 hidden valve state가 다르면 다음 pressure response가 달라질 수 있다.

이 부분이 `history`에 대해 검사할 state-aliasing 가설이다.

\[
P(P_{t+h}\mid P_t,u_t,H_t^{P,u})
\neq
P(P_{t+h}\mid P_t,u_t)
\]

여기서 predictor는 history가 있을 때 실제 held-out future pressure error가 감소하는지 측정한다.

## rate와 reserve는 어떤 물리 의미가 있는가?

### Rate: realized net-flow response

pressure rate는 이전 valve command와 hidden flow state가 실제 pressure에 만든 결과를 측정한 값이다. 현재 pressure가 같아도 상승 중인 상태와 하강 중인 상태의 짧은 미래는 다를 수 있다. 다만 History observation은 이미 최근 pressure를 갖고 있으므로 explicit rate의 역할은 새 센서가 아니라 다음 coordinate를 명시하는 것이다.

```text
두 개의 큰 absolute pressure를 MLP가 빼서 작은 slope를 학습
                         ↓
signed finite difference를 직접 제공
```

현재 policy normalization에서는 각 pressure가 `P/100`이고 explicit rate는 `rate/50`이다. 50 Hz 정의를 대입하면 explicit normalized rate는 `P_t-P_{t-1}`이지만 normalized history를 network가 직접 빼면 `(P_t-P_{t-1})/100`이다. 따라서 기존 rate 효과에는 약 100배의 수치적 강조가 포함될 수 있다. 최종 RL ablation에는 scale-matched rate control이 필요하다.

### Reserve: local pressure headroom

reserve는 valve 양단의 압력차를 단순화한 local headroom proxy다.

\[
r_+=P_{ch,+}-P_{act,+},\qquad
r_-=P_{act,-}-P_{ch,-}
\]

compressible orifice flow 자체는 단순 압력차만의 선형 함수가 아니며 pressure ratio, upstream absolute pressure, valve effective area의 영향을 받는다. 따라서 reserve를 mass flow 또는 전체 control authority와 동일시하면 안 된다. 하지만 actuator valve를 열었을 때 사용할 수 있는 flow-driving condition을 나타내는 해석 가능한 local coordinate로 사용할 수 있다.

현재 policy가 이미 chamber와 actuator pressure를 모두 보기 때문에 explicit reserve의 논문적 질문은 다음과 같다.

> 네 raw pressure를 입력한 finite MLP가 pressure headroom을 스스로 학습하도록 두는 것보다 reserve 좌표를 명시하는 것이 sample efficiency, robustness와 control behavior를 일관되게 개선하는가?

## 구현된 predictor 실험

분석 코드는 다음 파일이다.

```text
pneu_rl/scripts/analyze_lib3_predictive_observation.py
```

### 공정성을 위한 핵심 설계

1. 모든 predictor가 하나의 동일한 replay transition set을 사용한다.
2. reference는 plant transition에 직접 작용하지 않으므로 predictor 입력에서 제외한다.
3. \(h>1\)일 때 실제 기록된 `a[t:t+h]`를 모든 predictor에 똑같이 제공한다.
4. episode 단위로 train/validation/test를 나눠 adjacent row leakage를 막는다.
5. input과 target normalization 통계는 train split에서만 계산한다.
6. target은 policy state 전체가 아니라 실제 단위의 네 pressure 변화량이다.
7. 1/5/10/25 step, 즉 0.02/0.10/0.20/0.50 s를 각각 평가한다.
8. predictor seed 0/1/2의 평균과 표준편차를 저장한다.

기존 replay의 `done` mask는 모든 transition에서 1로 저장되어 episode boundary를 직접 제공하지 않는다. 분석 코드는 `infos.pkl`의 최종 physical step과 `cfg.yaml`의 512-step horizon으로 원래 global step과 episode group을 복원한다. 현재 source replay는 총 512,000 step 중 마지막 500,000 step이므로 local replay index 0은 global step 12,000에 해당한다. Multi-step sample은 복원한 episode boundary를 넘지 않으며 split도 이 episode group 단위로 수행한다.

### 비교하는 입력

| 이름 | Predictor state 입력 | 검증 질문 |
|---|---|---|
| `current` | 현재 네 pressure | 현재 값만으로 충분한가? |
| `pressure_history` | 11-step pressure | 진행 방향/history가 필요한가? |
| `pressure_action_history` | pressure history + action history | delayed valve command 정보가 필요한가? |
| `compact_rate_reserve_action_history` | 현재 pressure + rate/reserve + action history | 여섯 물리 coordinate가 pressure history를 압축할 수 있는가? |
| `history_action_zero2` / `history_action_reserve` | full history + zero 2개 / reserve 2개 | reserve 단독 predictor 효과와 dimension placebo |
| `history_action_zero4` / `history_action_rate` | full history + zero 4개 / rate 4개 | rate 단독 predictor 효과와 dimension placebo |
| `history_action_zero6` | full history + 6개 zero | 입력 차원/parameter 증가 placebo |
| `history_action_rate_reserve` | full history + rate/reserve | redundant physics coordinate가 finite-model prediction을 돕는가? |

`history_action_zero6`는 `history_action_rate_reserve`와 input dimension을 같게 만든다. 따라서 단순히 첫 layer parameter가 늘어난 효과와 실제 derived coordinate 효과를 구분한다.

### 실행 방법

먼저 저장 replay 구조와 deterministic feature를 검사한다.

```bash
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/analyze_lib3_predictive_observation.py \
  --check-only
```

전체 예비 분석은 다음과 같이 실행한다.

```bash
MPLCONFIGDIR=/tmp/mpl-cache \
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/analyze_lib3_predictive_observation.py
```

기본 출력은 다음 경로에 저장된다.

```text
pneu_rl/exp/lib3_predictive_observation/
└── 260824_hist_rate_reserve_replay_pilot/
    ├── metrics.csv
    ├── aggregate_metrics.csv
    ├── summary.json
    └── prediction_rmse.png
```

## 2026-08-24 smoke pilot

전체 pipeline이 실제 replay에서 끝까지 동작하는지 확인하기 위해 다음의 축소 실행을 수행했다.

- source: `260810_lib3_ablation_profile_hist_rate_reserve_seed0` replay
- horizon: 1/5/10/25 step
- horizon별 sample: 10,000
- predictor seed: 0 한 개
- hidden width: 128
- training: 최대 5 epoch
- 목적: 구현 검증과 약한 경향 확인
- 제한: 논문 결과 또는 observation 필요성의 최종 근거가 아님

네 pressure를 합친 test RMSE는 다음과 같다.

| Predictor 입력 | 0.02 s | 0.10 s | 0.20 s | 0.50 s |
|---|---:|---:|---:|---:|
| current | 0.372 | 1.588 | 2.512 | 3.218 |
| pressure history | 0.354 | 1.524 | 2.386 | 2.834 |
| pressure+action history | 0.339 | 1.448 | 2.269 | **2.742** |
| compact rate/reserve+action history | 0.334 | 1.452 | 2.363 | 2.952 |
| full history+zero-6 | 0.344 | 1.455 | 2.264 | 2.777 |
| full history+rate/reserve | **0.331** | **1.409** | **2.250** | 2.758 |

![lib3 predictor smoke pilot](../../../pneu_rl/exp/lib3_predictive_observation/260824_hist_rate_reserve_replay_smoke_pilot/prediction_rmse.png)

[원자료 metrics](../../../pneu_rl/exp/lib3_predictive_observation/260824_hist_rate_reserve_replay_smoke_pilot/metrics.csv) · [집계 metrics](../../../pneu_rl/exp/lib3_predictive_observation/260824_hist_rate_reserve_replay_smoke_pilot/aggregate_metrics.csv) · [실행 metadata](../../../pneu_rl/exp/lib3_predictive_observation/260824_hist_rate_reserve_replay_smoke_pilot/summary.json)

현재 허용되는 예비 해석은 다음뿐이다.

1. current-only보다 pressure history가 네 horizon 모두 낮았다.
2. action history를 추가하면 pressure history보다 네 horizon 모두 낮았다.
3. explicit rate+reserve는 full history 대비 0.02/0.10/0.20 s에서 각각 약 2.5/2.7/0.8% 낮았지만 0.50 s에서는 약 0.6% 높았다.
4. compact physics 입력은 0.02/0.10 s에서는 full history에 가까웠지만 0.20/0.50 s에서는 더 나빴다. 현재 설정에서는 rate/reserve가 긴 pressure history 전체를 대체한다고 볼 수 없다.
5. `zero-6`와의 차이가 작고 predictor seed가 하나뿐이므로 explicit feature의 이득은 아직 확정할 수 없다.

따라서 이 smoke pilot은 **history/action history의 predictive value를 더 정식으로 검사할 가치가 있음**을 보여주지만, rate/reserve의 독립적 효과를 확정하지 않는다. 기본 full run인 120,000 sample × predictor seed 3개와 policy-independent dataset 결과가 필요하다.

## 2026-08-24 정식 replay predictor 결과

Smoke 이후 horizon별 120,000 sample, predictor seed 0/1/2, 최대 40 epoch 조건으로 정식 실행했다. `rate-only/reserve-only`와 각각의 same-dimension zero placebo도 추가했다.

```text
pneu_rl/exp/lib3_predictive_observation/
├── 260824_hist_rate_reserve_replay_full/
└── 260824_rate_reserve_separated_full/
```

네 pressure를 합친 held-out RMSE의 seed 평균은 다음과 같다. 괄호 안 표준편차는 원자료 CSV에 저장되어 있다.

| Predictor 입력 | 0.02 s | 0.10 s | 0.20 s | 0.50 s |
|---|---:|---:|---:|---:|
| current pressure | 0.335 | 1.360 | 1.962 | 1.641 |
| pressure history | 0.139 | 0.896 | 1.611 | 1.548 |
| pressure+action history | 0.071 | 0.398 | 0.744 | 1.213 |
| full history+reserve | 0.069 | 0.397 | 0.742 | 1.213 |
| reserve dimension placebo, zero-2 | 0.072 | 0.399 | 0.741 | 1.221 |
| full history+rate | 0.027 | 0.252 | **0.604** | 1.173 |
| rate dimension placebo, zero-4 | 0.071 | 0.399 | 0.738 | 1.217 |
| full history+rate+reserve | **0.027** | **0.245** | 0.604 | **1.165** |
| combined dimension placebo, zero-6 | 0.071 | 0.395 | 0.739 | 1.216 |
| current+rate/reserve+action history | 0.038 | 0.338 | 0.722 | 1.214 |

![lib3 정식 predictor 결과](../../../pneu_rl/exp/lib3_predictive_observation/260824_hist_rate_reserve_replay_full/predictive_evidence_summary.png)

[전체 결합 결과](../../../pneu_rl/exp/lib3_predictive_observation/260824_hist_rate_reserve_replay_full/aggregate_metrics.csv) · [rate/reserve 분리 결과](../../../pneu_rl/exp/lib3_predictive_observation/260824_rate_reserve_separated_full/aggregate_metrics.csv) · [통합 결과](../../../pneu_rl/exp/lib3_predictive_observation/260824_hist_rate_reserve_replay_full/combined_aggregate_metrics.csv)

정식 결과의 해석은 다음과 같다.

1. **Pressure history 효과가 일관된다.** current-only 대비 네 horizon 모두 오차가 감소했다. 다만 0.5 s current error가 0.2 s보다 작아지는 현상은 recorded future action sequence가 closed-loop policy의 중간 관측을 간접적으로 포함하는 retrospective replay의 한계와 관련될 수 있다. 절대 horizon 곡선보다 동일 horizon 내 입력 비교를 사용한다.
2. **Action history 효과가 매우 크다.** pressure history 대비 0.02/0.10/0.20/0.50 s에서 각각 약 49/56/54/22% 감소했다. 숨은 valve/delay state의 measurable proxy라는 가설과 일치한다.
3. **Rate가 predictor 개선의 주효과다.** zero-4 대비 rate-only는 네 horizon에서 약 62/37/18/4% 감소했다.
4. **Reserve-only는 평균 future-pressure prediction을 거의 개선하지 않는다.** zero-2 대비 차이는 약 4/0.4/-0.1/0.6%로 작고 horizon에 따라 일관되지 않았다.
5. **Rate 위에서 reserve의 추가 이득도 작다.** rate-only 대비 rate+reserve 차이는 약 1.8/2.8/0.0/0.8%다. 따라서 결합 predictor 성능을 reserve의 일반적인 state-sufficiency 효과로 설명할 수 없다.
6. **Compact physics coordinate는 짧은 pressure history를 상당 부분 압축한다.** current+rate/reserve+action history는 full pressure+action history보다 0.02–0.20 s에서 더 낮았고 0.50 s에서 거의 같았다. 이 결과 역시 rate의 영향이 지배적이다.

Predictor 기준의 현재 결론은 다음과 같다.

> Pressure/action history는 미래 압력 예측에 실질적인 정보를 제공하고, explicit rate는 그 history에 포함된 local derivative를 finite network가 사용하기 쉽게 만든다. Raw reserve는 전체 dynamics prediction을 개선하는 일반 상태 변수가 아니라 조건부 flow authority feature인지 별도로 검사해야 한다.

빠른 smoke test 예시는 다음과 같다.

```bash
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/analyze_lib3_predictive_observation.py \
  --horizons 1 5 \
  --feature-sets current pressure_action_history history_action_rate_reserve \
  --seeds 0 \
  --max-samples 5000 \
  --epochs 2 \
  --output-dir /tmp/lib3_predictor_smoke \
  --no-plot
```

## 결과 해석 규칙

실험 전에 결론 기준을 고정한다.

| 관측 결과 | 허용되는 해석 | 금지되는 과장 |
|---|---|---|
| history가 current보다 held-out error를 낮춤 | temporal context가 future prediction에 유용 | 완전한 Markov state를 수학적으로 증명 |
| action history가 pressure history보다 낮음 | delayed valve state/action queue의 measurable proxy | valve internal state를 정확히 복원 |
| compact physics가 full history와 비슷함 | rate/reserve가 compact predictive summary일 가능성 | 두 feature가 독립적인 physical state |
| full history+rate/reserve가 zero-6보다 낮음 | finite network에 유용한 inductive bias | 새로운 정보가 추가됨 |
| prediction은 같고 RL만 개선 | optimization/sample-efficiency 효과 | observability 개선 |
| reserve-only 효과가 seed마다 불안정 | reserve 단독 주장을 축소 | seed-0 winner를 일반 법칙으로 주장 |

특히 다음 두 비교는 의미가 다르다.

\[
E(current)>E(history)
\]

은 observation history의 predictive information에 대한 근거가 될 수 있다. 반면

\[
E(history)>E(history+rate+reserve)
\]

는 rate/reserve가 deterministic feature이므로 정보 증가가 아니라 선택한 finite predictor architecture와 optimization에 유리한 coordinate라는 뜻이다.

## 기존 replay 실험의 한계와 논문용 dataset

현재 코드는 `260810 ... rate_reserve seed0` policy가 방문한 상태의 replay를 공통 dataset으로 사용한다. 입력 비교 자체는 공정하지만 상태 분포가 한 policy에 의해 만들어졌으므로 결과는 retrospective pilot이다.

논문용 최종 dataset은 policy-independent excitation으로 별도 수집해야 한다.

- safe random multi-sine/colored action
- 각 valve의 short pulse와 paired pulse
- PID tracking trajectory
- low/mid/high chamber reserve operating points
- 같은 current pressure에 접근하는 rising/falling trajectory
- training reference와 분리된 held-out trajectory bank
- simulator privileged diagnostic인 valve `z`, area, `q_pred`, `mdot`은 predictor target/oracle 분석에만 저장

Actor는 real에서 측정할 수 없는 privileged variable을 입력받지 않는다. Privileged predictor는 measurable history와 oracle 사이에 얼마나 정보 gap이 남는지만 보여주는 upper bound다.

## 인과 근거를 위한 matched-pulse 실험

실행 코드는 다음 파일이다.

```text
pneu_rl/scripts/run_lib3_matched_pulse_observability.py
```

2026-08-24에 다음 기본 설정을 실행했다.

```bash
MPLCONFIGDIR=/tmp/mpl-cache \
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/run_lib3_matched_pulse_observability.py
```

결과 경로는 다음과 같다.

```text
pneu_rl/exp/lib3_matched_pulse_observability/
└── 260824_reserve_history_pulse/
    ├── reserve_pulse_summary.csv
    ├── reserve_pulse_regression.csv
    ├── reserve_pulse_traces.csv
    ├── reserve_pulse_effect.png
    ├── history_candidates.csv
    ├── history_pair_summary.csv
    ├── history_pair_traces.csv
    ├── history_pair_actions.npz
    ├── history_pair_effect.png
    └── summary.yaml
```

### Reserve intervention

1. `ACT_POS/NEG`, current action과 reference가 비슷한 operating state를 만든다.
2. chamber pressure만 달리하여 low/high reserve 상태를 만든다.
3. 동일한 actuator-valve pulse를 적용한다.
4. 0.02–0.5 s의 signed `ΔP_act`, peak rate와 integrated mass flow를 비교한다.

같은 action에서 reserve에 따라 response가 반복적으로 달라지면 reserve의 물리적 의미를 지지한다. 이 실험은 explicit reserve input의 필요성이 아니라 **state-dependent action effect**를 검증한다.

### Rate/history intervention

1. 거의 같은 current pressure에 상승 방향과 하강 방향으로 접근한다.
2. current pressure와 다음 valve pulse를 match한다.
3. 최근 pressure rate 또는 valve history만 달리한다.
4. future pressure를 비교한다.

이 결과가 다르면 current-only observation의 state aliasing을 직접 시각화할 수 있다.

## 2026-08-24 matched-pulse 결과

### Reserve intervention 결과

각 branch에서 actuator pressure level 3개와 requested reserve 4개를 교차하여 12개 상태를 만들었다. 동일한 initialization/preparation을 두 번 replay하고 한쪽에만 physical command `0.95` pulse를 적용했다. `pulse response − no-pulse response`를 계산하여 자연 drift를 제거했다. Reserve path와 함께 atmosphere-connected actuator valve를 negative control로 측정했다.

Absolute actuator pressure의 confounding을 제거하기 위해 actuator level별 mean을 제거한 within-level fixed-effect 회귀를 사용했다.

| Branch/probe | 0.20 s slope | 0.20 s corr. | 0.50 s slope | 0.50 s corr. | 해석 |
|---|---:|---:|---:|---:|---|
| POS reserve path | 0.00722 | 0.896 | 0.16489 | 0.915 | reserve가 클수록 동일 inlet pulse 효과 증가 |
| POS atmosphere control | ≈0 | 0 | ≈0 | 0 | POS reserve에 민감하지 않음 |
| NEG reserve path | ≈0 | -0.674 | ≈0 | -0.681 | correlation 부호는 있으나 reserve-sensitivity slope가 수치적으로 거의 0 |
| NEG atmosphere control | ≈0 | -0.855 | ≈0 | -0.844 | 마찬가지로 reserve-sensitivity slope가 거의 0 |

여기서 NEG correlation은 slope가 `10^-7 kPa effect/kPa reserve` 수준인 거의 상수값의 작은 수치 변화에 대한 값이므로 물리적으로 큰 sensitivity로 해석하지 않는다. Pooled regression에서는 NEG reserve와 pulse response가 양의 상관처럼 보였지만, actuator level을 통제하면 사라졌다. 즉 pooled 관계는 reserve가 아니라 upstream actuator absolute pressure에 의해 만들어졌다.

POS reserve path의 측정 reserve 범위는 약 `12.0–52.0 kPa`였고, 0.2 s signed pulse effect는 `0.89–1.34 kPa`, 0.5 s에서는 `16.63–26.20 kPa`였다. NEG에서는 reserve `28.2–62.1 kPa` 범위에서도 같은 actuator level 안의 reserve 변화가 pulse effect를 거의 바꾸지 않았다.

이 비대칭은 compressible-orifice regime과 일치하는 설명 후보가 있다. lib3 식의 critical pressure ratio는 약 `0.528`이고, NEG reserve-path 조건 12개 중 9개가 `P_ch,neg/P_act,neg <= 0.528`인 choked 영역이었다. 반면 POS 조건은 12개 모두 `P_act,pos/P_ch,pos = 0.683–0.920`으로 subsonic 영역이었다. 따라서 단순한 signed pressure difference인 raw reserve보다 upstream pressure와 pressure ratio를 포함한 orifice-flow potential이 NEG authority를 더 정확히 표현할 가능성이 있다. 이는 현재 simulator 결과에 대한 inference이며 별도 orifice-potential ablation으로 검증해야 한다.

![lib3 reserve matched pulse](../../../pneu_rl/exp/lib3_matched_pulse_observability/260824_reserve_history_pulse/reserve_pulse_effect.png)

[Reserve 원자료](../../../pneu_rl/exp/lib3_matched_pulse_observability/260824_reserve_history_pulse/reserve_pulse_summary.csv) · [level별 회귀](../../../pneu_rl/exp/lib3_matched_pulse_observability/260824_reserve_history_pulse/reserve_pulse_regression.csv)

### History/rate matched-pair 결과

각 branch에서 1,200개의 30-step preparation sequence를 실제 simulator로 실행했다. 이후 다음 조건을 만족하는 rising/falling pair를 branch별 8개 선택했다.

- 현재 네 pressure의 최대 차이 ≤ 2.5 kPa
- POS/NEG reserve 최대 차이 ≤ 2.5 kPa
- target pressure rate의 부호가 반대
- 각 target absolute rate가 0.5–50 kPa/s
- pair 선택 후 동일 pulse와 no-pulse counterfactual replay

실제로 선택된 pair의 품질과 pulse-effect 차이는 다음과 같다.

| Branch | pair 수 | 현재 pressure 최대차 평균/최대 | reserve 최대차 평균/최대 | rate separation 평균 | 0.20 s effect 차이 절대평균 | 0.50 s effect 차이 절대평균 |
|---|---:|---:|---:|---:|---:|---:|
| POS | 8 | 0.466 / 0.706 kPa | 0.498 / 0.745 kPa | 71.1 kPa/s | 0.133 kPa | 4.282 kPa |
| NEG | 8 | 0.499 / 0.926 kPa | 0.414 / 0.554 kPa | 74.0 kPa/s | 1.404 kPa | 6.594 kPa |

1 step과 5 step 효과 차이는 모두 0이었고 10/25 step에서 나타났다. 이는 0.1 s sensor delay와 일치한다. 현재 pressure와 reserve를 약 1 kPa 이내로 맞추어도 recent rate/action history가 다른 상태에서 동일 pulse의 0.5 s 효과가 POS `4.28 kPa`, NEG `6.59 kPa`만큼 달랐다.

따라서 simulator 안에서는 다음 최소 결론이 성립한다.

> Current pressure와 raw reserve만으로는 short-horizon action effect가 유일하게 정해지지 않으며, recent pressure/action history가 hidden valve/delay state의 control-relevant proxy 역할을 한다.

다만 pair는 full hidden state가 동일한 인과 실험이 아니라 current measurable state를 근사 matching한 실험이다. real pulse repetition과 더 낮은 reference-rate 영역의 촘촘한 matching이 필요하다.

![lib3 history matched pulse](../../../pneu_rl/exp/lib3_matched_pulse_observability/260824_reserve_history_pulse/history_pair_effect.png)

[History pair 원자료](../../../pneu_rl/exp/lib3_matched_pulse_observability/260824_reserve_history_pulse/history_pair_summary.csv) · [재현 action sequence](../../../pneu_rl/exp/lib3_matched_pulse_observability/260824_reserve_history_pulse/history_pair_actions.npz)

## 이후 RL 재구성

기존 reward, reference, SAC hyperparameter와 simulator는 바꾸지 않고 observation과 control feature만 비교한다.

| RL ID | Observation | 목적 |
|---|---|---|
| R0 | History+action history | 기준선 |
| R1 | R0+rate | realized response coordinate |
| R2 | R0+reserve | headroom coordinate |
| R3 | R0+rate+reserve | 결합 interaction |
| R4 | R0+zero-6 | parameter-count placebo |
| R5 | R0+rate+reserve, scale matched | rate의 수치적 강조 분리 |

각각 train seed 0/1/2, 동일한 actual-step evaluation, 동일한 unseen reference bank로 평가한다. 기존 R0–R3 seed-0는 재사용할 수 있지만 seed 1/2와 R4/R5가 추가되어야 한다.

평가 지표는 다음을 함께 사용한다.

- POS/NEG RMSE, MAE와 steady-state offset
- physical-rate deadband를 적용한 direction agreement
- transition/plateau 구간별 성능
- learning curve와 목표 성능 도달 episode
- `mean |Δu|`, `mean |Δ²u|`와 action spectrum
- reserve violation과 chamber overcharge
- unseen reference와 initial pressure generalization

## PINN과의 관계: 같은 문제의식, 다른 방법

[Raissi et al.의 PINN](https://doi.org/10.1016/j.jcp.2018.10.045)은 neural network가 데이터만 맞추는 것이 아니라 governing differential-equation residual도 만족하도록 학습한다. 현재 lib3의 rate/reserve observation 추가는 loss에 pressure/mass-flow residual을 넣지 않으므로 PINN이 아니다.

현재 방법은 다음에 가깝다.

> Physics-derived feature engineering 또는 physics-informed observation design

교수님의 PINN 비유에서 가져올 핵심은 명칭이 아니라 검증 방식이다.

- 임의 feature가 아니라 pneumatic equation에서 candidate variable을 유도한다.
- next-state predictiveness를 검사한다.
- placebo feature와 비교한다.
- matched intervention으로 action effect를 검사한다.
- 필요하면 이후 predictor에 pressure/mass-balance residual auxiliary loss를 추가한다.

마지막 auxiliary loss까지 추가한 경우에만 PINN-like dynamics model이라고 부르는 것이 적절하다. 현재 단계에서 literal PINN을 RL actor에 바로 결합하면 observation 기여와 auxiliary-loss 기여가 섞이므로 먼저 수행하지 않는다.

## 관련 이론과 이 실험의 연결

### Markov state와 transition preservation

[DeepMDP, Gelada et al., ICML 2019](https://proceedings.mlr.press/v97/gelada19a.html)은 좋은 state representation을 reward와 next-state transition을 보존하는 관점에서 다룬다. 본 predictor 실험은 DeepMDP algorithm을 구현하는 것이 아니라, observation candidate가 physical next-pressure prediction을 보존하는지 검사한다는 제한된 원리를 가져온다.

### History와 self-predictive representation

[Ni et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/666c1861d709bd84e20b6e0e02a2c223-Abstract-Conference.html)은 state/history abstraction을 self-predictive property라는 공통 관점으로 분석한다. 이는 pressure/action history가 단순히 input dimension을 늘리는 것이 아니라 future-relevant representation인지 검사해야 한다는 근거다.

[Patil et al., AISTATS 2024](https://proceedings.mlr.press/v238/patil24b.html)은 function approximation에서도 history-based representation을 이용하는 정책을 이론적으로 다룬다. lib3에서는 0.1 s delay와 hidden valve dynamics 때문에 finite history 사용의 실험적 동기가 더 직접적이다.

### Causal state와 predictive state

[Zhang et al., Learning Causal State Representations](https://arxiv.org/abs/1906.10437)은 action-observation history에서 future observation을 예측하는 representation을 통해 partially observable environment의 causal state를 근사한다. 본 연구에서는 full causal-state learning을 주장하지 않고, pressure/action history가 future pressure에 주는 추가 설명력을 측정한다.

[Singh et al., Predictive State Representations, UAI 2004](https://arxiv.org/abs/1207.4167)은 hidden physical state를 직접 표현하는 대신 observable experiment outcome의 prediction으로 dynamical state를 표현하는 관점을 제시한다. 이는 real system에서 valve `z`를 센싱하지 못하더라도 measurable history의 predictive sufficiency를 검사할 수 있다는 근거다.

### Bisimulation과 state aliasing

[Zhang et al., ICLR 2021](https://openreview.net/pdf?id=-2FCwDKRREu)은 reward와 transition behavior가 유사한 state를 묶는 bisimulation 기반 representation을 다룬다. 역으로 같은 reduced observation으로 묶인 sample들이 다른 future transition을 보인다면 control 관점에서 중요한 state aliasing이 남아 있다는 동기가 된다. 본 연구의 matched pair는 이를 정식 bisimulation metric으로 증명하는 것이 아니라 시각적·통계적 진단으로 사용한다.

### Nonlinear observability

[Haber et al., IEEE TCNS 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6178986/)은 nonlinear system의 state estimation에서 측정 변수의 수, observation length와 estimation error 사이의 trade-off를 다룬다. 이 관점은 lib3에서 센서를 무한히 추가하는 대신 pressure/action observation window가 hidden dynamics 예측에 충분한지 확인하는 근거가 된다. 단, neural predictor RMSE 감소를 고전적 nonlinear observability rank condition의 증명으로 부르지는 않는다.

## 최종적으로 주장 가능한 단계

증거에 따라 문장 강도를 조절한다.

### 현재 코드 audit만으로 가능한 주장

> lib3의 explicit rate와 reserve는 pressure history/current pressure에서 정확히 계산되는 derived coordinates다.

### 현재 predictor와 simulator matched-pulse로 가능한 주장

> Pressure/action history improves short-horizon predictive state reconstruction, and explicit pressure rate provides a compact physics-derived coordinate for the same future-relevant trend information. Raw pressure reserve is not a uniformly sufficient control-authority coordinate: its effect is branch- and flow-regime-dependent.

이 문장은 predictor의 세 seed 결과와 simulator matched-pulse 결과까지만 반영한다. RL 성능 향상이나 real-system 인과성은 아직 포함하지 않는다.

### Predictor와 multi-seed RL까지 성공하면 가능한 주장

> Pressure/action history improves predictive state reconstruction, while explicit rate/reserve provides a physics-informed inductive bias that improves learnability and control generalization.

### Real matched-pulse와 lib4 재현까지 성공하면 가능한 주장

> The effective action-to-pressure response is state dependent; measurable pressure trend, chamber–actuator headroom and recent actions explain variations in short-horizon pneumatic control authority.

### 피해야 할 주장

- rate/reserve가 독립적인 새로운 sensor information이다.
- predictor RMSE 감소가 mathematical Markov sufficiency를 증명한다.
- reserve 두 개가 전체 state-dependent control authority와 같다.
- physics-derived observation을 사용했으므로 PINN이다.
- seed-0 RL 결과만으로 observation의 필수성을 증명했다.

## 현재 결론

2단계까지의 실험은 다음을 보여준다.

1. **History의 정보 역할은 지지된다.** Current pressure predictor보다 pressure history가 모든 horizon에서 낮은 RMSE를 보였고, action history를 더하면 오차가 추가로 감소했다. 현재 pressure와 raw reserve를 약 1 kPa 이내로 맞춘 simulator pair에서도 recent history가 다르면 동일 pulse의 0.5 s response가 POS 평균 `4.28 kPa`, NEG 평균 `6.59 kPa` 달랐다. 따라서 history는 hidden valve/delay state에 의한 state aliasing을 줄이는 control-relevant proxy다.
2. **Rate의 표현 역할은 강하게 지지된다.** Rate-only는 explicit feature가 만든 predictor 개선의 대부분을 설명했다. Rate+reserve 대비 rate-only의 RMSE 차이는 horizon별 약 `0–2.8%`에 불과했다. Rate는 새로운 sensor 정보가 아니라 history에 들어 있는 국소 추세를 finite network가 쉽게 이용하도록 압축한 derived coordinate다.
3. **Raw reserve의 보편적 물리 역할은 지지되지 않는다.** Reserve-only는 predictor에서 zero-placebo와 거의 같았다. Matched pulse에서는 POS reserve path에 level-controlled sensitivity가 있었지만, NEG에서는 actuator absolute pressure를 통제하면 raw reserve sensitivity가 거의 0이었다. 따라서 두 branch에 같은 pressure-difference 정의를 적용한 현재 reserve는 보편적인 control-authority coordinate가 아니다.
4. **NEG에는 flow regime을 포함한 비선형 feature가 필요할 가능성이 높다.** NEG 조건 12개 중 9개가 simulator orifice 식의 choked 영역이었다. 이 영역에서는 downstream pressure difference보다 upstream absolute pressure가 flow capacity를 더 잘 설명할 수 있다. 다음 candidate는 measurable pressure만 사용한 `upstream pressure × pressure-ratio flow function` 형태의 orifice-flow potential이어야 한다.
5. **기존 `rate+reserve` RL 상승을 reserve의 Markov 정보 효과라고 결론 내릴 수 없다.** 현재 증거로는 rate가 주효하고, reserve는 POS conditional headroom, operating-point regularization 또는 rate와의 optimization interaction으로 도움을 주었을 가능성이 남는다.

따라서 lib3를 갈아엎지 않는다. 기존 simulator, reward, SAC, reference를 유지하고 다음 RL 비교만 추가한다.

- `History`
- `History+rate`
- `History+raw reserve`
- `History+rate+raw reserve`
- `History+rate+orifice-flow potential`
- dimension/scale-matched placebo

각 모델은 seed 0/1/2와 동일 actual-step evaluation을 사용한다. 그 전에 orifice-flow potential을 현재 predictor와 matched-pulse 데이터에 적용해 raw reserve보다 NEG response를 더 잘 설명하는지 확인한다. 이 순서가 8월 10–11일 `rate+reserve` 결과를 단순 성능표에서 물리적으로 검증 가능한 observation-design 주장으로 바꾸는 최소 경로다.
