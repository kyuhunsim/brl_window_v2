# lib3/lib4 연구 방향과 novelty 설계

작성 기준일: 2026-08-23

## 0. 한 문장 결론

이 연구의 중심 novelty를 `soft actuator simulator를 만들고 SAC를 적용했다`로 잡으면 약하다. 가장 방어력 있는 중심 문장은 다음과 같다.

> **본 연구의 가장 유망한 중심 가설은 shared dual-pressure source와 soft actuator가 결합되면 controller의 유효 입력 능력(control authority)이 고정되지 않고 chamber–actuator reserve, 운동 상태와 valve/action history에 따라 변한다는 것이다. 본 연구는 이 state-dependent control authority를 short-horizon reachable output-rate로 정량화하고, 이를 reference certification과 temporal policy에 사용해 선행 pressure-source controller를 actuator-level simulation-to-real control로 확장하는 것을 목표로 한다. 이 문장은 아직 확정 novelty가 아니라 sim/real 실험으로 반증 가능하게 검증해야 할 가설이다.**

이 문장은 지금까지의 작업을 단순한 기능 추가 목록이 아니라 **simulator 식별 실패와 controller 학습 실패라는 서로 다른 두 단계**로 묶는다.

```text
[연구실 내 선행/동시 연구: 현재 논문의 novelty로 주장하지 않음]
정적·무지연 밸브 모델이 확장된 operating range에서 정확도 붕괴
→ opening/closing 비대칭 2차 지연, hysteresis, nonlinear flow 모델 개발

[현재 연구의 모델링 범위]
선행 동적 밸브 모델을 6개 valve에 개별 calibration
→ fit-only optimization에서 비물리적 parameter가 선택되는지 검증
→ 필요한 경우 안정성·응답성·단조성·full-open authority를 calibration constraint로 사용
→ chamber–actuator flow, leakage, variable volume, displacement dynamics와 통합

[제어 단계]
검증된 동적 simulator에 선행 direct SAC + MPObs + CAPS를 이식
→ 6-valve chamber–actuator 계의 hidden state와 delayed action effectiveness 때문에 실패
→ history + pressure-rate/reserve + 짧은 temporal action 구조 도출
→ lib4 simulation 학습, zero-shot real 평가, 제한된 real fine-tuning으로 검증
```

현재 단계에서 가장 강한 논문은 새로운 범용 RL 알고리즘이나 새로운 2차 밸브 방정식 논문이 아니라 **기존 동적 밸브 모델의 actuator-level 통합·검증, fidelity 증가로 발생한 controller scaling failure, failure-informed controller design, 실제 sim-to-real 검증을 결합한 robotics system paper**다. 이는 RA-L의 범위와도 잘 맞는다.

---

## 1. 선행 RA-L 논문의 novelty와 현재 연구의 위치

가장 가까운 선행 연구는 [Modeling and Reinforcement Learning-Based Control of Simultaneous Positive and Negative Pressure Generation in Pneumatic Systems](Modeling_and_Reinforcement_Learning-Based_Control_of_Simultaneous_Positive_and_Negative_Pressure_Generation_in_Pneumatic_Systems.pdf)이다. 이 논문의 공식적인 기여는 다음 두 축이다.

1. 단일 펌프가 양압과 음압을 동시에 만드는 시스템에 대해 airflow, pump kinematics, pneumatic component electromagnetics를 결합한 interactive simulator를 구축했다.
2. 두 압력을 넓은 범위에서 추종하기 위해 SAC에 MPObs와 CAPS를 결합하고, 실제 시스템과 외부 부하 변화에서 검증했다.

따라서 선행 논문의 novelty는 SAC, RK 적분기, MPObs, CAPS 각각의 최초성이 아니라, **새로운 dual-pressure power source 문제를 모델링하고 해당 시스템에서 필요한 학습 제어 구조를 실제로 검증한 시스템 수준의 기여**다. 논문은 RA-L 2025에 게재되었고, MPObs와 CAPS의 결합 효과를 ablation으로 제시했다 ([DOI](https://doi.org/10.1109/LRA.2025.3577422)).

또한 선행 논문은 이미 넓은 의미의 `real → sim → real` 연구다. 실제 장치 데이터로 solenoid-valve 계수를 linear least squares로 추정하고 check-valve 계수를 Nelder–Mead로 식별했으며, random command 실험으로 simulator 정확도를 검증했다. 이후 simulator에서 512,000 step을 학습하고 실제 시스템에서 102,400 step의 online adaptation을 수행했다. 따라서 현재 후속 연구의 novelty를 단순히 `real2sim2real` 또는 `simulation pretraining 후 real fine-tuning`으로 주장할 수는 없다.

선행 논문의 더 정확한 novelty는 다음 교집합이다.

```text
single pump로 simultaneous positive/negative pressure를 생성하는 새로운 plant
+ pump kinematics·airflow·valve electromagnetics를 결합한 interactive simulator
+ policy action trajectory를 이용한 MPObs
+ CAPS와 online temporal-weight scheduling을 이용한 real adaptation
+ DOB 비교, MPObs/CAPS ablation, 0–20 kg 부하 실험
```

즉 개별 알고리즘의 범용 최초성보다는, 당시 충분히 다뤄지지 않았던 **단일 펌프 dual-pressure source를 위한 모델–학습–실기 통합 해법**이 RA-L 수준의 시스템 novelty였다.

현재 연구는 그 문제를 다음 단계로 확장한다.

| 구분 | 선행 연구 | 현재 연구 |
|---|---|---|
| 최종 제어 대상 | 양압·음압 power-source chamber | chamber를 거쳐 구동되는 dual-pressure soft actuator |
| valve action | 2개 | 6개 |
| 밸브 모델 | 비교적 직접적인 정적/선형 opening 관계 | nonlinear area, hysteresis, opening/closing 비대칭 2차 dynamics |
| 압력 전달 | action에서 chamber pressure로 비교적 직접적 | action → valve state/flow → chamber reserve → actuator flow → actuator pressure |
| actuator | 부하로 연결 | leakage, variable volume, pressure-force-displacement coupling을 갖는 동적 상태 |
| 주요 숨은 상태 | 짧은 pneumatic delay와 coupled pressure | valve state, 유량, chamber reserve, actuator volume/displacement, leakage |
| 제어 목표 | dual pressure tracking | actuator pressure tracking 후 displacement tracking 및 real adaptation |

이 차이는 단순히 밸브 수가 2개에서 6개가 된 것이 아니다. 선행 연구는 `pressure-source regulation` 문제였고, 현재 연구는 `power source + transmission + compliant actuator`가 연결된 **다단 동적 제어 문제**다.

### 1.1 연구실 내 contribution 경계

사용자 설명에 따르면 정적 밸브 모델의 확장 실패와 opening/closing 비대칭 2차지연 밸브 모델은 연구실 동료가 자신의 논문에서 다루고 있다. 따라서 현재 논문은 해당 결과를 자체 novelty로 주장하지 않는다.

| 항목 | 현재 귀속 | 현재 논문에서의 처리 |
|---|---|---|
| 확장 조건에서 static valve model의 낮은 정확도, 예: 일부 조건의 `R²≈0.3` | 연구실 동료 논문 | motivation과 prior result로 인용; 정확한 표·데이터의 중복 사용 여부 협의 |
| asymmetric second-order valve dynamics와 기본 방정식 | 연구실 동료 논문 | adopted model로 명시하고 정식 citation 사용 |
| 6개 valve별 parameter calibration pipeline | 사용자 작업으로 이해됨 | 동료 논문과 중복 범위를 확인한 뒤 contribution 여부 확정 |
| `ω_n, ζ`, peak/HF, full-open, monotonicity 등의 feasibility 조건 | 사용자 작업으로 이해됨 | 실제 구현과 독립성을 확인하고 calibration contribution 후보로 사용 |
| dynamic valves와 chamber–actuator/leakage/variable-volume/displacement 통합 | 현재 연구 | coupled simulator의 system contribution 후보 |
| 선행 SAC/MPObs/CAPS의 lib3 실패 분석과 temporal physics-aware RL | 현재 연구 | controller contribution의 중심 |
| lib4 simulation, zero-shot real 평가, fixed-budget real fine-tuning | 현재 연구 | 최종 system-level evidence |

제출 전에는 연구실 동료 논문의 서지정보를 아래 placeholder 대신 확정해야 한다.

> `[Dynamic valve-model paper by laboratory collaborator — citation/preprint information required]`

동료 논문이 아직 공개되지 않았다면 static-vs-dynamic 결과나 동일 figure/data를 현재 논문에 그대로 중복 게재하지 않는다. 해당 model이 현재 논문의 필수 기반이라면 preprint 공개, 명확한 citation, 공동저자 범위 또는 저널의 related/concurrent-work 설명을 연구실 내에서 먼저 합의해야 한다.

---

## 2. 지금까지 실제로 수행한 연구 자산

### 2.1 채택한 동적 밸브 모델과 현재 연구의 확장 범위

현재 simulator가 채택한 valve model은 다음 세 요소를 결합한다.

1. Bouc-Wen 계열의 hysteresis state (z)
2. command, pressure, hysteresis에 따른 nonlinear effective area
3. 열림과 닫힘에 서로 다른 ω_n, ζ를 갖는 2차 유량 dynamics

대표식은 다음과 같다.

\[
Q_{s}=A_{\mathrm{eff}}(u,P_{\mathrm{in}},z)P_{\mathrm{in}}\Phi(P_{\mathrm{in}},P_{\mathrm{out}})
\]

\[
\dot x_1=x_2,
\qquad
\dot x_2=\omega_n^2(Q_s-x_1)-2\zeta\omega_nx_2,
\qquad
Q=x_1.
\]

관련 정리는 [solvalve.md](solvalve.md), lib3 구현은 [pneumatic_CT.cpp](../pneu_env/src/pneu_env/lib3/pneumatic_CT.cpp)에 있다. 그러나 이 동적 밸브 방정식과 static-model failure 자체는 연구실 동료 논문의 contribution으로 분리한다. 현재 연구가 주장할 수 있는 범위는 모델을 6개 valve에 calibration하고 coupled soft-actuator simulator 및 RL environment에 통합·검증한 부분이다.

### 2.2 물리적 feasibility를 고려한 parameter identification

연구실 동료의 선행/동시 연구에서는 정적 또는 무지연 밸브 모델이 확장된 6-valve operating range의 실제 phase lag와 opening/closing transient를 재현하지 못했으며, 일부 조건에서 `R²≈0.3` 수준과 큰 RMSE가 관찰되었다고 보고한다. 정확한 channel·dataset·수치는 해당 논문을 인용해 확정한다. 이 결과는 현재 논문의 novelty가 아니라 동적 모델을 채택하는 근거다.

현재 연구가 별도로 검증할 문제는 **채택한 동적 모델의 6-valve calibration**이다. 단순 SSE, RMSE, R² 최적화만 수행하면 데이터 구간에서는 잘 맞지만 다음과 같은 파라미터가 선택될 가능성이 있다.

- 지나치게 낮은 natural frequency로 실제 command response를 설명하지 못함
- 지나치게 낮거나 높은 damping ratio로 ringing 또는 비현실적인 저속 응답 발생
- command 1.0에서도 밸브가 충분히 열리지 않음
- 일부 command 구간에서 command 증가가 유량 증가로 이어지지 않음
- 평균 오차는 작지만 peak 또는 high-frequency response가 실제와 다름

따라서 constraint는 RL reward shaping이 아니라 채택한 동적 모델을 6개 실제 valve에 보정하기 위한 calibration prior다. 다만 이것이 독립 novelty가 되려면 동료 논문의 tuning 범위와 겹치지 않아야 하며, fit-only와 constrained calibration이 비슷한 held-out error를 보이면서도 physical consistency 또는 downstream control에서 차이가 난다는 증거가 필요하다. 그렇지 않으면 이 부분은 중요한 구현·검증 절차로 두되 메인 novelty로 과장하지 않는다.

현재 [optimizer.py](../pneu_env/src/pneu_env/optimizer.py)에서 코드로 확인되는 항목은 다음과 같다.

- ω_n bounds: 현재 기본 `11–25 rad/s`
- ζ bounds: 현재 기본 `0.4–0.6`
- high-frequency ratio penalty
- R² floor penalty
- peak-flow mismatch penalty
- full-open minimum-ratio penalty
- effective area와 주요 parameter의 양수화 또는 bounds 처리

사용자가 설명한 `command가 증가할수록 유량이 감소하지 않아야 한다`는 전 구간 단조성은 논문의 강한 제약조건 후보다. 다만 현재 확인한 optimizer에는 full-open 제약은 명시되어 있지만 전 command grid에서

\[
\frac{\partial Q_s}{\partial u}\ge 0
\]

를 직접 검사하는 항은 명확히 보이지 않는다. 논문 제출 전에는 이 조건을 코드·설정·결과 표에 명시적으로 남겨야 한다. 구현이 다른 스크립트에 있다면 해당 경로를 고정하고, 없다면 optimizer의 hard constraint 또는 dense-grid penalty로 추가해야 한다.

### 2.3 lib4 soft actuator multiphysics model

lib4에는 다음 결합이 들어가 있다.

- actuator positive/negative chamber leakage와 cross leakage
- displacement (L)에 따른 (V_{+}(L),V_{-}(L))
- (dV_{+}/dL,dV_{-}/dL)
- 압력에 따른 actuator force
- rod displacement와 velocity dynamics
- pressure-volume coupling

압력 dynamics는 개념적으로 다음과 같다.

\[
\dot P
=
\frac{RT}{V(L)}\dot m_{\mathrm{net}}
-
\frac{P}{V(L)}\frac{dV(L)}{dL}\dot L,
\]

누설은 현재 pressure difference 기반 질량유량으로 구현되어 있다.

\[
\dot m_{\mathrm{leak}}=k_{\mathrm{leak}}\Delta P.
\]

구현 근거는 [lib4 pneumatic_CT.cpp](../pneu_env/src/pneu_env/lib4/pneumatic_CT.cpp)와 [soft_actuator.cpp](../pneu_env/src/pneu_env/lib4/soft_actuator.cpp)다. RK 계열 적분기는 이 coupled ODE를 안정적으로 계산하기 위한 구현 요소이며, 그 자체를 novelty로 주장하면 안 된다.

### 2.4 lib3에서 확인한 선행 controller의 실패와 구조 개선

[lib3 ablation README](ablation/lib3_0807/README.md)에는 direct SAC에서 시작해 pressure history, action history, physics observation, integral error, K=2 action chunking으로 발전한 과정이 정리되어 있다.

현재 가장 중요한 관찰은 다음과 같다.

- Direct SAC는 최신 기준 궤적에서 POS/NEG RMSE `19.66/8.99 kPa`로 실패했다.
- History SAC는 큰 개형을 더 잘 추종했지만 offset과 reserve 문제가 남았다.
- 같은 조건에서 History SAC+CAPS와 History SAC+MPOBS1+CAPS는 History SAC보다 일관되게 좋아지지 않았고, 양압 chamber 과충전 또는 tracking 저하가 나타났다.
- `K=2 + physics observation`은 최신 기준에서 POS/NEG RMSE `1.208/0.981 kPa`, 방향 일치율 `0.998/0.996`을 기록했다.
- K=2 baseline만 사용하면 tracking 수치는 좋아져도 positive chamber가 약 `248.9 kPa`까지 올라갔다. 즉 temporal abstraction만으로 reserve 문제가 해결되지는 않았다.

이 결과는 `MPOBS와 CAPS가 틀렸다`는 결론이 아니다. **선행 시스템에서 유효했던 predictor와 smoothness prior가 더 복잡한 plant에서 자동으로 충분하지 않다**는 결과다. 이는 후속 연구의 좋은 연결점이 된다.

---

## 3. 왜 선행 구조가 lib3에서 실패했는가

### 3.1 현재 pressure만으로는 Markov state가 아니다

동일한 네 pressure를 관측하더라도 다음 값이 다르면 이후 pressure response가 달라질 수 있다.

- 밸브 내부 (z,x_1,x_2)
- 최근 action history
- chamber와 actuator 사이의 pressure gradient
- actuator displacement와 volume
- leakage와 flow direction

따라서

\[
o_t=[P_{ch,+},P_{ch,-},P_{act,+},P_{act,-},r_t]
\]

만으로는 실제 상태를 구분하지 못하는 state aliasing이 생긴다. 이때 SAC는 같은 observation에 서로 다른 action effect가 나타나는 것처럼 학습하게 된다.

### 3.2 six-action의 문제가 아니라 delayed action effectiveness의 문제다

6차원 action 자체가 로봇 arm보다 특별히 어려워서 실패한 것은 아니다. 차이는 각 action의 효과가 현재 state에 따라 크게 달라진다는 데 있다.

예를 들어 actuator positive pressure를 올릴 때 필요한 것은 단순히 한 밸브를 여는 것이 아니라:

```text
positive chamber에 충분한 reserve가 있는가
→ inlet/outlet valve의 현재 내부 opening state는 무엇인가
→ 실제 net flow가 actuator로 향하는가
→ variable volume과 leakage를 이기고 pressure가 증가하는가
```

를 함께 만족하는 것이다. action과 reward 사이의 경로가 길고 state-dependent이므로 direct policy는 방향성을 학습하기 어렵다.

### 3.3 MPObs의 predictor mismatch

MPObs는 predictor가 실제 plant의 future action effect를 충분히 맞출 때 강하다. lib3에서 predictor와 plant의 valve state, delay, pressure reserve 또는 초기화가 다르면 예측 pressure가 오히려 잘못된 state feature가 된다. 따라서 MPObs는 범용적으로 nonlinear system을 해결하는 장치가 아니라 **predictor fidelity에 의존하는 model-assisted observation**이다.

### 3.4 CAPS의 over-smoothing 가능성

CAPS는 action variation을 줄일 수 있지만, 다단 pneumatic plant에서 필요한 chamber 충전·방전 전환까지 억제할 수 있다. 따라서 낮은 \(\lVert\Delta u\rVert\)가 낮은 pressure error를 의미하지 않는다. lib3에서는 smoothness loss와 tracking gradient의 상대 크기, predictor mismatch, action threshold가 함께 작용한다.

---

## 4. 개선 구조를 원리에서 유도하는 방법

지금까지의 history, pressure rate, reserve, K=2는 우연히 붙인 기믹으로 설명하면 약하다. 각 항을 simulator 식에서 유도해야 한다.

### 4.1 action history: 숨은 valve state의 근사

2차 valve dynamics와 hysteresis state는 최근 command에 의존한다. 실제 배포에서 (z,x_1,x_2)를 직접 측정하기 어렵기 때문에 최근 pressure와 action history를 사용해 이를 간접 추정한다.

\[
o_t^{hist}
=
[P_{t-H:t},u_{t-H:t-1},r_{t-H:t+F}].
\]

이는 단순히 neural network input을 늘린 것이 아니라, partial observability를 줄이기 위한 finite-memory state reconstruction이다. 지연 환경에서 history 또는 state augmentation이 필요하다는 점은 ICLR 2021의 random-delay RL과 AISTATS 2024의 history-based policy 연구에서도 다뤄진다 ([ICLR 2021](https://openreview.net/forum?id=QFYnKlBJYR), [AISTATS 2024](https://proceedings.mlr.press/v238/patil24b.html)).

### 4.2 pressure rate: 현재 net-flow 방향의 측정 가능한 결과

\[
\dot P_{ch,+},\dot P_{ch,-},\dot P_{act,+},\dot P_{act,-}
\]

는 복잡한 valve state와 mass flow가 현재 pressure에 만든 결과를 압축해 나타낸다. 따라서 policy가 현재 action effect의 방향을 더 쉽게 구별하게 한다.

### 4.3 chamber reserve: 다음 actuator flow의 local authority

reserve는 다음처럼 정의한다.

\[
\rho_+=P_{ch,+}-P_{act,+},
\qquad
\rho_-=P_{act,-}-P_{ch,-}.
\]

부호 정의는 실제 valve flow direction과 일치하도록 최종 코드에서 고정해야 한다. 이 값은 actuator valve를 열었을 때 사용할 수 있는 pressure gradient를 나타낸다. 즉 reserve는 단순한 추가 pressure가 아니라 **현재 action이 유효한지를 나타내는 local control-authority feature**다.

### 4.4 state-dependent control authority의 정확한 정의

여기서 `control authority`는 action 차원 수, valve 최대 opening 또는 reserve 한 개를 뜻하지 않는다. 현재 plant state $x_t$에서 admissible action과 pressure/stroke/safety constraint를 지키며 짧은 시간 $K\Delta t$ 안에 만들 수 있는 **출력 변화의 집합**을 뜻한다.

상태, action과 제어 출력을 다음처럼 둔다.

\[
x_t=
[P_{ch,+},P_{ch,-},P_{act,+},P_{act,-},D,\dot D,h_t],
\qquad
u_t\in\mathcal U,
\]

\[
y_t=[P_{act,+},P_{act,-},D],
\]

여기서 $h_t$는 직접 측정하기 어려운 valve hysteresis/dynamic state를 나타내며 실제 policy에서는 pressure/action history로 근사한다. $K$-step local control-authority set은 다음처럼 정의한다.

\[
\mathcal A_K(x_t)
=
\left\{
\frac{y_{t+K}-y_t}{K\Delta t}
\;\middle|\;
u_{t:t+K-1}\in\mathcal U,
x_{t:t+K}\in\mathcal X_{safe}
\right\}.
\]

따라서 같은 displacement와 같은 tracking error라도 authority는 같지 않을 수 있다.

```text
상태 A: POS chamber reserve가 충분하고 수축 방향 유량이 이미 형성됨
→ 같은 valve command로 큰 +dP_act/dt와 +dD/dt를 만들 수 있음

상태 B: POS reserve가 작고 valve가 closing transient에 있으며 actuator가 복원 중
→ 같은 command를 넣어도 반응이 작거나 늦고, 먼저 chamber를 충전해야 함
```

`state-dependent`라는 말은 이 가능한 변화량이 $x_t$에 따라 달라진다는 뜻이다. `endogenous`라는 말은 controller의 현재 action이 chamber를 충전하거나 소모해 **미래의 authority 자체도 바꾼다**는 뜻이다. 이 시스템에서는 action이 현재 displacement만 제어하는 것이 아니라 이후에 사용할 pneumatic control resource도 함께 관리한다.

reserve

\[
\rho_+=P_{ch,+}-P_{act,+},
\qquad
\rho_-=P_{act,-}-P_{ch,-}
\]

는 이 집합을 결정하는 핵심 measurable feature지만 authority 그 자체는 아니다. Authority에는 reserve 외에도 pressure level, displacement/volume, velocity direction, leakage, valve/action history와 time horizon $K$가 포함된다. 따라서 `reserve 두 개를 observation에 붙였다`만으로는 state-dependent control-authority method라고 부르지 않는다.

실제 방법에서는 전체 hidden state 대신 측정 가능한 proxy

\[
\phi_t=[P_t,D_t,\dot D_t,\dot P_t,\rho_t,u_{t-H:t-1}]
\]

에서 

\[
\widehat{\mathcal A}_K(\phi_t)
\]

를 short rollout/action sweep으로 추정한다. 원하는 reference rate 

\[
\dot r_t
\]

가 이 집합을 얼마나 사용하는지는 capacity utilization으로 나타낼 수 있다. 이때 $\widehat{\mathcal A}_K$는 원점을 포함하는 conservative convex/star-convex under-approximation 또는 방향별 signed bound로 사용해야 한다.

\[
\eta_t
=
\inf\{\eta\ge0:\dot r_t\in\eta\,\widehat{\mathcal A}_K(\phi_t)\}
\]

- $\eta_t<1$: 현재 추정 authority 안에서 지원되는 요구
- $\eta_t\approx1$: edge-of-capacity 요구
- $\eta_t>1$: 현재 상태에서는 nominally unsupported한 요구

이 값은 reference를 쉬운 곡선으로 바꾸기 위한 점수로만 쓰지 않는다. 다음 세 위치에서 서로 다른 역할을 가질 수 있다.

1. **reference certification/curriculum**: dynamic oscillator의 원래 displacement target을 평가 전에는 바꾸지 않고, train/ID/edge 난이도를 사전에 분류한다.
2. **policy conditioning 또는 auxiliary prediction**: raw reserve만 제공하는 baseline과 달리 앞으로 가능한 signed output-rate를 policy가 구별하게 한다.
3. **safety/authority management**: 당장 error를 줄이는 action과 미래 reserve를 보존하는 action의 trade-off를 명시한다.

이 개념의 novelty는 아직 성립했다고 쓰면 안 된다. Dynamically admissible trajectory를 sampled reachable set에 projection하는 일반 연구는 이미 존재한다([DDAT, RSS 2025](https://www.roboticsproceedings.org/rss21/p078.html)). 현재 연구의 차별점 후보는 shared pneumatic supply의 reserve와 delayed valve state를 포함하는 **측정 가능한 signed authority**, 그리고 이것이 real displacement tracking 실패와 transfer gap을 설명하는지를 보이는 데 있다.

최소 반증 실험은 다음과 같다.

1. 같은 $D$, reference error와 유사한 pressure 조건에서 reserve, velocity direction, recent action을 달리한다.
2. 동일한 bounded action sweep을 적용해 $\max/\min\;\dot P_{act}$와 $\dot D$를 측정한다.
3. global rate limit, raw reserve, history-only와 $\widehat{\mathcal A}_K$의 held-out prediction력을 비교한다.
4. baseline controller의 saturation, reserve collapse와 tracking failure가 $\eta_t$에 따라 증가하는지 확인한다.
5. simulator에서 얻은 authority의 절대값뿐 아니라 상태별 ordering과 failure prediction이 real에서도 유지되는지 검증한다.

이 관계가 약하거나 raw reserve만으로 같은 설명력이 나오면 `state-dependent control authority`를 메인 novelty로 내리지 않고 reserve-aware engineering feature로 축소한다.

### 4.5 K=2: 짧은 temporal abstraction

현재 K=2 policy는 다음 두 action을 한 번에 출력한다.

\[
\pi(o_t)=[a_t,a_{t+1}],\qquad a_i\in\mathbb{R}^{6}.
\]

두 action은 같은 값으로 hold되지 않고 20 ms 간격으로 simulator에 순서대로 적용된다. chunk transition의 reward와 discount는 다음과 같다.

\[
R_t=r_t+\gamma r_{t+1},
\qquad
\Gamma=\gamma^2.
\]

K=2는 physical valve update를 50 Hz로 유지하면서 policy decision을 25 Hz로 만든다. 이것은 두 step 동안 open-loop라는 단점이 있으므로 범용적 우월성을 주장하면 안 된다. 대신 **빠른 policy switching과 느린 pneumatic response 사이의 시간척도 mismatch를 줄이는 짧은 temporal abstraction**으로 해석한다. Action chunking 자체는 RSS 2023의 ACT 등에서 이미 사용되었으므로, 그 자체가 novelty가 아니라 이 시스템에서 물리 observation과 함께 왜 필요한지와 ablation이 중요하다 ([RSS 2023](https://doi.org/10.15607/RSS.2023.XIX.016)).

### 4.6 최종 controller 후보

lib3 결과에 기반한 최소 구조는 다음과 같다.

\[
o_t=
[P_{t-H:t},u_{t-H:t-1},r_{t-H:t+F},\dot P_t,\rho_t]
\]

\[
[a_t,a_{t+1}]=\pi_\theta(o_t).
\]

논문의 algorithmic claim은 `새로운 SAC`가 아니라 다음으로 제한하는 것이 안전하다.

> Physics-derived state reconstruction and short temporal action abstraction for a partially observable, multi-stage pneumatic plant.

---

## 5. 제안하는 핵심 연구 질문과 가설

### RQ1. 채택한 동적 밸브 모델을 6개 valve에 물리적으로 일관되게 calibration할 수 있는가?

가설 H1:

> 동일한 dynamic model class에서도 fit-only calibration은 6개 valve 사이의 parameter compensation 또는 비물리적 해를 선택할 수 있다. Feasibility-aware calibration은 held-out flow accuracy를 유지하면서 command monotonicity, full-open authority, measured transient envelope와 long-rollout consistency를 개선할 수 있다.

이 가설이 입증되고 동료 논문의 calibration 범위와 중복되지 않을 때만 parameter tuning을 독립적인 **six-valve feasibility-aware calibration method**로 주장한다. 그렇지 않으면 reproducible simulator calibration protocol과 supporting contribution으로 둔다.

### RQ2. 선행 MPObs+CAPS가 6-valve actuator plant에서 왜 충분하지 않은가?

가설 H2:

> Predictor-only observation은 hidden valve/chamber state mismatch에 취약하고, temporal smoothing만으로는 pressure reserve를 복원할 수 없다. pressure/action history와 pressure-rate/reserve를 제공하면 action direction과 tracking이 개선된다.

### RQ3. K=2가 왜 효과가 있으며 항상 필요한가?

가설 H3:

> K=2는 fast switching을 줄이고 짧은 delayed credit를 묶어 주지만, reserve observation이 없으면 chamber overcharge를 막지 못한다. 즉 K=2와 physics observation은 서로 다른 실패 원인을 해결한다.

### RQ4. 더 정확하고 feasible한 simulator가 real fine-tuning 비용을 줄이는가?

가설 H4:

> constrained lib4에서 사전학습한 policy는 단순 또는 fit-only simulator에서 학습한 policy보다 zero-shot 성능 저하가 작고, 동일 real interaction budget에서 더 빠르게 fine-tuning된다.

H4가 최종적으로 입증되어야 simulator 연구와 RL 연구가 하나의 논문으로 강하게 연결된다.

### RQ5. state-dependent control authority가 기존 controller의 실패와 sim-to-real gap을 설명하는가?

가설 H5:

> 같은 reference error에서도 short-horizon signed output capacity는 chamber–actuator reserve, displacement/velocity와 valve/action history에 따라 유의하게 달라진다. 이 authority를 추정한 capacity utilization은 global rate limit이나 raw reserve만 사용할 때보다 saturation, chamber overcharge와 tracking failure를 더 잘 예측하며, authority-aware curriculum/policy는 unseen dynamic reference와 real plant에서 graceful degradation을 개선한다.

H5는 현재 가장 유망한 중심 contribution 가설이지만 아직 검증되지 않았다. 특히 `reserve feature 추가`, `feasible reference 생성`, `history 사용`을 각각 authority-aware method와 동일시하지 않는다. raw reserve 대비 authority-set estimation의 추가 설명력, sim-real trend 보존과 실제 controller 개선이 함께 확인되어야 한다.

---

## 6. 논문에서 주장할 기여

### Contribution 1 — Six-valve feasibility-aware calibration and coupled integration

연구실 동료가 제안한 nonlinear/hysteretic/asymmetric second-order valve model을 6개 실제 valve에 개별 보정하고, 다음 physical consistency 조건을 사용한다.

- stable and responsive \(\omega_n,\zeta\) range
- nonnegative effective area/flow
- command-to-flow monotonicity
- sufficient full-open authority
- peak and frequency-response consistency

핵심 결과 후보는 “새로운 2차 밸브 모델을 제안했다”가 아니다. **기존 동적 모델을 6개 coupled valve에 재현 가능하게 보정하고, coupled simulator에서 유효한 parameter set을 선택했다**는 것이다. 이 항목의 독립 novelty 강도는 동료 논문의 방법·데이터와의 중복 및 fit-only ablation 결과에 따라 결정한다.

### Contribution 2 — Real-time coupled dual-pressure soft-actuator simulator

single-pump dual-pressure source, six dynamic valves, chamber-actuator mass flow, leakage, variable volume, force, displacement를 하나의 real-time ODE simulator에 결합한다.

이 기여는 `모델 항이 많다`가 아니라 다음 조건을 만족할 때 강해진다.

- 각 subsystem ablation이 있음
- step/random/chirp에서 real 데이터를 검증함
- long-rollout drift와 frequency response를 검증함
- downstream controller와 sim-to-real 결과까지 연결함

### Contribution 3 — State-dependent-control-authority-aware temporal controller (검증 후 채택)

선행 direct SAC+MPObs+CAPS의 실패를 partial observability, predictor mismatch, reserve blindness, temporal mismatch로 분해한다. 측정 가능한 pressure/action history, pressure rate, reserve와 displacement motion에서 short-horizon signed output authority를 추정하고, 이를 reference 난이도 분류와 temporal policy에 연결한다. K=2는 authority 추정 자체가 아니라 느린 pneumatic response와 빠른 policy switching의 시간척도 mismatch를 줄이는 별도 구성요소로 비교한다.

이 부분은 새로운 범용 RL algorithm으로 과장하지 않는다. 또한 raw reserve 두 개를 observation에 추가한 결과만으로 contribution을 주장하지 않는다. **현재 state에서 실제로 가능한 signed output-rate가 달라지고, 이 차이를 명시적으로 추정·사용하는 것이 기존 controller의 실패와 real transfer를 개선한다는 인과관계**가 기여 후보다.

### Contribution 4 — Model quality와 real adaptation cost의 인과 검증

최종 기여는 다음 비교로 닫는다.

```text
simple/fit-only simulator policy
vs constrained lib4 policy
vs constrained lib4 + limited real fine-tuning
vs real-from-scratch 또는 PID/model-based baseline
```

동일 real interaction budget에서 tracking, stability, safety, adaptation speed를 비교해야 한다. Real fine-tuning 자체는 선행 논문과 기존 robot-learning 문헌에도 있으므로 novelty가 아니다. **Constrained dynamic identification과 temporal physics-aware policy가 zero-shot gap 및 필요한 real adaptation 양을 얼마나 줄이는지**가 기여가 되어야 한다.

---

## 7. “세계 최초” 주장에 대한 판단

### 7.1 그대로는 주장하기 어려운 문장

다음 문장은 최근 선행 연구 때문에 방어하기 어렵다.

> 세계 최초로 soft pneumatic actuator simulator를 만들고 RL을 학습했다.

이미 다음 연구들이 인접 영역을 선점하고 있다.

- pneumatic soft system의 simulation-based 설계와 pressure control: RA-L 2021 ([Xavier et al.](https://doi.org/10.1109/LRA.2021.3086425))
- learned dynamic model에서 soft manipulator DRL 학습: RA-L 2022 ([Centurelli et al.](https://doi.org/10.1109/LRA.2022.3146903))
- soft robot simulation, RL/co-optimization, zero-shot sim-to-real: RSS 2022 ([Schaff et al.](https://doi.org/10.15607/RSS.2022.XVIII.062))
- pneumatic soft arm의 dynamic task와 simulation-trained RL: CoRL 2023 ([Jitosho et al.](https://openreview.net/forum?id=rPye6EZxmI))
- soft robot simulator의 learned residual physics와 sim-to-real: RA-L 2024 ([Gao et al.](https://doi.org/10.1109/LRA.2024.3446287))

따라서 broad first claim은 사용하지 않는다.

### 7.2 인접 연구도 real→sim→real인가?

| 연구 | real data로 simulator 구성/보정 | simulation에서 RL/최적화 | real 배포 | real policy fine-tuning | 정확한 분류 |
|---|---|---|---|---|---|
| 선행 RA-L 2025 | valve/check-valve 계수 식별 및 simulator 검증 | SAC+MPObs+CAPS | 수행 | **수행, 102,400 step** | 가장 직접적인 real→physics sim→RL→real adaptation 선행 연구 |
| [Centurelli et al., RA-L 2022](https://doi.org/10.1109/LRA.2022.3146903) | real motor-babbling data로 LSTM forward dynamics 학습 | LSTM environment에서 TRPO | 수행 | 미수행 | real→**learned sim**→RL→real |
| [Schaff et al., RSS 2022](https://doi.org/10.15607/RSS.2022.XVIII.062) | real PneuNet 압력–변형 데이터로 action scaling 보정; 일부 물성은 문헌/정성 tuning | FEA/MOR에서 design-control co-optimization | zero-shot | 미수행 | 부분적으로 real-calibrated physics sim→optimization→real |
| [Jitosho et al., CoRL 2023](https://openreview.net/forum?id=rPye6EZxmI) | 약 30초 real data로 stiffness, damping, pressure–torque mapping 식별 | fit physics model에서 RL | zero-shot | 미수행 | real→system-identified physics sim→RL→real |
| [Gao et al., RA-L 2024](https://doi.org/10.1109/LRA.2024.3446287) | sparse real marker data로 FEM residual force 학습 | RL controller 학습이 핵심이 아님 | real 데이터에서 corrected simulator 검증 | 해당 없음 | real→residual-physics simulator이며 RL pipeline은 아님 |

따라서 `real2sim2real`은 현재 연구의 중요한 **파이프라인**이지만 그 자체가 novelty는 아니다. 현재 연구가 차별화되는 후보는 다음 세 가지의 연결이다.

1. 선행 동적 밸브 모델의 6-valve feasibility-aware calibration과 coupled actuator simulator 통합
2. single-pump dual-pressure source와 6-valve transmission, leakage, variable-volume soft actuator를 결합한 real-time simulator
3. 그 simulator에서 선행 controller의 실패 원인을 hidden state, reserve blindness, temporal mismatch로 분리하고 최소 temporal physics-aware policy를 도출한 결과

### 7.3 검토 후 사용할 수 있는 좁은 주장 후보

체계적 literature review에서 반례가 없을 때만 다음과 같이 제한한다.

> To the best of our knowledge, this is the first control-oriented simulation and learning framework that integrates a single-pump simultaneous positive/negative pressure source, six dynamically identified valves, and displacement-dependent dual-chamber soft-actuator dynamics for sim-to-real control.

이 문장의 핵심은 `soft actuator + RL`이 아니라 다음 교집합이다.

```text
single-pump simultaneous dual pressure
∩ six individually calibrated dynamic valves
∩ six-valve chamber–actuator transmission
∩ variable-volume/displacement soft actuator
∩ sim-to-real RL control
```

그래도 “first”는 reviewer가 가장 쉽게 공격하는 표현이다. RA-L 투고에서는 first claim 없이도, 명확한 시스템 gap과 quantitative evidence만으로 충분할 수 있다.

---

## 8. 반드시 수행할 ablation

### 8.1 Identification ablation

이 ablation은 사용자의 calibration method가 연구실 동료 논문과 독립적인 경우에만 현재 논문의 정식 ablation으로 사용한다. Static-vs-dynamic model 비교는 동료 논문의 결과를 인용하는 background baseline이며, 동일 데이터·figure를 허가 없이 중복 사용하지 않는다.

| ID | 식별 방식 | 목적 |
|---|---|---|
| I0 | fit-only SSE/RMSE | 높은 fit metric이 만드는 비물리 해를 재현 |
| I1 | I0 + \(\omega_n,\zeta\) bounds | 동특성 안정성 효과 |
| I2 | I1 + full-open/positive-area constraint | control authority 보장 효과 |
| I3 | I2 + command-flow monotonicity | direction reversal 제거 효과 |
| I4 | I3 + peak/HF 또는 frequency-weighted term | transient와 주파수 응답 효과 |

가장 중요한 비교는 test RMSE가 비슷한 I0와 I4에서 physical consistency와 coupled-simulator rollout이 달라지는지 확인하는 것이다. 차이가 없다면 이 항목을 메인 novelty에서 제외하고 reproducible calibration detail로 축소한다.

### 8.2 Simulator fidelity ablation

| ID | 모델 | 확인할 효과 |
|---|---|---|
| S0 | 채택한 6-valve dynamic source/chamber model, actuator 없음 | 동료 연구에서 현재 연구가 출발하는 baseline |
| S1 | S0 + chamber–actuator mass-flow transmission, fixed volume | 다단 pressure transmission 효과 |
| S2 | S1 + leakage/cross-leakage | steady-state drift와 pressure coupling 효과 |
| S3 | S2 + variable actuator volume | pressure–displacement volume coupling 효과 |
| S4 | S3 + force/displacement dynamics와 실제 적분 protocol | 최종 lib4 coupled simulator |

각 단계에서 pressure RMSE만 제시하지 말고 flow, phase, displacement, long-rollout drift를 함께 비교한다.

### 8.3 Controller ablation

| ID | 구조 | 분리되는 질문 |
|---|---|---|
| C0 | Direct SAC | 선행 direct policy baseline |
| C1 | + pressure history | pressure temporal context 효과 |
| C2 | + action history | hidden valve state 근사 효과 |
| C3 | C2 + pressure rate only | flow-direction 정보 효과 |
| C4 | C2 + reserve only | control authority 정보 효과 |
| C5 | C2 + rate + reserve | 두 physics feature의 결합 효과 |
| C6 | C2 + K=2 | temporal abstraction 단독 효과 |
| C7 | C5 + K=2 | 현재 유력 구조 |
| C8 | C7 + CAPS 또는 MPObs 각각 | 선행 contribution의 조건부 효과 |

최소 3개 train seed와 두 reference가 필요하다. reference는 하나의 feasible 학습 기준과 하나의 legacy/high-rate stress 기준을 사용하되, 모든 평가 plot은 동일한 2048 physical step 규칙으로 통일한다.

### 8.4 Sim-to-real ablation

| ID | 초기 policy/모델 | real 사용량 | 확인할 질문 |
|---|---|---:|---|
| R0 | PID 또는 기존 model-based baseline | tuning only | classical baseline |
| R1 | source-level 또는 uncoupled baseline에서 학습한 RL | 0 | actuator coupling 누락의 zero-shot gap |
| R2 | S4에서 학습한 RL | 0 | control-relevant model의 zero-shot 효과 |
| R3 | S4 RL + real fine-tuning | 고정 budget N | 제안 pipeline의 sample efficiency |
| R4 | real from scratch | 동일 budget N | simulation pretraining 가치 |

가능하면 real interaction budget을 1×, 2×, 5×로 나눠 adaptation curve를 제시한다. 최종 novelty는 최저 RMSE 한 점보다 **같은 real data에서 얼마나 빨리 좋아지는가**에서 더 강해질 수 있다.

---

## 9. 평가 지표

### 9.1 식별과 simulator

- held-out flow/pressure RMSE, NRMSE, R²
- command–flow monotonicity violation count와 최대 violation
- full-open flow authority
- step rise time, settling time, overshoot
- chirp gain/phase error 또는 주파수별 magnitude/phase
- peak-flow error와 high-frequency ratio
- 40 s 및 장기 rollout drift
- displacement와 velocity RMSE
- numerical divergence와 physical-bound violation

### 9.2 controller

- actuator POS/NEG RMSE와 MAE
- reference–actual direction agreement
- optimal lag와 rate correlation
- steady-state offset
- \(\mathrm{mean}|\Delta u|\), \(\mathrm{mean}|\Delta^2u|\), action spectrum
- chamber reserve violation 시간 비율
- chamber overpressure/underpressure
- inlet/outlet conflict 또는 wasted mass flow
- seed별 평균, 표준편차, confidence interval

### 9.3 sim-to-real

- zero-shot degradation: real metric − simulation metric
- fixed-budget fine-tuning improvement
- 목표 성능 도달까지 필요한 real step 또는 시간
- unsafe command/pressure violation 수
- payload, leakage, initial state, reference frequency 변화에 대한 robustness

---

## 10. 현재 novelty의 강도와 남은 증거

| 주장 | 현재 상태 | 논문화 조건 |
|---|---|---|
| 2차 동적 밸브 model 자체 | 연구실 동료 논문 기여 | 현재 논문에서는 citation과 adopted component로만 사용 |
| 6-valve calibration protocol | 사용자 구현 근거 있음 | 동료 논문과 중복 확인, fit-only 비교, 전 채널 재현성 |
| coupled 6-valve soft-actuator simulator | 구현 근거 있음 | transmission/leakage/volume/displacement ablation과 real 검증 |
| variable-volume soft actuator dynamics | lib4 구현 근거 있음 | displacement 포함 sim-real 검증 |
| direct SAC/MPObs/CAPS 실패 원인 | lib3 ablation 근거 있음 | rate와 reserve 분리, 3 seeds 통계 |
| history+physics+K2 개선 | lib3에서 강한 초기 근거 있음 | lib4 재현 및 real 배포 |
| sim pretraining이 real fine-tuning을 줄임 | 아직 미완료 | fixed-budget R1–R4 비교 |
| 세계 최초 | 미검증이며 broad claim은 반례 있음 | 좁은 교집합에 대한 systematic review 필요 |

현재 연구는 novelty가 없는 것이 아니라 **핵심 인과 주장을 아직 끝까지 검증하지 않은 상태**다. 구현 난이도 자체는 novelty의 기준이 아니다. 간단한 constraint라도 기존 평가가 놓치던 failure를 재현 가능하게 제거하고 downstream control/transfer를 개선하면 충분히 연구 기여가 된다.

---

## 11. RA-L 논문 구성안

### Introduction

1. single-pump dual-pressure source는 이동형 soft robot에 유리하다.
2. soft actuator까지 연결하면 6-valve, chamber reserve, leakage, variable volume, delayed valve dynamics가 결합된다.
3. 연구실 선행 연구의 동적 valve model을 사용해도 actuator가 연결되면 hidden state, reserve, leakage와 variable volume이 새로운 제어 문제를 만든다.
4. 선행 direct SAC+MPObs+CAPS는 이 다단 plant에서 hidden state와 control authority를 충분히 표현하지 못한다.
5. 본 연구는 six-valve calibration/integration, coupled simulator, temporal physics-aware policy와 sim-to-real 검증을 제안한다.

### Method I — Adopted dynamic-valve model and six-valve calibration

- 연구실 동료의 valve equation과 contribution boundary
- excitation data: step, random, chirp
- fit objective와 feasibility constraints
- parameter bounds의 물리적 근거
- fit-only 대비 constrained identification

### Method II — Coupled soft actuator simulator

- pump/chamber/6 valves
- mass flow and leakage
- variable volume and displacement dynamics
- integration and real-time speed

### Method III — Temporal physics-aware RL

- 왜 기본 observation이 non-Markov인지
- history, pressure rate, reserve의 식 기반 유도
- K=2 transition과 reward 정의
- SAC objective와 기존 MPObs/CAPS baseline

### Experiments

1. adopted valve model의 six-channel calibration/validation
2. calibration ablation(독립성이 확인된 경우)
3. coupled simulator component ablation
4. lib3/lib4 controller ablation
5. zero-shot sim-to-real
6. fixed-budget real fine-tuning
7. payload/leakage/reference-frequency robustness

### Discussion

- K=2가 항상 우수하지 않은 조건
- predictor fidelity에 따른 MPObs 효과
- CAPS smoothness와 tracking trade-off
- stability guarantee의 한계
- 다른 pneumatic soft actuator로의 일반화 가능성

---

## 12. 추천 title과 contribution 문장

### 가장 추천하는 title

H5가 sim/real에서 검증될 경우 가장 구별되는 후보:

> **Control-Authority-Aware Sim-to-Real Learning for a Shared Dual-Pressure Soft Actuator**

조금 더 system-paper 성격을 유지하는 후보:

> **Temporal Physics-Aware Reinforcement Learning for Sim-to-Real Control of a Six-Valve Dual-Pressure Soft Actuator**

Simulator integration까지 강조하려면:

> **Coupled Simulation and Temporal Reinforcement Learning of a Six-Valve Dual-Pressure Soft Actuator**

`dynamic valve model` 또는 `second-order valve modeling`을 title 전면에 두면 연구실 동료 논문의 contribution과 충돌할 수 있으므로 현재는 피한다.

### 논문 contribution 초안

1. Building on a previously developed dynamic valve model, we establish a reproducible six-valve calibration and integration pipeline for a single-pump dual-pressure source connected to a soft actuator. This contribution is retained only to the extent that its calibration method is distinct from the valve-model paper.
2. We develop a real-time coupled simulator that combines six calibrated valve dynamics with chamber–actuator mass-flow transmission, pressure-dependent leakage, displacement-dependent volumes, and pressure–force–motion coupling.
3. We formulate the actuator's short-horizon signed output-rate set as state-dependent control authority, estimate it from measurable pressure/motion/history features, and use its utilization to condition reference curricula and a temporal policy. This contribution is retained only if it explains failures and improves transfer beyond raw reserve/history baselines.
4. We evaluate the coupled simulator and policy through controlled component/controller ablations, zero-shot real deployment, and fixed-budget real-world fine-tuning.

---

## 13. Abstract 논리 초안

> Extending a simultaneous positive/negative pressure source to a soft actuator creates endogenous actuation limits: the same valve command can produce different pressure and motion rates depending on chamber reserve, displacement/velocity, and delayed valve state, while current actions also reshape future authority. We formulate this state-dependent control authority as a constrained short-horizon signed output-rate set and estimate it from measurable pressure, motion, and action-history features. Building on a calibrated coupled simulator, we test whether authority utilization explains the scaling failure of the preceding SAC/MPObs/CAPS controller and use it to construct capacity-certified dynamic references and a temporal policy. Controlled ablations against global-rate, raw-reserve, history-only, and prior-controller baselines isolate the contribution of authority estimation. Finally, identical unseen displacement references in simulation and hardware evaluate zero-shot degradation and fixed-budget real adaptation.

마지막 문장의 `demonstrate`와 구체적인 수치는 lib4 real 결과가 끝난 뒤 확정한다.

---

## 14. 최근 고급 저널·학회 문헌과 차별점

| 연구 | Venue | 이 연구와의 관계 | 남는 gap |
|---|---|---|---|
| `[연구실 동료의 dynamic valve-model paper — 서지정보 필요]` | 확인 필요 | static model의 확장 실패와 asymmetric second-order/hysteretic valve dynamics를 제공하는 직접 기반 | 6-valve actuator-level integration, controller scaling failure와 RL 해결은 현재 연구 범위 |
| [Xavier et al., Design and Control of Pneumatic Systems for Soft Robotics](https://doi.org/10.1109/LRA.2021.3086425) | RA-L 2021 | pneumatic system 설계와 pressure control을 simulation으로 연결 | single-pump dual pressure, 6-valve dynamic identification, RL transfer가 아님 |
| [Centurelli et al., Closed-loop Dynamic Control of a Soft Manipulator Using DRL](https://doi.org/10.1109/LRA.2022.3146903) | RA-L 2022 | LSTM forward model에서 TRPO를 학습한 soft manipulator controller | control-relevant physical valve identification과 dual-pressure supply가 아님 |
| [Graule et al., SoMoGym](https://doi.org/10.1109/LRA.2022.3149580) | RA-L 2022 | soft-robot controller/RL benchmark simulator | 특정 pneumatic power-source와 sim-real identification 기여가 아님 |
| [Schaff et al., Soft Robots Learn to Crawl](https://doi.org/10.15607/RSS.2022.XVIII.062) | RSS 2022 | efficient soft simulation, RL co-optimization, zero-shot transfer | actuator pressure-source dynamics와 control-relevant valve constraints가 아님 |
| [Djeumou et al., Physics-Informed Architectures and Constraints](https://proceedings.mlr.press/v168/djeumou22a.html) | L4DC 2022 | physical side information과 constraints가 data efficiency/generalization을 높임 | 본 시스템의 feasible valve identification과 폐루프 효과는 별도 문제 |
| [Richards et al., Control-Oriented Meta-Learning](https://doi.org/10.1177/02783649231165085) | IJRR 2023 | regression objective보다 closed-loop objective가 유리할 수 있음을 보임 | soft pneumatic system의 explicit physical constraints와 sim-to-real 검증이 아님 |
| [Jitosho et al., RL for Agile Soft Robot Arms](https://openreview.net/forum?id=rPye6EZxmI) | CoRL 2023 | simulator 수정과 domain randomization으로 dynamic soft-arm zero-shot transfer | high-fidelity dual-pressure valve/actuator identification이 중심이 아님 |
| [Zhao et al., ACT](https://doi.org/10.15607/RSS.2023.XIX.016) | RSS 2023 | action chunking으로 effective horizon과 temporal consistency를 개선 | imitation learning이며 delayed pneumatic SAC와는 구조·목표가 다름 |
| [Gao et al., Learned Residual Physics](https://doi.org/10.1109/LRA.2024.3446287) | RA-L 2024 | soft-robot simulator의 residual physics로 sim-real gap을 감소 | explicit valve feasibility와 pressure-source control이 아님 |
| [Wu et al., Auxiliary-Delayed RL](https://proceedings.mlr.press/v235/wu24af.html) | ICML 2024 | delay가 RL state augmentation과 sample efficiency를 어렵게 함 | 본 연구는 system-specific measurable physics state를 사용 |
| [Park et al., Simultaneous Positive/Negative Pressure Generation](https://doi.org/10.1109/LRA.2025.3577422) | RA-L 2025 | 직접 선행 연구: dual-pressure simulator + SAC/MPObs/CAPS | soft actuator, 6-valve chamber–actuator transmission, leakage/variable volume/displacement가 없음 |

이 문헌 지도에서 보이는 핵심은 soft robot simulator, RL, action chunking, physics constraints 각각은 이미 존재한다는 점이다. 현재 연구가 새로워지려면 **이들을 나열하는 것이 아니라, control-relevant fidelity failure를 하나의 새로운 문제로 정의하고 각 방법이 그 문제에서 왜 필요한지 실험으로 연결해야 한다.**

---

## 15. 다음 실행 우선순위

### 1순위 — 연구실 내 contribution 경계와 citation을 먼저 고정

1. 연구실 동료 논문의 title, submission/publication 상태, citation 정보를 확정한다.
2. static-model failure, second-order equation, dataset, tuning method 중 동료 논문이 주장하는 범위를 표로 합의한다.
3. 현재 사용자의 six-valve calibration/constraint 코드와 데이터가 독립 contribution인지 확인한다.
4. 독립성이 확인될 때만 I0–I4를 현재 논문의 ablation으로 수행하고, 그렇지 않으면 calibration detail로 축소한다.

### 2순위 — lib4 coupled simulator를 논문 버전으로 동결

```text
real dataset ID
→ identification config/result
→ lib4 source commit/build hash
→ RL config/model
→ simulation/real evaluation
```

이 provenance chain이 없으면 좋은 결과가 나와도 어떤 simulator에서 학습했는지 증명하기 어렵다.

### 3순위 — 최소 controller ablation

lib4에서 모든 과거 실험을 반복하지 말고 C0, C2, C3, C4, C5, C6, C7, C8을 우선한다. 특히 rate-only와 reserve-only를 분리해 교수님의 질문에 직접 답한다.

### 4순위 — real fine-tuning protocol을 사전에 고정

- zero-shot 평가 횟수
- fine-tuning step budget
- safety pressure bounds
- reset 방식
- reference seed와 frequency range
- real-from-scratch baseline budget

결과를 본 뒤 protocol을 바꾸지 않도록 먼저 문서화한다.

---

## 16. 최종 판단

현재 작업은 `안 되어서 이것저것 붙인 연구`로 끝날 수도 있고, 충분히 강한 후속 RA-L 연구가 될 수도 있다. 차이는 추가한 모듈 수가 아니라 다음 세 문장을 실험으로 증명하는 데 있다.

1. **동적 밸브 모델을 포함한 source-level controller가 soft actuator가 연결된 multi-stage plant에서 왜 그대로 확장되지 않는가?**
2. **왜 pressure/action history, rate, reserve, K=2가 각각 다른 failure를 해결하는가?**
3. **그 결과 lib4에서 zero-shot gap과 real fine-tuning에 필요한 데이터·위험이 실제로 줄어드는가?**

이 세 질문에 답하면 simulator의 정밀화, parameter constraints, lib3의 실패 실험, K=2, lib4, real fine-tuning이 하나의 연구로 연결된다. 반대로 최종 tracking RMSE만 제시하면 reviewer에게는 복잡한 simulator와 여러 RL 기믹을 조합한 사례 연구로 보일 가능성이 높다.

따라서 현재 가장 적절한 연구 정체성은 다음이다.

> **A coupled simulation and temporal physics-aware learning framework for sim-to-real control of a six-valve dual-pressure soft actuator.**

---

## 17. Feasible reference와 OOD를 novelty로 가져갈 수 있는가

### 17.1 먼저 OOD의 종류를 분리해야 한다

`feasible reference가 OOD를 줄인다`는 방향은 가능하지만, OOD를 한 단어로 묶으면 reviewer가 바로 반박할 수 있다. 이 연구에는 서로 다른 세 분포 이동이 있다.

| 종류 | 이 연구에서의 의미 | feasible reference의 직접 효과 |
|---|---|---|
| target/context OOD | 학습 reference의 압력·변위·rate·주파수 조합이 plant가 도달했거나 학습한 목표 분포 밖에 있음 | 직접 줄일 수 있음 |
| state-action visitation OOD | 불가능한 target을 쫓다가 chamber 고갈, valve saturation, 상충 action 등 replay buffer에 편향된 상태가 쌓임 | 간접적으로 줄일 수 있음 |
| dynamics/domain OOD | simulation의 valve/leakage/volume/load dynamics와 실제 장치가 다름 | reference만으로는 해결할 수 없음 |

따라서 논문에서 사용할 수 있는 정확한 문장은 다음과 같다.

> **The proposed reference construction reduces target-induced distribution shift by restricting training trajectories to a state-conditioned, empirically admissible pressure–motion envelope. It may consequently reduce pathological state-action visitation, but it does not by itself eliminate the simulator-to-real dynamics gap.**

즉 feasible reference는 domain randomization의 대체물이 아니다. 실제 system parameter mismatch는 별도의 simulator validation, parameter ensemble, domain randomization 또는 real fine-tuning으로 다뤄야 한다. simulation policy가 simulator 오차를 악용하는 `simulation optimization bias`는 오래전부터 구분되어 온 문제다 ([Muratore et al., CoRL 2018](https://proceedings.mlr.press/v87/muratore18a.html)).

### 17.2 왜 불가능한 reference가 RL 학습을 왜곡하는가

plant를 다음처럼 두자.

\[
x_{t+1}=f_{\theta}(x_t,u_t),
\qquad
y_t=h(x_t),
\qquad
u_t\in\mathcal U,
\quad x_t\in\mathcal X.
\]

초기 상태 \(x_0\)에서 reference \(r_{0:T}\)가 feasible하다는 강한 정의는, 제약을 만족하면서 reference 허용 오차 안에 들어가는 action sequence가 존재한다는 것이다.

\[
\exists u_{0:T-1}\in\mathcal U:
\quad
x_{t+1}=f_\theta(x_t,u_t),
\quad x_t\in\mathcal X,
\quad \lVert h(x_t)-r_t\rVert\le\epsilon.
\]

reference가 이 집합 밖이면 tracking error에는 두 종류가 섞인다.

\[
e_t
=
\underbrace{e_t^{\mathrm{policy}}}_{\text{controller가 줄일 수 있는 오차}}
+
\underbrace{e_t^{\mathrm{infeasible}}}_{\text{어떤 bounded controller도 제거할 수 없는 오차}}.
\]

이때 reward \(-\lVert e_t\rVert\)만 보면 좋은 policy와 나쁜 policy를 분리하기 어렵다. valve를 끝까지 열어도 못 따라가는 구간에서 critic은 서로 다른 action에 비슷하게 큰 음의 return을 주고, saturation 또는 고주파 switching이 합리적인 탐색처럼 남을 수 있다. 따라서 feasible reference는 단순히 task를 쉽게 만드는 것이 아니라, **정책이 개선할 수 있는 오차와 plant 한계에서 오는 오차를 분리하는 실험 설계**다.

그렇다고 infeasible reference를 전부 버리면 안 된다. 학습의 주분포에서는 avoidable error를 분리하되, test에서는 input saturation과 외란 아래에서 얼마나 graceful하게 성능이 저하되는지 봐야 한다. DATT처럼 potentially infeasible trajectory까지 adaptive structure로 추종하는 학습 제어 연구도 있으므로 ([Huang et al., CoRL 2023](https://proceedings.mlr.press/v229/huang23a.html)), 현재 연구도 legacy random을 OOD/stress suite로 유지해야 한다. 핵심은 feasible와 infeasible score를 하나의 평균 RMSE로 섞지 않는 것이다.

이 관점은 goal-conditioned RL에서 `현재 상태에서 가능한 goal을 생성해야 한다`는 문제와 맞닿아 있다. Contextual Imagined Goals는 불가능한 goal proposal이 학습을 막을 수 있음을 보이고 현재 상황에서 feasible하면서 다양한 goal을 생성한다 ([Nair et al., CoRL 2020](https://proceedings.mlr.press/v100/nair20a.html)). Imagined Subgoals는 valid-state distribution 밖의 subgoal이 실제로 도달 불가능할 수 있어 value-based reachability와 distribution constraint를 사용한다 ([Chane-Sane et al., ICML 2021](https://proceedings.mlr.press/v139/chane-sane21a.html)). AdaGoal은 아예 \(L\) step 안에 reachable한 goal 집합을 학습 대상으로 정의한다 ([Tarbouriech et al., AISTATS 2022](https://proceedings.mlr.press/v151/tarbouriech22a.html)).

다만 이 논문들은 대체로 goal reaching이나 subgoal generation 문제다. 현재 연구는 **연속시간에 가까운 pressure–displacement tracking reference 전체가 state-dependent actuator authority를 만족해야 한다**는 점에서 차이가 있다.

### 17.3 현재 lib3 reference는 얼마나 feasible한가

현재 기준 profile은 [lib3 capacity reference config](../pneu_rl/config/reference/lib3_capacity_random_medium_16x4x512_seed61098.yaml)이며, 생성기는 [generate_lib3_capacity_random_reference.py](../pneu_rl/scripts/generate_lib3_capacity_random_reference.py)다. 현재 방식은 다음 네 단계를 갖는다.

1. 여러 pressure operating point에서 six-valve action sweep을 수행한다.
2. 각 state에서 POS/NEG actuator pressure의 signed local rate capacity를 추정한다.
3. 보수적인 quantile과 upper cap으로 reference rate를 정하고, independent minimum-jerk waypoint를 pressure 공간에서 생성한다.
4. joint signed-rate support와 별도의 stateful PID replay로 pressure tracking, 방향성, chamber reserve 및 bounds를 검사한다.

현재 artifact의 실측 요약은 다음과 같다.

| 항목 | 현재 값 |
|---|---:|
| bank | 16 members × 2049 samples |
| physical duration/member | 40.96 s = 4 × 512 carry |
| actuator POS 전체 범위 | 101.325–145.0 kPa |
| actuator NEG 전체 범위 | 60.002–101.325 kPa |
| startup 포함 max sampled rate | POS 11.99, NEG 23.63 kPa/s |
| 정상 reference rate cap | POS +10/−12, NEG +4.5/−5 kPa/s |
| joint support fraction | 16개 모두 1.0 |
| PID POS RMSE 범위 | 2.79–3.28 kPa |
| PID NEG RMSE 범위 | 1.97–2.76 kPa |
| 5 kPa reserve fraction | POS/NEG 모두 전 member 1.0 |
| event direction agreement | POS 모두 1.0, NEG 최소 0.9167 |
| training NPZ의 oracle action | 없음 |

이 결과로 현재 lib3 reference는 다음처럼 부르는 것이 가장 정확하다.

> **state-conditioned capacity-screened and closed-loop-certified reference bank**

또는 한국어로는 `상태 조건부 capacity와 독립 폐루프 검증을 통과한 경험적 feasible reference bank`다.

강점은 다음과 같다.

- pressure를 직접 부드럽게 설계하므로 특정 oracle command의 고주파 흔들림을 target으로 복사하지 않는다.
- POS와 NEG를 독립적으로 변화시켜 두 축의 다양한 조합을 만든다.
- local pressure-rate, simultaneous signed-rate support, chamber reserve와 controller replay를 모두 사용한다.
- PID/action은 training reference나 policy observation에 들어가지 않는다.

하지만 아직 수학적 reachability certificate는 아니다.

- local rate sweep은 one-step 또는 짧은 구간의 capacity 근사이며 긴 trajectory 전체의 exact reachable set이 아니다.
- nearest operating point의 joint support는 연속 state 전 구간을 보장하지 않는다.
- PID 통과는 충분조건에 가까운 경험적 certificate다. PID가 실패한 reference도 다른 controller로는 feasible할 수 있다.
- capacity와 certification이 nominal simulator 하나에 의존하므로 real plant와 parameter uncertainty에 대한 robust feasibility는 아직 없다.
- 초기 대기압 transient의 NEG rate가 정상 구간 cap보다 크다. 따라서 초기 상태가 달라지는 실제 배포에서는 별도 startup profile이 필요하다.

결론적으로 현재 lib3 reference는 **legacy random보다 연구적으로 훨씬 방어력이 있지만, reference 방법 자체를 메인 novelty로 내기에는 formalization과 real validation이 부족하다.**

### 17.4 현재 lib4 oracle reference는 무엇을 증명하고 무엇을 못 증명하는가

lib4에는 다음 reference 계보가 이미 있다.

| profile | pressure/displacement span | 의미 |
|---|---|---|
| R1 independent-smooth | POS 8.95 kPa, NEG 14.07 kPa, displacement 6.69 mm | low-stroke smoke/regression |
| R2 topology-pulse | POS 32.95 kPa, NEG 32.58 kPa, displacement 16.21 mm | one-member high-stroke smoke |
| R3 train core | 16 members, displacement 0–20.99 mm | data-supported 공통 학습 source |
| R3-ID unseen core | 8 members, displacement 0–20.61 mm | 별도 in-distribution 검증 source |
| R4 nominal edge | 4 members, displacement 0–27.03 mm | 수집 범위 밖 외삽 stress; 학습 금지 |

R2–R4는 full six-valve prefix를 tuned simulator에 replay해 얻은 plant output이며, R3 train, R3-ID와 R4 artifact는 모두 독립 replay validator를 통과했다. 따라서 **같은 nominal simulator와 정확히 같은 초기 상태에서는 action witness가 존재한다**는 점은 lib3보다 강하게 증명한다. 저장 action은 audit용이고 policy에는 제공되지 않는다.

R2의 `bank_size=1` 과적합 위험 때문에 실제 screening source는 R3 16-member bank로 교체되었다. 다만 reference 분포가 여전히 특정 command proposal과 nominal model에 의해 결정되므로 다음 문제는 남는다.

- command proposal의 topology와 시간 패턴에 reference 다양성이 종속됨
- same-simulator exact replay는 real feasibility를 증명하지 않음
- initial state가 달라지면 같은 output trajectory의 action witness가 더 이상 유효하지 않을 수 있음
- 시작 1 s를 제외한 rate validation만 통과하므로 startup transient를 일반화할 수 없음
- pressure와 displacement가 plant output으로 묶여 있어 사용자가 원하는 dynamic oscillator형 displacement trajectory coverage를 직접 제어하기 어려움
- R3/R3-ID의 20.99/20.61 mm는 수집 데이터 최대 contraction 21.47 mm 안이지만, R4의 21.47–27.03 mm는 nominal stroke 32 mm 안에서도 data-unsupported extrapolation임

따라서 R3는 **현재 첫 RL screening을 시작할 수 있는 nominal-feasible training bank**이지, real feasibility나 state-dependent authority를 입증한 reference는 아니다. R4는 robust parameter sweep과 실제 저속 확인 전까지 OOD/edge stress로만 사용한다.

## 18. Feasible-reference 설계를 독립 연구 기여로 만드는 방법

### 18.1 추천 방법: robust capacity-conditioned reference curriculum

현재 lib3의 장점과 lib4 oracle replay의 장점을 합치면 다음 방법을 제안할 수 있다.

1. state \(x\) 또는 측정 가능한 feature \(\phi(x)\)에서 action sweep/short rollout을 수행한다.
2. model parameter ensemble \(\theta\in\Theta\)에 대해 다음 local output-rate set을 추정한다.

\[
\mathcal C_K(x)
=
\left\{
\frac{y_{t+K}-y_t}{K\Delta t}
\;\middle|\;
u_{t:t+K-1}\in\mathcal U,
x_{t:t+K}\in\mathcal X
\right\}.
\]

3. 여러 model/real calibration uncertainty에서 공통으로 가능한 robust under-approximation \(\underline{\mathcal C}_K(x)\)를 만든다. 여기서 \(\underline{\mathcal C}_K\)는 4.4절의 estimated authority \(\widehat{\mathcal A}_K\)를 reference certification에 쓰기 위해 보수적으로 줄인 집합이다.
4. pressure–displacement waypoint를 생성한 뒤 \(\dot r_t\)가 이 집합 안에 들도록 transition duration 또는 target amplitude를 조절한다.
5. reference가 쉬운 한 영역에만 몰리지 않도록 pressure, rate, reserve, displacement, velocity feature의 coverage를 최대화한다.
6. 마지막에 generator와 독립된 controller 또는 trajectory optimizer로 full-prefix certificate를 수행한다.

현재의 global rate cap보다 중요한 개선은 **같은 \(10\;\mathrm{kPa/s}\)라도 chamber reserve, displacement, valve hidden state에 따라 가능한지 다르게 판정하는 것**이다.

reference difficulty는 임의의 `easy/medium/hard`가 아니라 capacity utilization로 정의한다.

\[
\eta_t
= 
\inf\{\eta\ge0:\dot r_t\in\eta\,\underline{\mathcal C}_K(x_t)\}.
\]

- \(\eta\le0.4\): easy curriculum
- \(0.4<\eta\le0.7\): nominal training
- \(0.7<\eta\le0.9\): edge-of-capacity test
- \(\eta>1\): nominally unsupported/OOD stress

숫자 경계는 실제 sweep 통계로 다시 정해야 하지만, 이런 정의를 쓰면 `random이 더 어렵다`, `새 reference가 더 어렵다`는 주관적 논쟁을 정량화할 수 있다.

동적으로 admissible한 trajectory를 black-box dynamics의 reachable-set under-approximation에 projection하는 최근 연구도 같은 문제의식을 갖는다. DDAT는 predicted successor를 sampled reachable polytope에 순차 projection해 dynamic admissibility를 높였다 ([Bouvier et al., RSS 2025](https://www.roboticsproceedings.org/rss21/p078.html)). 현재 연구는 diffusion trajectory가 아니라 pneumatic tracking reference를 생성하고, pressure reserve와 displacement dynamics를 명시적으로 사용한다는 차이가 있다.

### 18.2 reference 다양성과 feasibility를 동시에 평가하는 지표

reference 논문으로 발전시키려면 `통과/실패`만으로는 부족하다. 다음 지표를 같이 보고해야 한다.

#### Feasibility

- state-conditioned support fraction
- robust capacity utilization의 median/95th percentile/max
- independent oracle/optimizer의 tracking residual
- pressure, reserve, displacement, velocity bound violation
- action saturation 및 inlet–outlet conflict fraction
- initial-state perturbation과 parameter ensemble에서 certificate pass rate

#### Diversity and coverage

- pressure–rate phase-plane occupied-bin ratio
- \((P_{act,+},P_{act,-},L,\dot L)\) coverage
- transition amplitude, hold time, frequency 분포
- POS/NEG/displacement change-direction 조합 수
- nearest-neighbor distance와 bank member 간 중복률

#### Target OOD

reference feature를

\[
\psi_t=[r_t,\dot r_t,\ddot r_t,\text{reserve guide},L_t,\dot L_t]
\]

로 정의하고 다음을 보고한다.

- train-bank에 대한 test reference의 nearest-neighbor 또는 Mahalanobis distance
- train convex hull 또는 learned density의 outside-support fraction
- in-support, edge-support, unsupported stress의 성능 곡선
- reference OOD distance와 tracking error/action saturation의 상관관계

이 지표가 있어야 `feasible reference가 OOD를 줄였다`가 시각적 인상에서 검증 가능한 주장으로 바뀐다.

### 18.3 필요한 reference ablation

같은 RL architecture와 같은 budget으로 아래 네 source를 비교하는 것이 가장 해석하기 좋다.

| ID | reference | 분리되는 효과 |
|---|---|---|
| G0 | legacy independent RandomRef | pressure range/frequency만 정한 기존 방식 |
| G1 | global rate-limited random | discontinuity와 global rate 제한 효과 |
| G2 | state-conditioned capacity reference | local control authority 반영 효과 |
| G3 | exact oracle-output bank | nominal simulator에서 action witness가 있는 상한/teacher-free source |

모든 policy는 training 중 oracle action을 보지 않는다. 같은 controller를 G0–G3에서 학습하고, 각 policy를 공통 test suite에서 교차 검증해야 한다.

```text
ID test: 같은 source의 unseen seed
interpolation: train pressure/rate envelope 내부의 새 조합
edge: capacity utilization 0.7–0.9
stress/OOD: legacy jump 또는 utilization >1
real: 동일한 bounded displacement reference
```

여기서 G2가 G3보다 일반화가 좋다면 `oracle command imitation`이 아니라 **capacity-based target distribution 설계가 효과적**이었다고 말할 수 있다. 반대로 G3만 잘되면 reference가 proposal-controller trajectory를 암기하기 쉽게 만든 것일 수 있다.

trajectory curriculum이 nonlinear tracking 학습을 안정화한다는 인접 근거도 있다. spherical-pendulum tracking 연구는 trajectory distribution을 policy capability에 따라 확장해 baseline보다 빠르고 안정적인 학습 및 real transfer를 보였다 ([Klink et al., 2023 preprint](https://arxiv.org/abs/2309.14096)). 다만 현재 연구는 performance-only curriculum보다 plant capacity와 pneumatic constraints를 먼저 사용한다는 차별점이 필요하다.

### 18.4 이것만으로 RA-L이 가능한가

현재 코드 그대로의 feasible reference만으로는 RA-L 메인 contribution이 되기 어렵다. 이유는 다음과 같다.

- lib3는 system-specific heuristic와 PID certifier에 의존한다.
- lib4는 one-member oracle bank라 일반 reference generator로 보기 어렵다.
- real system에서 feasibility/OOD 감소가 아직 검증되지 않았다.
- reference를 feasible하게 만들면 당연히 tracking score가 좋아지는 trivial 결과로 보일 수 있다.

반면 다음 세 가지가 추가되면 독립 contribution 후보가 된다.

1. state-conditioned robust capacity set과 difficulty 지표 \(\eta\)를 명확히 정의한다. reserve 기호 \(\rho\)와 혼용하지 않는다.
2. G0–G3 ablation으로 feasibility, diversity, 학습 속도, cross-reference generalization을 분리한다.
3. lib3 pressure tracking과 lib4 displacement tracking, 그리고 실제 장치에서 같은 원리가 재현됨을 보인다.

가장 현실적인 선택은 feasible reference를 논문의 유일한 novelty로 세우는 것이 아니라, **temporal physics-aware controller를 정당하게 학습하고 평가하기 위한 두 번째 방법론적 contribution**으로 넣는 것이다.

## 19. Under-1 mm displacement 결과는 얼마나 강한가

### 19.1 숫자 하나만으로는 novelty가 되지 않는다

실제 장치에서 displacement RMSE가 1 mm 미만이면 좋은 결과지만, 그 숫자만으로 RA-L novelty가 자동으로 생기지는 않는다. 최근 soft pneumatic actuator position control에서도 10 mm line 및 4 mm circle을 약 0.8 mm/s로 추종하며 오차 진폭 0.3 mm 이하를 보고한 사례가 있다 ([Liu et al., Mechanical Sciences 2024](https://ms.copernicus.org/articles/15/407/2024/)). 또한 dual-segment pneumatic soft manipulator가 point-to-point/load 조건에서 1 mm, 약 0.27% 위치 정확도를 보고했다 ([Sensors and Actuators A 2022](https://doi.org/10.1016/j.sna.2022.113644)).

따라서 `RMSE <1 mm`는 기여 자체라기보다 **현재 방법이 실제로 유효하다는 강한 결과 조건**이다. 비교할 때 다음이 같아야 한다.

- full stroke와 normalized error
- trajectory speed/frequency와 acceleration
- static point regulation인지 dynamic tracking인지
- payload와 leakage 변화
- sensor resolution 및 filtering delay
- trial 수와 worst-case/max error

예를 들어 16.2 mm displacement span에서 1 mm RMSE는 span-normalized RMSE가 약 6.2%다. 절대 오차는 작지만 다른 논문의 0.27% positioning claim보다 상대 정확도가 반드시 우수한 것은 아니다. 반대로 현재 시스템이 동시에 six-valve dual pressure, chamber reserve, leakage, dynamic displacement를 다루고 더 빠른 reference를 추종한다면 task 난이도 자체가 중요한 차별점이 된다.

### 19.2 논문 결과로 강하게 만드는 최소 조건

lib4 real 결과는 다음 기준으로 사전에 평가하는 것이 좋다.

1. **dynamic tracking**: 최소 3개 frequency band와 step/smooth-random/chirp를 포함한다.
2. **full-stroke reporting**: RMSE, MAE, max error와 함께 NRMSE를 보고한다.
3. **repeatability**: controller seed 3개, hardware trial 최소 10회 또는 통계적으로 충분한 반복을 사용한다.
4. **zero-shot와 adaptation 분리**: simulation-only policy, fixed-budget real fine-tuning, real-from-scratch 또는 classical baseline을 비교한다.
5. **robustness**: payload, leakage, initial pressure, reference amplitude/frequency 중 최소 2개를 바꾼다.
6. **safety/control quality**: chamber bounds, reserve, saturation, action spectrum과 vibration을 함께 보고한다.
7. **sensor floor 확인**: displacement sensor의 반복정밀도와 delay가 목표 RMSE보다 충분히 작아야 한다.

권장 목표는 단순 `best trial <1 mm`가 아니라 다음과 같다.

> **Unseen dynamic references over a practically meaningful stroke are tracked below 1 mm mean RMSE across repeated real trials, without pressure-bound violation, and a simulator-pretrained policy reaches that accuracy with substantially less real interaction than real-only learning.**

이 문장이 성립하면 결과가 method와 system novelty를 강하게 받쳐준다.

## 20. 최종 novelty 선택지와 추천

### 선택지 A — coupled system + temporal RL, feasible reference는 enabling method

가장 현실적이고 현재 자산과 잘 맞는 방향이다.

```text
정확도가 검증된 6-valve + soft-actuator coupled simulator
→ state/action history, pressure rate, reserve, K=2의 failure-informed controller
→ capacity-conditioned feasible reference로 학습 신호와 OOD stress를 분리
→ lib4 real dynamic displacement <1 mm 및 fixed-budget fine-tuning
```

장점은 현재 lib3 ablation을 그대로 활용하고, reference가 `쉬운 문제를 만들었다`는 반론도 G0–G3 cross-evaluation으로 방어할 수 있다는 점이다.

### 선택지 B — feasible/OOD-aware reference generation이 메인

가능하지만 추가 연구량이 크다. robust capacity set, difficulty metric, coverage optimization, 여러 initial state/model ensemble, lib3/lib4/real 검증이 모두 필요하다. 이 수준까지 만들면 다음과 같은 title이 가능하다.

> **Capacity-Conditioned Reference Curricula for Reinforcement-Learned Tracking of Delayed Pneumatic Soft Actuators**

현재 one-member lib4 oracle bank만으로 이 title을 사용하면 안 된다.

### 선택지 C — under-mm 결과 중심의 system paper

real 결과가 매우 강하면 가능하지만 위험하다. reviewer는 `왜 RL인가`, `PID/MPC보다 무엇이 나은가`, `다른 soft actuator의 sub-mm 결과와 무엇이 다른가`를 묻는다. under-mm 결과에 dynamic bandwidth, six-valve dual-pressure coupling, robustness와 adaptation efficiency가 결합되어야 한다.

### 최종 추천

현재는 A를 메인으로 하고 B를 두 번째 contribution으로 성장시키는 것이 가장 좋다.

> **본 연구는 state-dependent control authority를 무시한 reference가 delayed pneumatic RL의 학습과 평가를 왜곡함을 보이고, capacity-conditioned reference bank와 temporal physics-aware policy를 결합해 six-valve dual-pressure soft actuator의 real displacement tracking으로 확장한다.**

이 framing의 핵심은 `OOD를 줄였다`는 추상 문장이 아니다. 다음 인과관계를 실험으로 증명하는 것이다.

```text
unsupported target/rate
→ unavoidable error + saturation/chattering + biased replay visitation
→ capacity-conditioned reference가 avoidable tracking error를 분리
→ history/rate/reserve/K=2가 delayed control authority를 학습
→ unseen feasible/OOD references와 real plant에서 generalization 차이 확인
```

## 21. lib4에서 바로 적용할 연구 gate

현재 lib4는 simulator/reference plumbing, R3 train 16개/R3-ID 8개/R4 edge 4개의 독립 replay와 A0–A5 dry-run을 통과했다. seed-0 screening에서는 A2 rate, A3 reserve, A4 rate+reserve, A5 rate+reserve+K2가 1000 episode를 완료했고, A0 Direct는 870 episode에서 중단됐으며 A1 History-only는 시작되지 않았다. matched R3-ID 평가에서 A4는 현재 provisional winner지만 plateau 진동과 재현성 검증이 남았고, A5 K=2는 displacement RMSE를 낮추면서 actual-step action variation과 reserve violation을 키웠다. 상세 수치와 실행 순서는 [lib4 결과 및 후속 계획](ablation/lib4/results.md)을 따른다. 아직 real authority 검증은 수행하지 않았다.

### Gate L4-0 — simulator provenance와 initial-state 조건 고정

- tuned parameter bundle, build hash, initial pressure/displacement를 저장한다.
- pressure 4개, displacement, velocity의 real held-out rollout을 검증한다.
- reference feasibility가 어느 initial-state neighborhood에서 유지되는지 검사한다.

### Gate L4-1 — reference bank 확장

- 완료: R3 train 16개, 별도 R3-ID 8개, R4 edge 4개를 서로 다른 seed와 file hash로 분리했다.
- 완료: data-supported core는 0–20.99 mm, nominal edge는 27.03 mm까지 확장했다.
- 남음: 선행논문형 dynamic oscillator displacement target을 원본 그대로 두고 authority utilization으로 train/ID/edge를 분류하는 generator를 만든다.
- 남음: parameter/initial-state ensemble과 저속 real pulse로 R4 외삽 영역을 검증하기 전에는 R4를 학습에 사용하지 않는다.

### Gate L4-2 — 최소 controller ablation

처음부터 lib3의 모든 실패를 반복하지 않는다.

| ID | 구조 | 목적 |
|---:|---|---|
| L0 | Direct SAC | 선행 direct structure의 scaling failure |
| L1 | History SAC | hidden valve/delay reconstruction |
| L2 | L1 + pressure/displacement rate | net-flow와 motion direction 정보 |
| L3 | L1 + raw chamber reserve | pressure-gradient feature의 효과 |
| L4 | L1 + rate + reserve | physical observation 결합 |
| L5 | L4 + estimated authority/utilization | raw feature 대비 authority 추정의 추가 효과; estimator 검증 후 실행 |
| L6 | L4 또는 L5 + K=2 | short temporal action structure |
| L7 | best + integral error | steady-state displacement bias가 남을 때만 추가 |

CAPS와 MPOBS는 L0–L6의 원인을 분리한 뒤 best 구조에 하나씩 붙인다. lib3에서 자동으로 좋아지지 않았기 때문에 기본 구성으로 숨기지 않는다. L5는 `reserve observation`의 이름만 바꾼 모델이 아니라 bounded action sweep에서 $\widehat{\mathcal A}_K$ 또는 signed rate bound를 예측하고 policy/reference에 명시적으로 사용하는 모델이어야 한다.

### Gate L4-3 — reference/OOD cross-evaluation

각 controller를 다음 세 영역에서 평가한다.

- matched unseen feasible bank
- capacity edge bank
- legacy/random unsupported stress bank

이때 stress bank RMSE를 matched RMSE와 합산하지 않는다. stress test에서는 error보다 saturation, safety, recovery, action vibration과 graceful degradation을 본다.

### Gate L4-4 — real2sim2real claim

real 평가군은 최소 다음 네 개다.

```text
classical PID/model-based baseline
simulation-only temporal RL
simulation pretraining + fixed-budget real fine-tuning
real-only learning 또는 동일 budget의 가능한 대조군
```

최종 논문에서 가장 강한 결과는 최저 RMSE 한 점이 아니라, `같은 real interaction budget에서 feasible-reference + temporal policy가 더 빠르게 under-mm에 도달한다`는 adaptation curve다.
