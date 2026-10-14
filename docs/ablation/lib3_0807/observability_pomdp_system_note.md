# 우리 6-input, 4-pressure system에서 Control Observability와 RL POMDP의 분리

> 이 문서의 개념 정리를 실제 lib3 source와 numerical output-map rank로 검증한
> 결과는 [lib3 nonlinear observability 분석](lib3_nonlinear_observability_analysis.md)에
> 정리했다.

## 먼저 용어를 고정한다

우리 pressure-control plant를 다음처럼 쓴다.

\[
u_t\in\mathbb R^6,\qquad
p_t=
[P_{\rm ch+},P_{\rm ch-},P_{\rm act+},P_{\rm act-}]^\top
\in\mathbb R^4.
\]

- \(u_t\): chamber valve 2개와 actuator-side valve 4개의 command
- \(p_t\): controller가 이용하는 4개 pressure measurement
- 현재 lib3 RL task의 tracking reference는
  \(r_t=[r_{\rm act+},r_{\rm act-}]^\top\in\mathbb R^2\)이다. 즉 4개
  pressure를 측정하지만, 현재 reward에서 직접 tracking하는 target은 actuator
  pressure 2개이고 chamber pressure 2개는 reserve/constraint state다.

Real 장치에서는 pressure 외에 유량도 측정할 수 있다. 측정 가능한 flow channel은
정확히 6개이며, 위치는 다음과 같다.

\[
f_t^{\rm meas}=\begin{bmatrix}
f_{\rm pump\to ch,+} & f_{\rm pump\to ch,-} &
f_{\rm ch\to act,+} & f_{\rm ch\to act,-} &
f_{\rm act\to atm,+} & f_{\rm act\to atm,-}
\end{bmatrix}^{\!\top}\in\mathbb R^6 .
\]

따라서 real sensor output은 pressure만 쓸 때 \(y_t^p=p_t\)이고, flow를
함께 쓸 때 \(y_t^{p,f}=[p_t^\top,(f_t^{\rm meas})^\top]^\top\)이다.
유량은 state가 아니라 plant가 내보내는 **physical output**이다. 반면 simulator
내부의 `q`, `q_dot`은 flow-lag를 적분하기 위한 latent state이며, 두 값을 real
flow sensor 값과 동일시하면 안 된다.

lib3 C++의 내부 `mass_flowrate` 배열은 simulator 진단용으로 10개의 항을
보관한다. 이는 real에서 측정하는 위 6개 line flow와 일대일로 자동 대응한다고
가정할 수 없다. RL observation에 flow를 넣을 때는 실제 배관 위치에 맞는 6개
채널을 명시적으로 선택하고, 방향(sign), 단위, sensor delay, filtering을
calibration해야 한다.

따라서 발표에서는 “6개 control, 4개 pressure state/measurement”라고 쓰고,
현재 task가 2개 actuator-pressure target을 추종한다는 점을 별도로 적는
표현이 가장 정확하다. 만약 real experiment에서 네 pressure 모두에 reference를
붙이는 별도 task가 있다면, 그때만 \(r_t\in\mathbb R^4\)로 바꾸면 된다.

---

# A. Control system 관점: 교수님이 말한 Observability

## A-1. 제어용 reduced pressure model

압력 제어 문제를 다음 reduced model로 잡자.

\[
x_t^{\rm ctrl}=p_t\in\mathbb R^4,
\qquad
x_{t+1}^{\rm ctrl}=F_{\rm ctrl}(x_t^{\rm ctrl},u_t),
\qquad
y_t=p_t.
\]

이 정의에서는

\[
y_t=x_t^{\rm ctrl},
\qquad
C=I_4.
\]

따라서 local linearization
\[
\delta x_{t+1}=A\delta x_t+B\delta u_t,\qquad
\delta y_t=C\delta x_t
\]
에 대해

\[
\mathcal O=
\begin{bmatrix}
C\\CA\\\vdots\\CA^3
\end{bmatrix}
=
\begin{bmatrix}
I_4\\A\\\vdots\\CA^3
\end{bmatrix}
\]

는 첫 block \(I_4\)만으로 rank 4이다. **이 reduced control model에서는
state observer가 필요 없고, system은 fully observable이다.**

여기서 input이 6개이고 state/output이 4개인 것은 observability와 충돌하지
않는다. \(B\in\mathbb R^{4\times6}\)인 over-actuated control-allocation
문제일 뿐이며, observability는 기본적으로 \(A,C\)와 sensor choice의
문제다.

이것이 교수님이 “우리 system은 full observed가 아닌가?”라고 하신 해석에
가장 가깝다.

## A-2. 이 결론이 성립하려면 필요한 가정

\(x^{\rm ctrl}=p\)가 올바른 control state가 되려면, 실무적으로

\[
p_{t+1}\approx F_{\rm ctrl}(p_t,u_t)
\]

가 충분히 잘 성립해야 한다. 즉 현재 pressure와 known command만으로
다음 pressure를 예측할 수 있어야 한다. valve lag, hysteresis, pump dynamics를
명시적으로 state로 넣지 않는다면 이들은 reduced model에서 unmodeled
dynamics/disturbance로 취급된다.

따라서 다음 두 표현을 구분한다.

- **맞는 표현:** “4-pressure reduced control model은 \(C=I_4\)이므로
  fully observed이다.”
- **아직 검증이 필요한 표현:** “4 pressure만으로 실제 plant의 next state가
  정확히 Markov하다.”

첫 문장은 state 정의에 대한 제어공학적 사실이다. 두 번째 문장은 model
closure/adequacy에 대한 실험적 질문이다.

---

# B. Simulator 관점: reduced model 밖에 무엇이 추가되는가

lib3 simulator는 4 pressure만 적분하지 않는다. RK integrator의 explicit
state vector는

\[
x_t^{\rm RK}=
[t,\;P_{\rm ch+},\;P_{\rm ch-},\;\theta,\;
P_{\rm pump1},\;P_{\rm pump2},\;P_{\rm act+},\;P_{\rm act-}]^\top
\in\mathbb R^8
\]

이다. measurement로 나오는 것은 이 중 4 pressure다.

\[
y_t=
[P_{\rm ch+},P_{\rm ch-},P_{\rm act+},P_{\rm act-}]^\top .
\]

또한 6개 valve에는 별도의 runtime dynamics가 있다. 각 valve \(i\)는 대략

\[
\xi_{i,t}=[z_{i,t},q_{i,t},\dot q_{i,t},
I_{{\rm prev},i,t},s_{{\rm prev},i,t}]
\]

를 갖는다.

- \(z_i\): hysteresis internal state
- \((q_i,\dot q_i)\): second-order flow-lag state
- \(I_{\rm prev},s_{\rm prev}\): 이전 current와 opening/closing branch state

따라서 simulator 전체 implementation state는 개념적으로

\[
x_t^{\rm sim}=[x_t^{\rm RK},\xi_{1,t},\ldots,\xi_{6,t}]
\]

이다. 이 augmented simulator state를 대상으로 하면 sensor equation은
\(C=I\)가 아니다.

이 점은 **simulator 내부 state를 모두 복원해야 real controller가 제대로
동작한다**는 뜻은 아니다. real valve에도 대응되는 물리 현상은 있지만,
reduced pressure controller에서는 이를 모델링하지 않고 disturbance/dynamics로
흡수할 수 있다. simulator는 그 현상을 더 명시적으로 구현했을 뿐이다.

정리하면 다음과 같다.

| 관점 | state | measurement | 결론 |
|---|---|---|---|
| Real/reduced pressure control | \(x^{\rm ctrl}=p\in\mathbb R^4\) | \(y=p\) | \(C=I_4\), fully observable |
| lib3 simulator internal model | \(x^{\rm sim}=[x^{\rm RK},\xi_1,\ldots,\xi_6]\) | 4 pressures | internal full state는 직접 측정되지 않음 |
| Simulator full-state analysis | augmented nonlinear state | 4 pressures + known input history | local observability rank를 별도로 계산해야 함 |

기본 lib3에는 0.1 s measurement delay도 있다. 따라서 policy가 보는 값은
엄밀히는 최근 plant state가 아니라 delayed pressure measurement다.

\[
y_t^{\rm meas}=p_{t-5}+v_t
\qquad (50\ {\rm Hz}).
\]

이 delay는 reduced pressure model의 \(C=I_4\) 자체를 부정하지 않는다.
대신 real controller를 구현할 때 delay compensation 또는 recent input
history가 유용해지는 이유다.

---

# C. RL 관점: POMDP가 무엇이고, 우리 system은 POMDP인가

## C-1. POMDP의 정의

MDP에서는 agent가 쓰는 state \(z_t\)가 미래를 결정하기에 충분하다.

\[
p(z_{t+1},r_t\mid z_t,a_t,\mathrm{past})
=
p(z_{t+1},r_t\mid z_t,a_t).
\]

POMDP에서는 environment 내부에는 Markov state \(s_t\)가 존재하지만, agent는
그 전체를 보지 못하고 observation \(o_t\)만 받는다.

\[
s_{t+1}\sim T(s_{t+1}\mid s_t,a_t),\qquad
o_t\sim Z(o_t\mid s_t).
\]

서로 다른 내부 state가 같은 observation을 만들 수 있고, 같은 action 뒤의
미래도 다르면 agent의 current observation은 Markov state가 아니다.

\[
s_t^{(1)}\neq s_t^{(2)},\quad
o_t^{(1)}=o_t^{(2)},\quad
p(s_{t+1}\mid s_t^{(1)},a_t)
\neq
p(s_{t+1}\mid s_t^{(2)},a_t).
\]

POMDP는 “system이 제어 불가능하다” 또는 “제어공학적으로 unobservable하다”는
말이 아니다. **agent에게 직접 주어진 observation이 full Markov state가
아니라는 RL의 정보 구조**를 뜻한다.

## C-2. 우리 simulator environment는 어떻게 POMDP가 되는가

simulator의 Markov environment state는 delay buffer까지 포함해

\[
s_t^{\rm env}=
[x_t^{\rm sim},\;b_t^{\rm delay},\;r_t]
\]

로 둘 수 있다. \(x_t^{\rm sim}\)에는 4 pressure, pump state, crank phase,
6개 valve state가 들어가고, \(b_t^{\rm delay}\)는 0.1 s sensor delay를
만드는 최근 pressure buffer다.

raw policy가 받는 것은 대략

\[
o_t^{\rm raw}=[p_{t-5}^{\rm meas},\;r_t]
\]

뿐이다. 같은 delayed 4 pressure가 보여도 pump pressure, valve hysteresis,
flow-lag, 최근 command가 다를 수 있다. 따라서 같은 새 command를 넣은 뒤의
pressure response가 달라질 수 있다.

\[
\boxed{\text{lib3 full simulator state를 기준으로, raw 4-pressure policy는 POMDP이다.}}
\]

이것은 simulator에 hidden state가 있으므로 자동으로 성립하는 단순한 선언이
아니다. valve state가 output에 즉시 나타나지 않고 measurement가 delayed되므로,
서로 다른 \(s_t^{\rm env}\)가 같은 \(o_t^{\rm raw}\)를 실제로 만들 수 있기
때문이다.

여기에 실제 6-channel flow를 추가하면 observation은

\[
o_t^{p,f}=[p_{t-d}^{\rm meas},f_{t-d_f}^{\rm meas},r_t]
\]

가 된다. 이 경우 flow가 valve/pump hidden state에 민감하면 같은 pressure를
만드는 내부 state를 더 잘 구분할 수 있으므로 POMDP의 ambiguity가 **완화**될
수 있다. 그러나 flow 역시 지연·noise가 있는 measurement이고, pressure와 flow
만으로 모든 simulator state를 항상 복원한다는 보장은 없다. 따라서
“flow를 추가하면 MDP가 된다”가 아니라, six-channel sensor map을 포함한
prediction/rank 검증으로 information sufficiency를 확인해야 한다.

## C-3. 현재 history-action SAC는 POMDP를 어떻게 다루는가

현재 policy input의 핵심은

\[
o_t^{\rm hist}=
[p_{t-10:t}^{\rm meas},\;u_{t-10:t-1},\;r_{\rm history/preview}].
\]

즉 11개의 pressure sample과 10개의 이전 action을 준다. 이는 0.1 s delay와
valve lag가 남긴 input-output response를 policy가 볼 수 있게 한다. history는
belief-state approximation 또는 finite-memory dynamic output feedback 역할을
한다.

다만 엄밀한 결론은 다음이다.

\[
\boxed{\text{현재 history-action SAC는 POMDP에 history를 제공하는 방법이지,}}
\]
\[
\boxed{\text{이미 exact MDP가 되었다고 증명된 것은 아니다.}}
\]

history \(h_t\)가 충분한 statistic이면

\[
b_t(s)=p(s_t=s\mid o_{0:t},a_{0:t-1})
\approx \psi(h_t)
\]

가 되고, \(\psi(h_t)\)는 approximate Markov information state가 된다. 이것이
history-only가 장기 학습에서 따라잡을 가능성이 있는 이론적 이유다.

## C-4. History, belief state, MDP의 정확한 관계

### 1) Raw pressure만 주는 경우

\[
o_t^{\rm raw}=p_{t-5}^{\rm meas}
\]

는 valve hysteresis, pump pressure, delay buffer를 생략한다. 같은 raw pressure가
보여도 hidden state가 다르면 다음 pressure가 달라질 수 있으므로, 가장 강한
partial-observation setting이다.

### 2) 고정 길이 pressure/action history를 주는 경우

\[
h_t^L=[o_{t-L+1:t},\;a_{t-L+1:t-1}]
\]

는 hidden state에 관한 정보를 더 많이 주지만, 일반적으로 POMDP를 없애지는
않는다. 두 trajectory가 최근 \(L\) step의 pressure/action history는 같지만
그 이전 input, valve hysteresis state, pump state가 다를 수 있기 때문이다.

\[
h_t^{L,(1)}=h_t^{L,(2)}
\quad\not\Rightarrow\quad
s_t^{(1)}=s_t^{(2)}.
\]

따라서 현재 history-action SAC는 **POMDP 위에서 동작하는 finite-memory
policy**다. partial observability를 완화하지만, exact MDP를 보장하지 않는다.

### 3) Full history로 계산한 belief state

알려진 transition/measurement model이 있다면 전체 history는 belief를 만든다.

\[
b_t(s)=p(s_t=s\mid o_{0:t},a_{0:t-1}).
\]

새 observation을 받은 뒤 belief는 Bayes filter로 갱신된다.

\[
b_{t+1}(s')
\propto
Z(o_{t+1}\mid s')
\int T(s'\mid s,a_t)b_t(s)\,ds.
\]

이 \(b_t\)는 과거 전체를 압축한 sufficient statistic이므로, belief를 state로
쓰면 이론적으로 belief-MDP가 된다. 그러나 현재 SAC는 \(T,Z\)를 알고 이
filter를 계산하는 방법이 아니라, 고정 길이 history를 neural network에 넣어
그 기능을 근사 학습하는 방법이다.

### 4) 유한 history가 진짜 MDP state가 되는 예외

어떤 길이 \(L\)에 대해

\[
s_t=\psi_L(h_t^L)
\]

가 정확히 성립하거나, 적어도 control-relevant future distribution이
\(h_t^L\)만으로 결정되면 유한 history 자체를 state로 쓸 수 있다.

\[
p(s_{t+1},r_t\mid h_t^L,a_t,\mathrm{past})
=p(s_{t+1},r_t\mid h_t^L,a_t).
\]

이 경우에만 “history를 주어 MDP가 되었다”고 말할 수 있다. 우리 system에서는
valve hysteresis, flow lag, pump dynamics, delay 때문에 이 성질을 가정하지 않고
prediction/reconstruction test로 확인해야 한다.

### 5) p_dot과 reserve의 위치

\[
\dot p_t\approx(p_t-p_{t-1})/\Delta t,
\qquad
\rho_t=[P_{\rm ch+}-P_{\rm act+},\;P_{\rm act-}-P_{\rm ch-}]^\top.
\]

이는 이미 주어진 history의 deterministic transform이다. 따라서 POMDP를
belief-MDP로 바꾸지는 못한다. 다만 hidden dynamics 추정에 필요한 rate와
pressure headroom을 network가 쉽게 읽게 하는 inductive bias가 될 수 있다.

## C-5. 우리 system에 대한 최종 판정

| 어떤 state/observation을 기준으로 말하는가 | 판정 |
|---|---|
| Control state를 4 pressure로 정의, \(y=p\) | fully observable reduced control model |
| Full simulator state를 environment state, raw 4 pressure만 policy input | **POMDP** |
| Full simulator state를 environment state, pressure/action history를 policy input | POMDP이지만 history로 approximate belief/Markov state를 구성하려는 setting |
| Reduced pressure model을 RL environment state로 채택 | \(p_{t+1}=F(p_t,u_t)\)가 충분히 성립할 때 MDP approximation |

따라서 발표에서 가장 정확한 문장은 다음이다.

> 우리 pressure controller의 reduced 4-state model은 fully observed이다.
> 그러나 valve dynamics와 delay를 포함한 simulator environment에서 current
> pressure observation만 받는 RL agent는 POMDP에 놓인다. 우리는 pressure-action
> history로 이 partial observability를 완화하는 finite-memory policy를 사용한다.

---

# D. p_dot과 reserve는 어느 위치인가

\[
\dot p_t\approx\frac{p_t-p_{t-1}}{\Delta t},
\qquad
\rho_t=
\begin{bmatrix}
P_{\rm ch+}-P_{\rm act+}\\
P_{\rm act-}-P_{\rm ch-}
\end{bmatrix}.
\]

history-action SAC에 pressure history가 이미 있다면 p_dot과 reserve는
새 sensor도, control observability를 새로 만드는 state도 아니다.

\[
\sigma(o_t^{\rm hist},\dot p_t,\rho_t)
=\sigma(o_t^{\rm hist}).
\]

이들의 역할은 RL representation이다.

- p_dot: pressure trajectory의 진행 방향/속도를 actor와 critic이 쉽게 읽게 함
- reserve: actuator authority를 결정하는 chamber headroom을 직접 좌표화함

따라서 주장할 수 있는 것은 “dynamics-aligned coordinate가 finite-budget
RL의 learning/convergence를 개선한다”이다. “p_dot이 system을 observable하게
만들었다” 또는 “reserve가 POMDP를 해결했다”는 주장은 맞지 않다.

---

# 교수님께 바로 말할 답변

> 제어공학적으로 pressure 4개를 control state로 두면 측정값이 곧 state라
> \(C=I_4\)이고, 이 reduced pressure model은 fully observable입니다.
> 6개의 valve input은 observability 문제가 아니라 control allocation
> 문제입니다.
>
> 다만 simulator는 pump pressure와 valve hysteresis/flow-lag state를
> 명시적으로 갖고, policy에는 0.1 s delayed pressure만 들어옵니다.
> 그래서 RL에서는 current pressure snapshot이 충분한 Markov information인지
> 별도 질문이 생깁니다. 현재 history-action SAC는 지난 pressure와 action을
> 이용해 delay와 hidden dynamics를 추론하는 finite-memory dynamic output
> feedback입니다.
>
> Real 장치에서 측정 가능한 flow는 pump→chamber 2개, chamber→actuator 2개,
> actuator→atmosphere 2개로 총 6개입니다. 이는 simulator의 latent `q`를
> 그대로 노출하는 것과 달리 deployable physical output을 추가하는 것이므로,
> pressure-only와 pressure+six-flow observation의 ambiguity와 prediction error를
> 별도로 비교하겠습니다.
>
> p_dot과 reserve는 새 information을 더하는 것이 아니라, 이 history를
> pressure dynamics에 맞는 좌표로 바꿔 finite-budget learning을 쉽게 하는
> inductive bias로 해석하겠습니다.

---

# 다음 검증: 두 축을 섞지 않는다

## Control-model adequacy 축

reduced pressure state가 실용적으로 closed state인지 검사한다.

\[
{\rm error}_{\rm next}
=
\left\|p_{t+1}-\widehat F_{\rm ctrl}(p_t,u_t)\right\|.
\]

같은 \(p_t,u_t\)에서 past/hidden condition에 따라 next pressure residual이
체계적으로 달라지면, reduced model에 unmodeled memory가 남아 있다는 뜻이다.

## RL information 축

다음 입력 간 held-out next-pressure prediction과 policy learning을 비교한다.

\[
p_t
\;\rightarrow\;
p_{t-L:t}
\;\rightarrow\;
[p_{t-L:t},u_{t-L:t-1}].
\]

마지막 조건의 prediction error가 유의하게 줄면 action history가 delay/hidden
dynamics를 추론하는 데 기여한다는 근거가 된다. 그 다음에만 p_dot/reserve를
추가해, 같은 information의 coordinate bias가 learning speed와 closed-loop
tracking을 얼마나 바꾸는지 분리해서 본다.
