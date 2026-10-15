# 교수님 논의용: pneumatic RL observation을 왜 이렇게 구성해야 하는가

작성일: 2026-08-30

> 이 문서의 LaTeX 수식이 문자 그대로 보이는 viewer에서는
> [plain 수식 요약](formula_reference_plain.md)을 사용하세요. Markdown 자체는
> LaTeX 수식을 렌더링하지 않으며 MathJax/KaTeX 확장이 필요합니다.

## 0. 이번 주에 가져갈 결론

가장 방어 가능한 결론은 다음이다.

> 이 시스템에서 **현재 pressure만으로는 Markov state를 구성할 수 없다.** 같은 현재 pressure와 valve command라도 관측되지 않는 valve dynamic state, pump phase/pressure, sensor delay state가 다르면 다음 pressure response가 달라지기 때문이다. 따라서 pressure/action history는 hidden state를 간접 추정하여 state aliasing을 줄이는 temporal information이다. 반면 explicit pressure rate와 chamber–actuator pressure difference는 history와 raw pressure에서 계산 가능한 값이므로 새로운 정보를 추가하지 않는다. 이들은 finite network가 pressure trend와 flow headroom을 쉽게 사용하도록 만드는 **physics-structured inductive bias**다. 특히 pressure rate는 short-horizon MPObs의 free-response 항과 유사하고 local derivative feedback을 쉽게 만들어 진동을 줄일 수 있지만, future action-conditioned nonlinear prediction을 포함하지 않으므로 MPObs와 수학적으로 동일하지 않다.

따라서 지금 주장하면 안 되는 문장은 다음이다.

- “rate는 시스템을 observable하게 만드는 필수불가결한 센서다.”
- “rate가 MPObs를 완전히 대체한다.”
- “RL return이 좋아졌으므로 observation의 물리적 필요성이 증명됐다.”

대신 다음처럼 주장한다.

- **History의 필요성:** current-only observation은 구조적으로 non-Markov이고, finite pressure/action history가 predictive state aliasing을 감소시킨다.
- **Rate의 역할:** rate는 새 정보가 아니라 realized pressure trend를 명시한 inductive bias이며, 제한된 데이터·network·optimizer에서 damping-relevant coordinate를 학습하기 쉽게 한다.
- **Reserve의 역할:** reserve도 새 정보가 아니라 local pressure headroom을 명시한 좌표다. compressible flow 전체를 결정하는 충분통계는 아니다.
- **MPObs와의 관계:** rate는 MPObs의 짧은 시간 free response 일부를 저비용·model-free하게 근사할 가능성이 있으며, 대체인지 보완인지는 factorial ablation으로 판정한다.

이 해석에는 직접적인 선행 근거가 있다. Char and Schneider의 NeurIPS 2023 연구는 partial observation history를 무조건 GRU/Transformer에 맡기기보다, PID의 합과 차분 연산을 history encoder의 inductive bias로 사용했다. 따라서 “history로 계산 가능한 derivative를 명시적으로 제공하는 것이 왜 의미가 있는가?”에 대한 가장 가까운 이론·실험적 precedent다. 단, 그 논문은 tracking-error derivative와 learned generalized difference encoder를 사용하므로 현재의 raw pressure rate와 완전히 같은 방법은 아니다.

---

## 1. 먼저 질문을 세 개로 분리해야 한다

“이 observation이 필요한가?”에는 서로 다른 세 질문이 섞여 있다.

| 질문 | 판정 대상 | 가능한 결론 |
|---|---|---|
| 정보가 필요한가? | current pressure vs pressure/action history | hidden dynamics를 구별하는 temporal information인가 |
| 표현이 도움이 되는가? | history vs history + rate/reserve | 같은 정보를 물리적으로 유리한 좌표로 바꿨는가 |
| 제어에 필요한가? | tracking, 진동, safety, sim-to-real | finite policy가 실제 closed loop에서 그 좌표를 유용하게 쓰는가 |

history는 첫 질문에 해당하지만, rate와 reserve는 주로 두 번째 질문에 해당한다. 이를 한꺼번에 “observation이 더 많아져서 성능이 좋아졌다”로 설명하면 novelty도 약하고 반론도 쉽다.

---

## 2. 무엇이 pressure를 결정하는가

### 2.1 최소 물리식

isothermal pneumatic volume의 pressure dynamics는 다음 형태다.

\[
\dot P
=
-\frac{P}{V}\dot V
+\frac{RT}{V}\dot m.
\]

따라서 다음 pressure는 현재 pressure 하나가 아니라 적어도 다음에 의해 결정된다.

\[
P_{t+1}
=F\!\left(
P_t,V_t,\dot V_t,
\dot m_t,
\Delta t
\right).
\]

여기서 mass flow는 단순히 valve command와 pressure difference만의 함수가 아니다.

\[
\dot m_t
=
g\!\left(
P_{u,t},P_{d,t},
A_{\mathrm{eff}}(z_t,u_t),
x_{1,t},x_{2,t}
\right).
\]

compressible orifice의 대표적인 형태는

\[
\dot m
=C_d A_{\mathrm{eff}}
\frac{P_u}{\sqrt{RT}}
\Phi\!\left(\frac{P_d}{P_u}\right)
\]

이므로 absolute upstream pressure, pressure ratio, choking 여부, effective area가 함께 중요하다. 따라서 \(P_{ch}-P_{act}\)는 유용한 headroom proxy이지만 mass flow 그 자체는 아니다.

### 2.2 현재 simulator에서 policy가 못 보는 상태

코드상 full simulator state에는 다음이 포함된다.

- chamber positive/negative pressure
- pump piston pressure와 crank phase
- actuator positive/negative pressure
- 각 6개 valve의 hysteresis state z
- 각 valve의 dynamic flow state x1, x2
- previous current와 switching state
- observation delay buffer
- lib4에서는 actuator displacement, velocity와 이에 따른 variable volume

반면 lib3 policy가 받는 직접 측정은 기본적으로 네 pressure다. 즉 실제 상태와 측정의 관계는

\[
x_{t+1}=f(x_t,u_t),\qquad y_t=h(x_t)
\]

이고, \(h\)는 많은 hidden state를 버리는 many-to-one mapping이다.

현재 구현 근거:

- [simulator state와 pressure output](../pneu_env/src/pneu_env/lib3/pneumatic_simulator.cpp)
- [valve runtime state 정의](../pneu_env/src/pneu_env/lib3/pneumatic_simulator.h)
- [pressure·flow·dynamic valve 식](../pneu_env/src/pneu_env/lib3/pneumatic_CT.cpp)
- [observation delay buffer](../pneu_env/src/pneu_env/sim3.py)

### 2.3 current pressure가 Markov state가 아님을 보이는 counterexample

Markov observation이라면 같은 observation에서 같은 action을 주었을 때 next-observation distribution도 같아야 한다.

\[
y(x)=y(x')
\Longrightarrow
p(y_{t+1}\mid x,u)=p(y_{t+1}\mid x',u)
\quad \forall u.
\]

하지만 simulator에서 두 상태 \(x,x'\)가 같은 네 pressure를 갖더라도 valve flow state가 \(x_1\neq x'_1\)일 수 있다. 구현상

\[
q_{pred}=\max(x_1,0),\qquad
\dot m\propto q_{pred},\qquad
\dot P\propto \dot m
\]

이므로 일반적으로

\[
P_t=P'_t,\quad u_t=u'_t,\quad x_{1,t}\neq x'_{1,t}
\Longrightarrow
P_{t+1}\neq P'_{t+1}.
\]

이것은 **현재 simulator에 대한 구조적 non-Markov counterexample**다. 실제 장치에 대해서는 모델 식만으로 증명했다고 하지 않고, 같은 current pressure를 맞춘 matched-pulse 실험으로 검증해야 한다.

---

## 3. 그러면 history가 “필수”라는 것은 어떻게 보이는가

finite lifted history를

\[
\xi_t=
[y_{t-H:t},u_{t-H:t-1}]
\]

로 정의한다. 이상적으로는 \(\xi_t\)가 full history의 **information state**여야 한다.

현재 lib3의 H10 설정은 50 Hz에서 과거 10개와 현재 1개의 pressure sample, 그리고 과거 10개의 action을 사용한다. 즉 pressure window는 0.20 s이고, 약 0.10 s인 sensor delay보다 길다. 이 길이는 “10이 최적”이라는 뜻이 아니라 최소한 delay pipeline에 들어 있는 command와 그 뒤에 나타난 pressure response를 동시에 볼 수 있다는 뜻이다.

\[
p(y_{t+1},r_t\mid h_t,u_t)
=
p(y_{t+1},r_t\mid \xi_t,u_t).
\]

하지만 nonlinear system에서 특정 finite \(H\)가 정확히 sufficient하다고 증명하려면 observability 조건, noise model과 정확한 plant가 필요하다. RL return만으로는 이를 증명할 수 없다. 이 프로젝트에서는 “exact Markov state”보다 **approximate information state**라는 표현이 정확하다.

검증은 다음 세 가지를 연결한다.

1. **구조적 반례:** 같은 current pressure라도 hidden valve state가 다르면 next pressure가 달라짐.
2. **matched intervention:** current pressure를 맞추고 같은 valve pulse를 주었는데 recent history가 다른 경우 response가 달라짐.
3. **predictive sufficiency:** future action을 동일하게 condition했을 때 history를 넣으면 held-out future-pressure error와 conditional variance가 감소함.

현재 확보한 근거는 이미 이 방향과 맞는다.

- current pressure와 reserve를 약 1 kPa 이내로 맞춘 matched pair에서 같은 pulse의 0.5 s 효과가 POS 4.28 kPa, NEG 6.59 kPa 차이 났다.
- 차이가 0.1 s 이후 나타났으며 simulator sensor delay와 일치했다.
- future pressure predictor에서 current-only보다 pressure history가, pressure history보다 pressure+action history가 더 정확했다.

held-out combined pressure RMSE:

| predictor input | 0.02 s | 0.10 s | 0.20 s | 0.50 s |
|---|---:|---:|---:|---:|
| current pressure | 0.335 | 1.360 | 1.962 | 1.641 |
| pressure history | 0.139 | 0.896 | 1.611 | 1.548 |
| pressure + action history | 0.071 | 0.398 | 0.744 | 1.213 |

이 결과는 “current-only로는 미래가 alias되고, 과거 action까지 있어야 delayed input effect를 구별하기 쉽다”는 근거다. 다만 이것을 strict observability proof라고 부르지는 않는다.

직접적인 선행 사례로 Xia et al.은 부분 pressure measurement를 쓰는 유동 제어에서 과거 measurement와 action을 NARX 형태로 구성했다. measurement history만 사용했을 때보다 past action을 포함했을 때 drag reduction이 67.45%에서 97.00%로 증가했다. pneumatic system과 물리는 다르지만, **partial measurement + delayed actuation에서 measurement/action history가 필요하다**는 논리는 직접 대응된다.

---

## 4. history로 계산 가능한 \(\dot P\)를 왜 explicit하게 넣는가

### 4.1 먼저 수학적으로 인정해야 할 사실

현재 History SAC에서

\[
\dot P_t=\frac{P_t-P_{t-1}}{\Delta t}
\]

이고 reserve도

\[
\rho_{+,t}=P_{ch,+,t}-P_{act,+,t},\qquad
\rho_{-,t}=P_{act,-,t}-P_{ch,-,t}
\]

이므로 둘 다 history/raw pressure의 deterministic function이다. \(D_t=g(H_t)\)라 쓰면

\[
\sigma(H_t,D_t)=\sigma(H_t),
\qquad
I(x_t;H_t,D_t)=I(x_t;H_t).
\]

즉 **rate와 reserve는 information-theoretic observability를 증가시키지 않는다.** 무한 데이터와 충분한 universal approximator를 가정하면 expressivity 면에서도 필수적이지 않다. 이 점은 오히려 명확하게 인정해야 논리가 강해진다.

### 4.2 그래도 finite SAC에서 도움이 될 수 있는 이유

실제 policy는 무한한 함수가 아니라 고정 폭 MLP, 유한 replay, normalization, optimizer와 regularization을 가진다. 원하는 local control law에 derivative feedback이 있다고 하자.

\[
u_t=k_p(r_t-P_t)-k_d\dot P_t.
\]

raw history만 쓰면 network가 강하게 correlated된 두 absolute pressure를 빼고 \(1/\Delta t\)를 곱하는 방향을 먼저 학습해야 한다.

\[
-k_d\dot P_t
=-\frac{k_d}{\Delta t}P_t
+\frac{k_d}{\Delta t}P_{t-1}.
\]

explicit rate를 주면 같은 물리 방향이 하나의 coordinate가 된다.

\[
-k_d\dot P_t=w_d D_t.
\]

따라서 fixed parameter norm, weight decay 또는 limited optimization 아래에서는 두 표현이 같은 함수 집합을 실질적으로 탐색하지 않는다. rate coordinate는 “pressure level”과 “진행 방향”을 분리하고 derivative-like feedback을 낮은 복잡도로 표현하게 한다. 이것이 여기서 말하는 **inductive bias**다.

PINN처럼 governing equation residual을 loss에 강제하는 것은 아니므로, 더 정확한 명칭은 다음 중 하나다.

- physics-derived feature
- physics-structured observation
- coordinate-level inductive bias

### 4.3 scale confound는 반드시 제거해야 한다

현재 pressure history는 \(P/100\)으로 들어가지만 explicit rate는

\[
\frac{\dot P}{50}=P_t-P_{t-1}
\]

로 들어간다. network가 normalized pressure history에서 직접 만든 차분은

\[
\frac{P_t-P_{t-1}}{100}
\]

이므로 기존 explicit rate에는 약 100배의 numerical emphasis가 있다. 따라서 History와 History+rate/50만 비교하면 “물리 좌표 효과”와 “입력 scaling 효과”가 섞인다.

필수 비교군은 다음 네 개다.

| 이름 | 입력 | 분리하는 효과 |
|---|---|---|
| H | history | 기준 |
| R50 | H + rate/50 | 현재 구현의 coordinate + scale 효과 |
| R5000 | H + rate/5000 | history 차분과 scale을 맞춘 coordinate 효과 |
| Z4 | H + zero 4차원 | input dimension/fan-in placebo |

현재 seed-0 결과에서 R50은 H보다 tracking과 진동 지표가 좋았지만, R5000은 chamber overcharge를 이용한 unsafe policy가 되었고 Z4도 성능을 변화시켰다. 따라서 아직 “rate가 최종 성능을 인과적으로 개선했다”고 확정할 수 없다. 동일 chamber constraint를 training에 적용한 multi-seed 재실험이 필요하다.

### 4.4 NeurIPS 2023의 PID-inspired inductive bias와의 연결

Char and Schneider의 “PID-Inspired Inductive Biases for Deep Reinforcement Learning in Partially Observable Control Tasks”는 현재 논리를 직접 뒷받침하는 선행 연구다. 이 연구의 출발점도 다음과 같다.

- true state가 부분 관측되므로 policy가 observation history를 사용해야 한다.
- GRU나 Transformer처럼 매우 유연한 encoder는 simulator의 우연한 temporal pattern에 과적합할 수 있다.
- 많은 control task에서는 PID가 사용하는 합과 차분만으로도 control-relevant history를 요약할 수 있다.
- 따라서 history encoder에 summing/differencing structure를 넣으면 useful information과 robustness 사이의 inductive bias를 만들 수 있다.

#### PIDE: tracking error의 P/I/D를 직접 출력

tracking error를

\[
e_t=x_t-r_t
\]

로 두면 논문의 PID Encoder(PIDE)는 대략 다음 3개를 history encoding으로 출력한다.

\[
z_t^{PIDE}
=
\left[
e_t,\;
\sum_{i=1}^{t} e_i\Delta t,\;
\frac{e_t-e_{t-1}}{\Delta t}
\right].
\]

즉 derivative 항은

\[
\dot e_t
=
\dot x_t-\dot r_t
\]

에 해당한다. 이 값 역시 raw signal/reference history에서 계산 가능하므로 새로운 information은 아니다. 저자들의 주장은 P/I/D summary가 arbitrary sequence encoder보다 control에 적합한 history representation을 우선하도록 만든다는 것이다.

#### GPIDE: 임의 observation의 차분을 사용하는 generalized encoder

PIDE가 tracking problem에 한정된다는 점을 보완하기 위해 논문은 GPIDE도 제시한다. 각 시점에서 대략

\[
v_i^h
=f_\theta^h
\left(
o_{i-1},a_{i-1},r_{i-1},o_i-o_{i-1}
\right)
\]

를 만들고, 여러 head가 이를 sum, exponential smoothing 또는 attention 방식으로 누적하여 history representation \(z_t\)를 구성한다. 여기서 \(o_i-o_{i-1}\)가 PID의 derivative와 대응하는 핵심 difference coordinate다.

따라서 현재 방식과의 관계는 다음과 같다.

| 구분 | Char and Schneider PIDE | Char and Schneider GPIDE | 현재 방식 |
|---|---|---|---|
| 차분 대상 | tracking error \(e_t-e_{t-1}\) | 전체 observation \(o_t-o_{t-1}\) | pressure \(P_t-P_{t-1}\) |
| history 처리 | P/I/D로 압축 | learned projection + structured aggregation | raw history를 유지하고 rate를 추가 |
| past action | tracking task observation에 last action 포함 | encoder 입력에 \(a_{t-1}\) 포함 | 10-step action history |
| 새 정보인가 | 아님 | 아님 | 아님 |
| inductive bias | tracking feedback | 일반적인 temporal change/accumulation | pressure trend와 damping |

현재 구현은 PIDE처럼 history를 P/I/D로 완전히 압축하지 않고, raw pressure/action history를 유지하면서 \(\dot P\)를 추가한다. 따라서 더 정확한 표현은 다음이다.

> 본 연구의 explicit pressure-rate observation은 Char and Schneider의 PID-inspired history encoding과 같은 원리, 즉 control-relevant differencing operation을 미리 제공하는 원리를 사용한다. 다만 본 연구는 이를 delayed pneumatic flow의 realized response를 나타내는 physics-derived coordinate로 사용하고 raw history와 함께 보존한다.

#### \(\dot P\)와 error derivative는 같지 않다

현재 tracking error를 \(e_t=r_t-P_t\)로 정의하면

\[
\dot e_t=\dot r_t-\dot P_t.
\]

- plateau에서는 \(\dot r_t=0\)이므로 \(\dot e_t=-\dot P_t\)다. 현재 관찰된 plateau 진동 감소는 PIDE의 D-term 해석과 직접 일치한다.
- ramp나 reference switching에서는 \(\dot r_t\neq0\)이므로 pressure rate만으로는 error derivative와 같지 않다.
- 현재 policy가 reference preview를 받으므로 이론적으로 \(\dot r_t-\dot P_t\)를 구성할 수 있지만, 이것도 finite network가 학습해야 하는 연산이다.

따라서 NeurIPS 논문을 강하게 연결하려면 다음 작은 진단 ablation이 유용하다.

| 입력 | 질문 |
|---|---|
| H | raw history만으로 충분한가 |
| H + \(\dot P\) | plant의 realized rate가 중요한가 |
| H + \(\dot e\) | tracking D-term 자체가 중요한가 |
| H + \(\dot P,\dot r\) | plant trend와 desired trend를 분리하는 것이 좋은가 |

\(\dot P\)가 plateau damping에만 좋고 \(\dot e\)가 ramp tracking에도 좋다면 PID-inspired 설명이 강해진다. 반대로 \(\dot P\)가 reference 종류와 무관하게 model mismatch에서 더 robust하면, 단순 PID error feature를 넘어 **plant-state trend proxy**라는 현재 시스템 고유의 해석이 가능하다.

#### 이 논문이 증명해 주는 것과 그렇지 않은 것

이 선행 연구를 다음 정도로 사용한다.

- **지원함:** redundant difference/sum feature도 finite history encoder에 유용한 inductive bias가 될 수 있음.
- **지원함:** partial observability와 simulator mismatch에서 structured temporal encoder가 flexible sequence encoder보다 robust할 수 있음.
- **직접 증명하지 않음:** pneumatic system에서 pressure rate가 반드시 필요함.
- **직접 증명하지 않음:** pressure rate가 MPObs를 대체함.
- **직접 증명하지 않음:** 현재 rate/50 scaling에서 얻은 성능이 물리 좌표 때문임.

논문은 tracking 문제뿐 아니라 locomotion benchmark에서도 기존 recurrent history encoder 대비 평균적으로 더 높은 성능과 robustness를 보고했다. 그러나 현재 논문에서는 그 결과를 pneumatic system의 결론으로 가져오지 말고, rate/reserve observation을 **inductive bias라고 부를 수 있는 선행 근거**로만 사용해야 한다.

### 4.5 “History-only만으로도 제어되어야 하지 않는가?”에 대한 답

#### 짧은 답: 표현 가능성 관점에서는 맞다

교수님의 지적은 이론적으로 맞다. 현재 rate와 reserve는 history의 선형 변환이므로, history-only policy에도 augmented policy와 동일한 control law가 존재한다. 이 사실은 universal approximation을 가정하지 않아도 현재와 같은 MLP의 첫 affine layer에서 정확히 보일 수 있다.

normalized history를 \(H\), rate/reserve feature를

\[
\phi(H)=LH
\]

라고 하자. 여기서 \(L\)은 finite difference와 pressure subtraction을 수행하는 고정 행렬이다. augmented MLP의 첫 층은

\[
h_1
=\sigma\left(W_HH+W_\phi\phi(H)+b\right)
=\sigma\left(W_HH+W_\phi LH+b\right).
\]

이를 정리하면

\[
h_1
=\sigma\left((W_H+W_\phi L)H+b\right).
\]

따라서 history-only MLP의 첫 층 weight를

\[
\widetilde W_H=W_H+W_\phi L
\]

로 두면 augmented MLP와 모든 입력에서 정확히 같은 hidden activation과 action을 출력한다. 즉 현재처럼 raw history를 그대로 유지하면서 linear rate/reserve를 concatenate하는 경우:

\[
\Pi_{\text{History+rate+reserve}}
=
\Pi_{\text{History}}
\]

가 expressivity 관점에서 성립한다. 같은 width의 MLP로도 충분하다.

그러므로 현재 실험에서 history-only가 reference 위아래로 진동하고 explicit rate policy가 안정화되었다는 사실은 다음을 의미하지 않는다.

> History에는 안정화에 필요한 정보가 없었다.

정확한 의미는 다음이다.

> 안정화 policy는 history-only network에도 표현 가능했지만, 현재 SAC·MLP·normalization·finite replay·exploration 조건에서는 그 high-pass/difference direction을 안정적으로 학습하지 못했다. Explicit rate가 optimization geometry와 data collection trajectory를 바꾸어 해당 solution에 도달하기 쉽게 했을 가능성이 있다.

#### 왜 같은 함수 집합인데 학습 결과가 달라지는가

함수의 표현 가능성과 optimizer가 그 함수를 찾는 것은 다른 문제다.

1. **입력 conditioning**
   - pressure history는 큰 absolute level과 작은 temporal difference가 겹쳐 있다.
   - derivative를 만들려면 인접 pressure에 크고 반대 부호인 weight가 필요하다.
   - explicit rate는 이 high-pass direction을 하나의 axis로 회전시킨 것과 같다.

2. **현재 normalization**
   - history 내부 차분은 \((P_t-P_{t-1})/100\)이다.
   - 현재 explicit rate는 \(P_t-P_{t-1}\) 크기로 들어간다.
   - 동일한 derivative sensitivity를 history-only 첫 층에서 만들려면 약 100배 큰 반대 부호 weight가 필요할 수 있다.

3. **implicit/explicit regularization**
   - 두 parameterization은 같은 함수를 표현하더라도 필요한 weight norm이 다르다.
   - initialization, Adam step, weight decay와 early stopping은 작은 norm/쉬운 gradient 방향을 선호한다.

4. **critic bootstrapping**
   - SAC actor만 derivative를 발견하면 되는 것이 아니다.
   - critic도 delayed transition에서 history의 어느 차분이 value-relevant한지 동시에 학습해야 한다.

5. **on-policy data distribution의 차이**
   - 초기 rate sensitivity가 조금만 달라도 overshoot와 exploration trajectory가 달라진다.
   - 그러면 replay buffer 자체가 달라지고 이후 학습 경로가 더욱 벌어진다.

따라서 explicit rate의 연구 가치는 “없던 정보를 추가했다”가 아니라 다음 중 하나로 검증해야 한다.

- sample-efficiency bias
- optimization/conditioning bias
- robustness-promoting temporal bias
- shorter-history compression

#### reference 위아래 진동의 가장 가능성 높은 해석

delay가 있는 tracking system에서 error level에만 강하게 반응하고 approach velocity를 충분히 사용하지 못하면 다음 limit-cycle이 생길 수 있다.

~~~text
pressure가 reference 아래
→ valve를 계속 크게 엶
→ delayed pressure가 reference를 통과해도 command 효과가 남아 있음
→ overshoot
→ 반대 valve command
→ 같은 과정이 반대 방향으로 반복
~~~

이는 P-like control은 학습했지만 D-like phase lead를 충분히 학습하지 못한 모습과 일치한다. Explicit \(\dot P\)는 “아직 error는 남아 있지만 reference를 빠르게 향하고 있으므로 미리 command를 줄여야 한다”는 방향을 쉽게 만든다.

하지만 trajectory 모양만으로 메커니즘을 확정하면 안 된다. 다음 세 검사가 필요하다.

- reference crossing 전후에 history-only와 rate policy가 command를 언제 줄이는지 비교
- 같은 \(e_t\)에서 \(\dot P_t\)가 다른 sample을 모아 action conditional plot 작성
- policy의 \(\partial\pi/\partial\dot P\) 또는 equivalent history-direction Jacobian 비교

#### 가장 강한 즉시 검증: first-layer folding

학습된 History+rate actor의 rate branch를 history weight로 정확히 접어 넣을 수 있다.

\[
\widetilde W_H=W_H+W_{\dot P}L_{\dot P}.
\]

그 뒤 rate input을 제거한 folded History-only actor를 실행하면 원래 actor와 numerical precision 내에서 같은 action과 rollout을 내야 한다. 이것은 새 학습이 필요 없다.

- 같게 작동하면: rate가 새 information이나 expressivity를 추가하지 않았다는 constructive proof.
- 그런데 scratch History-only만 실패하면: 차이는 representation이 아니라 learnability/optimization path에 있음.
- folded model도 다르면: feature reconstruction, normalization, history indexing 또는 implementation에 예상하지 못한 차이가 있음.

이 검사는 교수님의 질문에 가장 직접적인 답이면서, inductive-bias 주장을 오히려 더 정확하게 만든다.

#### “단순 observation 조합은 무한히 많은데 왜 이것인가?”

이 반론도 맞다. history에서 만들 수 있는 arbitrary nonlinear feature는 무한히 많기 때문에, 성능이 좋아지는 조합을 사후 탐색하면 feature engineering 이상의 결론을 내기 어렵다.

따라서 candidate feature는 다음 기준으로 제한한다.

1. **governing equation에서 유도되는가**
   - \(\dot P\): net mass-flow와 volume change가 실현한 pressure response
   - \(P_{ch}-P_{act}\): valve 양단의 local flow headroom
   - action history: valve/delay pipeline에 들어간 known input

2. **하나의 failure mechanism과 연결되는가**
   - rate: overshoot와 phase/damping
   - reserve: authority depletion과 chamber safety
   - action history: delayed actuation과 hysteresis-state inference

3. **real system에서 동일하게 측정·계산 가능한가**

4. **낮은 차원이며 명확한 placebo가 있는가**
   - rate vs scale-matched rate vs zero4
   - reserve vs zero2

5. **사전에 반증 조건을 정할 수 있는가**
   - scale/placebo 후 효과가 사라지면 rate 고유효과를 철회
   - 충분히 학습한 history encoder가 따라잡으면 “필수 feature” 주장을 철회
   - mismatch/real에서만 효과가 남으면 robustness bias로 결론을 제한

즉 답은 “가능한 조합 중 잘 되는 것을 골랐다”가 아니라 다음이어야 한다.

> 후보를 pneumatic governing equation과 관찰된 failure mode에서 사전에 제한했고, 각 feature가 새 정보인지, 단순 scaling인지, optimization bias인지, 물리적으로 예상한 mechanism을 실제로 사용하는지를 별도 control로 분해한다.

#### 이 피드백을 판정할 최소 비교

| 비교군 | 판정 질문 |
|---|---|
| H-MLP scratch | vanilla history 학습의 기준 |
| H-MLP larger/longer | 단순 capacity·training 부족인가 |
| H + scale-matched \(\dot P\) | coordinate bias가 남는가 |
| H + zero4 | dimension/fan-in 효과인가 |
| H + fixed difference layer | 정보는 H만 사용하면서 structured operation만 준 효과인가 |
| PIDE/GPIDE 또는 1D temporal encoder | 임의 MLP보다 적절한 history architecture가 해결하는가 |
| folded H actor | augmented policy와 expressivity가 정확히 동일한가 |

결과 해석은 다음과 같다.

- H가 충분한 학습 후 따라잡음: rate는 convergence/sample-efficiency bias.
- fixed difference/PIDE만 안정적: structured temporal representation이 핵심.
- larger H-MLP도 계속 실패하지만 folded H는 성공: optimization landscape 문제가 강함.
- 모든 H 계열이 multi-seed에서 따라잡음: explicit rate의 논문적 contribution은 철회.
- rate가 mismatch/real에서만 반복적으로 우세: simulator temporal pattern에 덜 과적합하는 robustness bias.

#### 교수님께 답할 문장

> 맞습니다. Information과 expressivity 관점에서는 history-only만으로 같은 제어가 가능해야 합니다. Rate와 reserve가 history의 선형 조합이기 때문에, augmented actor의 첫 층 weight를 history 쪽으로 접으면 완전히 동일한 history-only actor를 만들 수도 있습니다. 따라서 제가 주장하려는 것은 rate가 없으면 제어가 원리적으로 불가능하다는 것이 아닙니다. 현재 관찰한 진동은 vanilla SAC가 delay가 있는 pressure history에서 derivative 방향을 학습하지 못한 optimization failure일 가능성이 큽니다. 제가 확인해야 할 연구 질문은 explicit difference coordinate 또는 PID-inspired encoder가 이 가능한 해를 더 빠르고 반복적으로 찾게 하는지, 그리고 그 이득이 scale control과 multi-seed, model mismatch 및 real system에서도 남는지입니다. 그렇지 않으면 rate contribution은 철회하겠습니다.

### 4.6 실제 History-only seed-0은 더 학습하면 따라잡는 추세였는가

확인 모델:

~~~text
pneu_rl/models/260810_lib3_ablation_profile_hist_base_seed0
~~~

100-episode 구간별 combined RMSE는 다음과 같다. 이는 episode별 POS/NEG RMSE의 root-mean-square이며, training return과 달리 추적 오차를 직접 나타낸다.

| Episode | combined RMSE |
|---:|---:|
| 1–100 | 7.699 |
| 101–200 | 8.187 |
| 201–300 | 5.437 |
| 301–400 | 4.587 |
| 401–500 | 5.343 |
| 501–600 | 4.245 |
| 601–700 | 3.732 |
| **701–800** | **3.144** |
| 801–900 | 3.442 |
| 901–1000 | 3.480 |

History-only는 701–800에서 가장 좋았고 이후 두 구간에서 개선되지 않았다. Reference/carry cycle과 serial dependence의 영향을 줄이기 위해 episode 601–1000을 연속 16-episode block 25개로 묶어 회귀한 결과:

- combined RMSE slope: 100 episode당 -0.065
- RMSE slope 95% interval: [-0.264, +0.133]

즉 **관측된 601–1000 범위에서는 combined RMSE 기준 유의미한 catch-up trend가 없다.**

같은 training log 기준으로 R50은 episode 101–200에서 combined RMSE 2.492에 도달했고, 같은 구간 History-only는 8.187이었다. 현재 scaling을 포함한 explicit rate가 early learning을 크게 바꾼 것은 분명하다.

다만 다음 문장까지만 말할 수 있다.

> History-only는 1000 episode budget 안에서 600 episode 이후 plateau에 들어갔고 R50을 따라잡는 추세를 보이지 않았다.

다음 문장은 아직 증명되지 않았다.

> History-only는 아무리 오래 학습해도 rate policy를 따라잡을 수 없다.

그 이유는 seed가 하나이고, 1000 episode 이후 continuation data가 없으며, 훈련 중 reference 난이도와 carry state가 변하기 때문이다. 현재 checkpoint에는 actor, critic, target critic, optimizer와 alpha state가 모두 저장되어 있어 continuation 자체는 가능하다.

더 엄밀한 convergence 판정은 다음과 같이 수행한다.

1. H와 rate 계열 모두 1000 episode checkpoint에서 2000까지 continuation한다.
2. 50 episode마다 checkpoint를 저장하고 동일한 16-member fixed evaluation bank로 평가한다.
3. training return이 아니라 paired fixed-bank RMSE, direction, HF ripple, constraint violation을 primary outcome으로 둔다.
4. scratch seed 0/1/2와 checkpoint-continuation seed를 분리한다.
5. 16-episode 또는 carry-cycle block 단위 slope와 confidence interval을 사용한다.
6. learning AUC, time-to-threshold와 final evaluation을 각각 보고한다.
7. H-MLP larger/longer, fixed difference layer와 PIDE/GPIDE를 함께 두어 “시간 부족”과 “history architecture 부족”을 분리한다.

## 5. 왜 \(\dot P\)가 진동 감소에 도움을 줄 수 있는가

local closed-loop 설명은 가능하다. plant를 동작점 근처에서 다음 second-order pressure mode로 근사한다.

\[
\ddot P+2\zeta\omega_n\dot P+\omega_n^2P=bu.
\]

policy가 local하게

\[
u=k_p(r-P)-k_d\dot P
\]

를 구현하면 closed loop는

\[
\ddot P+
(2\zeta\omega_n+bk_d)\dot P+
(\omega_n^2+bk_p)P
=bk_pr.
\]

\(bk_d>0\)인 영역에서는 effective damping이 증가한다. 이 식은 “왜 rate가 진동을 줄일 수 있는가”에 대한 local sufficient explanation이지, nonlinear SAC 전체의 안정성 증명은 아니다.

실제 policy가 이 메커니즘을 사용했는지는 다음으로 확인한다.

1. plateau sample에서 policy Jacobian \(\partial\pi/\partial\dot P\)를 계산한다.
2. finite perturbation으로 \(+\epsilon,-\epsilon\) rate를 넣고 valve action 방향을 확인한다.
3. local plant \(A,B\)를 식별하고 \(A+B\,\partial\pi/\partial x\)의 eigenvalue 또는 discrete spectral radius를 비교한다.
4. pressure PSD/HF ripple, action HF RMS, total variation을 함께 보고 tracking RMSE와 분리한다.
5. derivative는 noise를 \(1/\Delta t\)만큼 증폭하므로 raw difference와 filtered rate를 비교한다.

현재 H10 대비 H10+rate의 seed-0 결과는 다음과 같다.

- combined RMSE: 4.370 → 3.922 kPa
- plateau HF ripple: POS 약 91%, NEG 약 75% 감소
- action HF RMS: 약 60% 감소
- mean absolute action difference: 약 20% 감소

이는 damping 가설과 일치하지만 scale confound와 single seed 때문에 **mechanistic evidence의 시작점**으로만 제시한다.

---

## 6. \(P_{ch}-P_{act}\)는 왜 넣는가

reserve는 actuator valve 양단의 local flow-driving condition을 명시한다.

~~~text
absolute pressure 4개를 network가 조합
              ↓
positive/negative branch별 available headroom을 직접 제공
~~~

의미는 다음과 같이 제한해야 한다.

- reserve가 크면 같은 opening에서 더 큰 flow가 가능한 경우가 많다.
- 하지만 compressible flow는 pressure ratio와 upstream absolute pressure에 의존한다.
- choked regime에서는 downstream pressure 변화의 영향이 작을 수 있다.
- 따라서 reserve는 “control authority 전체”가 아니라 **local headroom proxy**다.

현재 predictor에서 full history+rate가 0.50 s RMSE 1.173, 여기에 reserve를 더하면 1.165로 차이가 매우 작았다. matched-pulse에서도 POS branch는 reserve sensitivity가 있었지만 NEG branch는 choking을 통제하면 약했다. 따라서 reserve는 rate와 묶어 하나의 필수 observation으로 주장하지 말고, **safety/resource management 또는 state-dependent authority 가설**로 별도 검증하는 편이 낫다.

---

## 7. \(\dot P\)와 MPObs는 어떤 관계인가

MPObs는 model과 action sequence를 사용하여 future observation을 rollout한다.

\[
\hat x_{t+k+1}=\hat f(\hat x_{t+k},u_{t+k}),
\qquad
m_t=[\hat y_{t+1},\ldots,\hat y_{t+K}].
\]

이를 짧은 horizon에서 local linear/Taylor expansion하면

\[
\hat y_{t+k}
\approx
y_t+k\Delta t\,\dot y_t
+\sum_{j=0}^{k-1}CA^{k-1-j}Bu_{t+j}
+O((k\Delta t)^2)+\epsilon_{model}.
\]

여기서 explicit \(\dot P\)가 주는 것은 주로 첫 번째 **free-response/drift term**이다. history와 action history는 hidden state와 delayed input effect를 간접 추정한다. 그러나 rate에는 다음이 없다.

- proposed future action의 effect
- nonlinear multi-step rollout
- chamber–actuator coupled future prediction
- reference별 future controllability 정보

따라서 정확한 문장은 다음이다.

> rate는 short-horizon MPObs가 제공하는 정보 중 현재 pressure trajectory의 1차 drift 성분을 model-free하게 명시한다. 따라서 model mismatch가 큰 영역에서는 더 robust할 수 있지만, MPObs 전체와 동등하지는 않다.

대체인지 보완인지는 아래 2×2 ablation으로 판단한다.

| | MPObs 없음 | MPObs 있음 |
|---|---|---|
| rate 없음 | H | H + M |
| rate 있음 | H + D | H + D + M |

판정량을

\[
\Delta_M^{(0)}=J(H+M)-J(H),\qquad
\Delta_M^{(D)}=J(H+D+M)-J(H+D)
\]

로 둔다.

- \(\Delta_M^{(0)}>0\), \(\Delta_M^{(D)}\approx0\): rate가 해당 조건에서 MPObs의 실용적 이득을 대부분 대체.
- 둘 다 양수: rate와 MPObs가 보완적.
- nominal sim에서는 MPObs가 좋고 model-mismatch/real에서는 rate가 좋음: model-free drift feature의 robustness가 핵심 결과.
- rate 효과가 scale-matched/zero-placebo 후 사라짐: 기존 개선은 representation보다 scaling/optimization effect.

모든 모델은 같은 base history, 같은 action dimension, 같은 CAPS 설정, 같은 safety constraint를 써야 한다. 기존 MPObs 구현에서 multi-step action dimension까지 함께 변한다면 MPObs 단독 효과가 아니므로 반드시 고정해야 한다.

---

## 8. 이번 주에 수행할 최소 실험

### A. 새 RL 학습 없이 가능한 것

1. **Simulator state audit**
   - pressure를 결정하는 hidden state와 식을 한 장 diagram으로 정리.
   - current-pressure non-Markov counterexample를 코드 식과 함께 제시.

2. **Predictive observation test**
   - 같은 dataset, network, future action에서 current / P-history / P+u-history / +rate / +reserve 비교.
   - target은 0.02, 0.10, 0.20, 0.50 s future pressure.
   - episode split을 사용해 adjacent sample leakage 방지.

3. **Matched pulse test**
   - current pressure/reserve가 같은 pair를 선택.
   - recent rate/action history만 다르게 만든 뒤 동일 pulse 적용.
   - next-pressure conditional spread를 측정.

4. **Policy mechanism test**
   - 저장된 policy의 \(\partial\pi/\partial\dot P\)와 finite perturbation 방향 검사.
   - plateau pressure/action PSD와 local damping 지표 계산.

### B. 새 학습이 필요한 최소 ablation

우선순위 1:

\[
H / H+D_{50} / H+D_{5000} / H+Z_4
\]

- 동일 chamber safety reward/termination
- 동일 reference bank와 training budget
- seed 0/1/2 이상
- learning AUC와 final performance를 분리
- tracking뿐 아니라 HF ripple, action variation, reserve violation을 함께 평가

우선순위 2:

\[
H / H+D / H+M / H+D+M
\]

- nominal simulator
- valve time constant/hysteresis/delay mismatch sweep
- 마지막으로 같은 real trajectory와 fixed-budget fine-tuning

reserve는 이 두 핵심 질문과 분리하여 \(H / H+reserve / H+zero2\)로 검증한다.

### C. Lambda discrepancy를 이용한 control-relevant Markovness 검사

Allen et al.의 NeurIPS 2024 “Mitigating Partial Observability in Sequential Decision Processes via the Lambda Discrepancy”는 observation이 value prediction에 충분한지 검사하는 보조 분석이다. 다만 이 metric 하나로 “어떤 observation을 넣은 policy가 더 잘 제어하는가”를 직접 판정하는 방법은 아니다.

#### 핵심 직관: average와 propagation의 순서가 달라진다

같은 observation \(z\)가 실제로는 여러 hidden state \(s_1,s_2\)를 섞고 있다고 하자.

- TD(0)는 \(z\)를 하나의 state로 취급하고 매 one-step backup마다 hidden states를 평균낸 effective MDP를 평가한다.
- TD(1), 즉 Monte Carlo는 실제 hidden-state trajectory를 끝까지 따라간 return을 계산한 뒤 observation \(z\)별로 평균낸다.

즉 두 계산은 개념적으로 다음과 같다.

~~~text
TD(0): hidden states를 observation으로 투영/평균
       → 한 step 전파
       → 다시 투영/평균

TD(1): 실제 hidden state trajectory를 여러 step 전파
       → 마지막에 observation별로 평균
~~~

Markov representation이면 “평균한 뒤 전파”와 “전파한 뒤 평균”이 같은 value fixed point를 만든다. Observation aliasing이 있으면 hidden-state identity와 trajectory correlation이 중간에 사라지므로 보통 두 fixed point가 달라진다.

#### TD(\(\lambda\))와 discrepancy

\(n\)-step return을

\[
G_t^{(n)}
=
\sum_{k=0}^{n-1}\gamma^kr_{t+k}
+\gamma^nV(z_{t+n})
\]

라 하면 \(\lambda\)-return은 여러 \(n\)-step return의 exponential mixture다.

\[
G_t^\lambda
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}G_t^{(n)}.
\]

- 작은 \(\lambda\): short backup과 bootstrap 비중이 큼.
- 큰 \(\lambda\): long actual return과 Monte Carlo 비중이 큼.
- Markov state에서는 모든 \(n\)-step Bellman unrolling이 같은 value를 나타내므로 모든 \(\lambda\) fixed point가 같음.
- POMDP observation에서는 \(\lambda\)에 따라 effective dynamics를 평균내는 정도가 달라져 fixed point가 달라질 수 있음.

논문은 두 TD(\(\lambda\)) Q-function fixed point의 차이를 다음처럼 정의한다.

\[
\Lambda_{\lambda_1,\lambda_2}^{\pi}(Z)
=
\left(
\mathbb E_{(z,a)\sim d_\pi}
\left[
\left(
Q_{\pi,Z}^{\lambda_1}(z,a)
-
Q_{\pi,Z}^{\lambda_2}(z,a)
\right)^2
\right]
\right)^{1/2}.
\]

Expectation은 fixed policy \(\pi\)가 방문하는 on-policy observation-action distribution에 대해 계산한다. 실제 deep implementation에서는 PPO와 GRU를 사용하고 두 \(V^\lambda\) head의 squared difference를 auxiliary loss로 최소화했다.

#### 논문의 이론이 실제로 보장하는 것

1. Markov/block-MDP observation이면 모든 policy에서 exact lambda discrepancy가 0이다.
2. 어떤 policy 하나에서 non-zero discrepancy를 만들 수 있는 POMDP라면 measure-zero policy 집합을 제외한 거의 모든 policy에서 non-zero다.
3. 하지만 non-Markov observation인데도 reward/transition/observation의 대칭성 때문에 차이가 projection 과정에서 사라지는 false negative가 존재한다. 논문의 Parity Check가 그 예다.
4. Partial observability가 커질수록 discrepancy가 항상 단조 증가한다는 일반 정리는 없다. 논문의 monotonic plot은 T-maze observation을 controlled interpolation한 실험이다.
5. Deep auxiliary loss는 수렴한 fixed points의 차이가 아니라 학습 중인 두 value estimate의 차이다. 저자도 이를 이론의 근사라고 limitation에 명시한다.

따라서 lower discrepancy는 “그 fixed policy와 reward에 대해 value prediction이 더 Markov-like하다”는 증거이지, 다음을 직접 보장하지 않는다.

- 더 높은 RL return
- 더 좋은 tracking 또는 damping
- full physical state reconstruction
- 모든 policy에서 더 좋은 representation

#### 현재 observation 중 무엇을 비교할 수 있는가

먼저 representation을 다음처럼 나눈다.

| Representation \(Z\) | 질문 |
|---|---|
| Current pressure + reference | current observation은 value prediction에 Markov한가 |
| Pressure history + reference | measurement history가 discrepancy를 낮추는가 |
| Pressure/action history + reference | delayed action context가 추가로 낮추는가 |
| H + explicit \(\dot P\) | 같은 정보의 coordinate bias가 finite critic discrepancy를 낮추는가 |
| H + reserve | headroom coordinate가 value prediction에 도움 되는가 |
| Privileged simulator state + reference | estimator/function-approximation floor는 얼마인가 |

비교 결과의 정확한 해석:

| 결과 | 허용되는 해석 |
|---|---|
| \(\Lambda(Current)>\Lambda(P\text{-history})\) | measurement history가 value-relevant aliasing을 줄임 |
| \(\Lambda(P\text{-history})>\Lambda(P/U\text{-history})\) | past action이 delayed-input ambiguity를 줄임 |
| \(\Lambda(H)\approx\Lambda(H+\dot P)\) | 두 representation이 같은 정보를 가진다는 이론과 일치 |
| finite critic에서 \(H+\dot P\)가 더 빨리 낮아짐 | difference coordinate가 value learning/conditioning을 도움 |
| \(\Lambda(Privileged)>0\) | estimator가 아직 수렴하지 않았거나 function approximation/finite-horizon error가 큼 |

특히 \(H+\dot P\)와 \(H+reserve\)는 주의해야 한다. Rate와 reserve는 \(H\)의 deterministic linear transformation이므로:

\[
\sigma(H)=\sigma(H,\dot P,reserve).
\]

따라서 exact lambda discrepancy로는 세 representation이 같아야 한다. Lambda discrepancy는 rate가 history에 없는 partial observability를 해결했다고 증명할 수 없다. 다만 fixed-size neural value learner가 같은 정보를 어느 coordinate에서 더 잘 사용하는지는 학습속도와 residual empirical discrepancy로 볼 수 있다.

#### 공정한 비교를 위한 policy 선택

Lambda discrepancy는 policy-dependent하다. 서로 다른 observation으로 각각 학습한 H, R50, rate+reserve policy를 사용하면 policy와 state-visitation distribution이 모두 달라지므로 discrepancy 크기를 바로 순위화할 수 없다.

Representation 비교용 main test에서는 모든 representation이 실행할 수 있는 **공통 policy**가 필요하다.

권장 공통 policy:

- current pressure/reference만 사용하는 stochastic policy
- 또는 observation과 무관한 bounded valve pulse/random-excitation policy

같은 common policy로 trajectory를 한 번 수집하고 각 timestep에서 Current, P-history, P/U-history, H+rate, privileged state를 모두 사후 구성한다. 한 policy가 너무 좁은 operating region만 방문하면 aliasing이 드러나지 않을 수 있으므로 여러 bounded excitation policy를 사용하고 결과를 policy별로 보고한다.

H+rate trained policy를 그대로 Current representation 평가에 사용하는 것은 main theorem과 맞지 않는다. 그 policy의 action이 Current만의 함수가 아니기 때문이다. 각 trained policy의 자체 discrepancy는 별도 진단으로 보고할 수 있지만 representation 간 공정한 ranking으로 사용하지 않는다.

#### 현재 시스템용 sample-based 비교 절차

1. **공통 policy 고정**
   - current-only stochastic policy 또는 bounded pulse policy 3–5개를 미리 고정.
   - reference와 reward도 모두 동일.

2. **on-policy sequence 수집**
   - policy별 독립 2048-step rollout을 충분히 수집.
   - carry가 이어지는 네 512-step block을 하나의 sequence로 취급.
   - 모든 raw pressure, action, reference와 privileged simulator state 저장.

3. **Representation을 post-hoc 생성**
   - Current
   - pressure history
   - pressure/action history
   - H+rate
   - H+reserve
   - privileged state

4. **두 value head 학습**
   - representation별 같은 width/optimizer/data split 사용.
   - low-\(\lambda\)와 high-\(\lambda\) target을 각각 학습.
   - 이론 설명에는 0/1이 명확하지만 MC variance 때문에 실제 시작값은 논문의 deep experiment처럼 0.1/0.95가 현실적.
   - 0.1/0.5, 0.1/0.9, 0.1/0.95 sensitivity도 확인.

5. **held-out trajectory에서 측정**

\[
\widehat\Lambda_Z
=
\sqrt{
\frac{1}{N}
\sum_{t=1}^N
\left(
V_{Z,\lambda_{low}}(z_t)
-
V_{Z,\lambda_{high}}(z_t)
\right)^2
}.
\]

Reward scale로 normalization하려면:

\[
\widehat\Lambda_Z^{norm}
=
\frac{\widehat\Lambda_Z}
{\operatorname{Std}(G_t)+\epsilon}.
\]

6. **estimation-error control**
   - value-network seed 5개 이상
   - trajectory-level paired bootstrap interval
   - 두 head의 held-out \(\lambda\)-return prediction error를 함께 보고
   - privileged-state result를 experimental floor로 사용
   - shuffled-history positive control과 zero-feature dimension placebo 추가

#### 무엇을 primary conclusion으로 둘 것인가

Lambda discrepancy와 다른 실험의 역할은 다르다.

| 분석 | 측정하는 것 |
|---|---|
| Future-pressure predictor | observation이 plant dynamics를 얼마나 예측하는가 |
| Lambda discrepancy | observation이 현재 policy/reward의 value prediction에 얼마나 Markov한가 |
| RL return/oscillation | 그 representation으로 실제 control을 얼마나 잘 학습했는가 |

Observation이 “더 효과적”이라는 최종 판단에는 최소한 다음 두 조건이 모두 필요하다.

1. common-policy test에서 lambda discrepancy 또는 predictive aliasing이 감소.
2. independently trained multi-seed policy에서 fixed-bank tracking/damping/safety가 개선.

Lambda discrepancy만 낮고 control이 같으면 representation은 더 Markov-like하지만 practical policy gain은 없다. Control만 좋고 discrepancy가 같다면 partial observability 해결보다 optimization bias일 가능성이 높다.

#### SAC에 바로 auxiliary loss로 넣지 않는 이유

논문의 scalable algorithm은 online recurrent PPO와 truncated \(\lambda\)-return을 사용한다. 현재 SAC는 off-policy replay를 사용하므로 그대로 이식하면:

- behavior/evaluation policy mismatch
- sequential trace sampling
- importance weighting 또는 Retrace/V-trace
- target critic과 두 \(\lambda\)-head의 안정성

문제가 추가된다. 따라서 1단계는 common-policy fresh rollout을 이용한 diagnostic이다. 이 진단이 Current와 History를 안정적으로 구분한 뒤에만 recurrent SAC auxiliary loss를 별도 method contribution으로 고려한다.

이 분석이 성공하면 다음과 같은 강한 결과 구성이 가능하다.

~~~text
Current observation
→ 높은 lambda discrepancy

Pressure/action history
→ discrepancy 감소

History + rate
→ true information은 같지만 finite critic discrepancy와 learning time 감소

Privileged state
→ estimator floor
~~~

---

## 9. 이 실험으로 논문에서 무엇을 결론낼 수 있는가

### 결과가 잘 나왔을 때의 중심 연구 질문

> Can a real-measurable, physics-structured history representation recover a control-relevant approximate information state and closed-loop damping more robustly than model-predicted observations in a delayed, hysteretic pneumatic soft-actuator system?

한국어로는 다음과 같다.

> 지연과 hysteresis가 있는 pneumatic soft-actuator에서 측정 가능한 pressure/action history가 control-relevant approximate state를 구성하는가? 그리고 explicit pressure-rate coordinate가 model-predicted observation보다 단순하고 robust하게 short-horizon trend와 damping을 제공하는가?

### 가능한 contribution 세 개

1. **문제 규명:** 6-valve chamber–actuator system에서 current pressure가 non-Markov인 구조적 이유와 matched intervention 증거를 제시.
2. **표현 설계:** pressure/action history와 physics-derived rate/reserve를 “정보 복원”과 “coordinate bias”로 분리하고, scale/placebo-controlled ablation으로 효과를 검증.
3. **선행법과의 경계:** rate와 MPObs의 관계를 local expansion으로 설명하고, nominal/mismatch/real에서 대체·보완 조건을 실험적으로 규명.

이 방향이면 novelty는 “SAC에 observation 몇 개를 더 넣었다”가 아니다. 핵심은 다음이다.

> **복잡해진 pneumatic plant에서 무엇이 non-Markovness를 만들고, model rollout 없이 측정 가능한 temporal physics coordinate가 어느 범위까지 그 문제를 완화하는지 분해해 보였다.**

### 결과가 약하게 나와도 남는 결론

- history가 필요하고 rate가 불필요하다면: 충분한 history가 derived feature를 흡수한다는 negative result.
- rate가 short history에서만 좋다면: rate는 history compression/sample-efficiency bias.
- MPObs가 항상 더 좋다면: rate는 damping 보조 feature이고 prediction은 대체하지 못함.
- rate가 mismatch/real에서만 좋다면: model-free robustness가 contribution.
- reserve 효과가 없으면: raw pressure로 충분하거나 해당 operating region이 choked임을 보인 결과.

즉 어떤 결과가 나와도 질문이 잘 분해되어 있으면 논문의 결론을 만들 수 있다.

---

## 10. 교수님께 그대로 설명할 90초 버전

> 현재 시스템에서 pressure 네 개만 보는 것은 Markov하지 않습니다. simulator 내부에는 각 valve의 hysteresis와 dynamic flow state, pump state, 0.1초 delay buffer가 있고, pressure 미분식은 이 hidden state가 만든 mass flow에 직접 의존합니다. 그래서 같은 현재 pressure와 같은 command라도 다음 pressure가 달라질 수 있습니다. 실제 matched-pulse에서도 현재 pressure를 맞췄는데 recent history가 다르면 0.5초 response가 POS 4.28 kPa, NEG 6.59 kPa 달랐습니다. 따라서 pressure와 action history의 역할은 hidden dynamics로 인한 state aliasing을 줄이는 것입니다.
>
> 반면 pressure rate와 chamber–actuator pressure difference는 history에서 계산 가능하므로 새 정보는 아닙니다. 이것을 필수 observation이라고 주장하면 안 되고, finite MLP가 두 absolute pressure를 빼서 trend나 headroom을 학습하는 대신 그 물리 좌표를 직접 준 inductive bias라고 설명하는 게 맞습니다. 이 해석은 NeurIPS 2023의 PID-inspired encoder와 직접 연결됩니다. 그 논문도 history에서 계산 가능한 error derivative와 observation difference를 명시하여 partial observation과 simulator mismatch에 더 적합한 temporal bias를 만들었습니다. 다만 그 논문은 error derivative를 사용하고 우리는 plant pressure rate를 사용하므로, plateau에서는 동일한 D-term 해석이 가능하지만 ramp에서는 \(\dot e=\dot r-\dot P\)를 별도로 비교해야 합니다. Rate가 진동을 줄인 것은 local하게 derivative feedback이 damping을 증가시키는 식으로 설명할 수 있지만, 실제 policy Jacobian과 scale-controlled multi-seed ablation으로 확인해야 합니다.
>
> MPObs와 rate도 동일하지 않습니다. 짧은 horizon에서 MPObs를 전개하면 현재 pressure, pressure rate, future action effect로 나뉘는데 rate는 그중 free-response의 1차 항만 제공합니다. 그래서 history, rate, MPObs, rate+MPObs의 2×2 비교를 nominal simulation, model mismatch와 real에서 수행하면 rate가 MPObs를 대체하는지, 보완하는지 결론낼 수 있습니다. 논문의 기여는 SAC 성공 자체가 아니라, 이 복잡한 pneumatic system에서 non-Markovness의 원인을 밝히고 측정 가능한 history와 physics coordinate가 어떤 역할을 하는지 분해해 검증하는 것입니다.

---

## 11. 직접 연결되는 참고문헌

1. C. Xia, J. Zhang, E. C. Kerrigan, and G. Rigas, “Active flow control for bluff body drag reduction using reinforcement learning with partial measurements,” *Journal of Fluid Mechanics*, vol. 981, A17, 2024. [DOI / article](https://doi.org/10.1017/jfm.2024.69)  
   부분 pressure measurement와 delay에서 과거 measurement/action을 NARX state로 사용한 직접적인 application example.

2. J. Subramanian, A. Sinha, R. Seraj, and A. Mahajan, “Approximate Information State for Approximate Planning and Reinforcement Learning in Partially Observed Systems,” *JMLR*, vol. 23, 2022. [Paper](https://jmlr.org/papers/v23/20-1165.html)  
   history representation이 reward와 next observation을 예측하면 approximate information state로 다룰 수 있다는 이론적 틀.

3. G. Patil, A. Mahajan, and D. Precup, “On learning history-based policies for controlling Markov decision processes,” *AISTATS*, 2024. [Paper](https://proceedings.mlr.press/v238/patil24b.html)  
   history-based feature abstraction을 사용하는 RL policy의 이론적 분석.

4. T. Ni et al., “Bridging State and History Representations: Understanding Self-Predictive RL,” *ICLR*, 2024. [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/666c1861d709bd84e20b6e0e02a2c223-Abstract-Conference.html)  
   좋은 history/state representation을 next-state/reward prediction 관점에서 해석하는 근거.

5. I. Char and J. G. Schneider, “PID-Inspired Inductive Biases for Deep Reinforcement Learning in Partially Observable Control Tasks,” *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*, 2023. [Official paper](https://papers.nips.cc/paper_files/paper/2023/hash/ba1c5356d9164bb64c446a4b690226b0-Abstract-Conference.html)  
   PIDE는 tracking error의 P/I/D를, GPIDE는 observation difference와 previous action/reward를 structured history encoder에 사용한다. 현재 explicit pressure rate를 inductive bias로 해석하는 가장 직접적인 선행 근거.

6. G. E. Karniadakis et al., “Physics-informed machine learning,” *Nature Reviews Physics*, vol. 3, pp. 422–440, 2021. [Paper](https://doi.org/10.1038/s42254-021-00314-5)  
   물리 지식을 feature, architecture, constraint 등에 넣는 broader physics-informed learning 맥락. 현재 rate feature를 PINN이라고 부르기보다 physics-structured inductive bias라고 구분하는 데 사용.

7. S. Park et al., “Modeling and Reinforcement Learning-Based Control of Simultaneous Positive and Negative Pressure Generation in Pneumatic Systems,” *IEEE Robotics and Automation Letters*, 2025. [DOI](https://doi.org/10.1109/LRA.2025.3577422)  
   현재 연구의 직접 predecessor이자 MPObs/CAPS 비교 기준.

8. C. Allen et al., “Mitigating Partial Observability in Sequential Decision Processes via the Lambda Discrepancy,” *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*, 2024. [Official paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/73073ccb3bc559fd001e66b9079d6d5e-Abstract-Conference.html)  
   서로 다른 TD(\(\lambda\)) value estimate의 discrepancy로 representation의 value-relevant non-Markovness를 검출하고 memory learning을 보조하는 방법.

## 관련 내부 분석

- [observation predictive evidence](ablation/lib3_0807/observation_predictive_evidence.md)
- [rate scale/placebo control](ablation/lib3_0807/observation_coordinate_control_experiment.md)
- [lib4 partial observability note](lib4_partial_observability_note.md)
- [observer/POMDP system audit](observer_pomdp_system_audit.md)
- [research novelty review](research_novelty_review.md)

---

## 12. 랩미팅용 약 2페이지 요약

### Title

**Soft Actuator pneumatic RL에서 history observation의 필요성과 pressure-rate inductive bias 검증**

### Recommendation

현재 결과를 “\(\dot P\)가 없으면 제어가 불가능하다”로 주장하지 않는다. Pressure/action history는 hidden valve dynamics와 delay로 인한 state aliasing을 줄이는 정보로, \(\dot P\)와 \(P_{ch}-P_{act}\)는 같은 history에서 계산 가능한 physics-derived coordinate로 구분한다. 이번 주에는 History-only의 convergence plateau, scale-controlled rate 효과와 lambda discrepancy를 사용해 information insufficiency와 finite-learning bias를 분리한다.

### 1. 지난 피드백과 핵심 질문

현재 observation에는 pressure 변화율 4개와 chamber–actuator pressure difference 2개를 추가했다. 피드백은 이 값들이 기존 pressure history의 조합으로 계산 가능하므로 history-only만으로도 제어가 되어야 하지 않느냐는 것이었다.

이론적으로 이 피드백은 맞다. Rate와 reserve를 \(\phi(H)=LH\)라 하면 augmented MLP 첫 층

\[
\sigma(W_HH+W_\phi LH+b)
\]

은 history-only 첫 층

\[
\sigma((W_H+W_\phi L)H+b)
\]

로 정확히 변환된다. 따라서 rate와 reserve는 새로운 information이나 expressivity를 추가하지 않는다. 현재 연구 질문은 “이 feature가 필수인가?”가 아니라 다음으로 수정해야 한다.

> Delayed nonlinear pneumatic system에서 pressure/action history는 control-relevant partial observability를 얼마나 줄이는가? 그리고 explicit pressure-rate coordinate는 vanilla MLP/SAC가 history에 이미 존재하는 derivative direction을 더 빠르고 robust하게 학습하도록 하는가?

### 2. 왜 current pressure만으로는 부족한가

Pressure dynamics는

\[
\dot P=-\frac{P}{V}\dot V+\frac{RT}{V}\dot m
\]

이고 mass flow는 upstream/downstream pressure뿐 아니라 valve hysteresis state, dynamic flow state, effective area와 previous command에 의존한다. Simulator에는 policy가 관측하지 않는 valve state \(z,x_1,x_2\), pump pressure/crank phase와 0.1 s delay buffer가 존재한다. 따라서 같은 current pressure와 같은 action에서도 hidden valve state가 다르면 next pressure가 달라질 수 있다.

Matched-pulse 실험에서도 current pressure와 reserve를 약 1 kPa 이내로 맞췄지만 recent history가 다른 상태에 같은 pulse를 가했을 때 0.5 s response가 POS 4.28 kPa, NEG 6.59 kPa 차이 났다. Future-pressure predictor도 current → pressure history → pressure/action history 순서로 held-out error가 감소했다. 이는 history가 state aliasing을 줄인다는 근거이나 strict observability proof는 아니다.

### 3. History-only training progress 재검토

대상 모델은 260810_lib3_ablation_profile_hist_base_seed0이다. 따라서 현재 증거만으로 History-only를 “제어 실패”라고 부르지 않는다. 이 모델은 추적을 수행하지만, 1000-episode budget 안에서 더 낮은 오차 또는 낮은 진동의 해에 도달하지 못한 것으로 기술한다. History-only는 episode 701–800에서 combined RMSE 3.144로 가장 좋았지만, 801–900은 3.442, 901–1000은 3.480으로 개선되지 않았다.

episode 601–1000을 16-episode block으로 분석하면 combined-RMSE slope의 신뢰구간이 0을 포함했다. 따라서 현재 budget 안에서는 600 episode 이후 plateau이며 R50을 따라잡는 경향은 없다. 반면 R50은 episode 101–200에 RMSE 2.492에 도달하여 H(8.187)보다 빠르게 학습했다.

그러나 한 seed와 1000 episode 종료만으로 asymptotic failure를 주장할 수 없다. H와 R50을 2000 episode까지 continuation하고, 50 episode마다 같은 fixed evaluation bank로 paired 평가해야 한다. Primary outcome은 training return이 아니라 full-bank RMSE, direction agreement, HF ripple과 constraint violation으로 둔다. 즉 **tracking quality와 vibration/smoothness는 별도 축**이다.

### 4. \(\dot P\)의 inductive-bias 해석

NeurIPS 2023의 Char and Schneider는 partial observation history에 PID의 sum/difference 구조를 넣은 PIDE와 GPIDE를 제시했다. PIDE는 tracking error의 P/I/D를, GPIDE는 observation difference와 previous action/reward를 structured history encoder에 사용한다. 현재 \(\dot P\)도 history에서 계산 가능한 difference coordinate라는 점에서 같은 계열이다.

현재 plateau에서는 \(\dot r=0\)이므로

\[
\dot e=\dot r-\dot P=-\dot P.
\]

따라서 rate policy의 진동 감소는 PID D-term의 damping 해석과 일치한다. 실제 seed-0 평가에서 H 대비 R50은 plateau HF ripple을 POS 약 91%, NEG 약 75%, action HF RMS를 약 60% 감소시켰다. 다만 pressure history 내부 차분보다 현재 rate/50 입력이 약 100배 강조되므로 H/R50/R5000/zero4 비교와 동일 safety constraint가 필요하다.

### 5. Lambda discrepancy를 이용한 더 엄밀한 검사

NeurIPS 2024의 lambda discrepancy는 TD(0)와 TD(1)처럼 서로 다른 TD(\(\lambda\)) value estimate의 차이를 이용해 representation의 non-Markovness를 검사한다. Current pressure, pressure history, pressure/action history, H+rate와 privileged simulator state를 동일 frozen policy와 동일 trajectory에서 비교한다.

예상 판정은 다음과 같다.

- Current보다 H의 discrepancy가 작음: history가 value-relevant partial observability를 감소.
- Pressure history보다 pressure/action history가 작음: delayed command context가 필요.
- H와 H+rate의 exact information은 같음. Finite critic에서 H+rate가 더 작으면 information이 아니라 representation/conditioning bias.
- Privileged state는 estimator와 function-approximation error floor.

Lambda discrepancy는 future-pressure predictor를 대체하지 않는다. Predictor는 dynamics prediction, lambda discrepancy는 policy/reward의 value prediction, RL ablation은 closed-loop control을 각각 검사한다. 세 증거가 같은 방향일 때만 observation 설계를 강하게 주장한다.

### 6. 이번 주 결론과 수행 항목

현재 결론은 다음이다.

1. Current pressure는 hidden valve dynamics와 delay 때문에 구조적으로 non-Markov이다.
2. Pressure/action history는 approximate information state를 구성하기 위한 temporal information이다.
3. Rate와 reserve는 새 information이 아니라 physics-derived inductive bias다.
4. History-only는 현재 1000-episode budget에서 plateau했지만 asymptotic impossibility는 확인되지 않았다.
5. Rate의 가장 일관된 seed-0 효과는 final RMSE보다 빠른 학습, directionality와 oscillation damping이다.

이번 주 우선순위:

1. Trained H+rate actor의 first-layer folding으로 동일 history-only actor를 만들어 expressivity equivalence 확인.
2. H/R50/R5000/zero4를 동일 chamber constraint와 multi-seed로 비교.
3. Frozen-policy lambda-discrepancy diagnostic을 current/H/H+rate/privileged state에 적용.
4. H와 rate 모델을 continuation하면서 50-episode checkpoint와 fixed-bank evaluation 수행.
5. 이후에만 MPObs와 rate의 2×2 nominal/mismatch/real comparison을 진행.

---

## 13. 위 랩미팅 양식 빈칸용 초압축 문안

### Title

**Soft Actuator pneumatic RL의 history observation 및 pressure-rate inductive bias 검증**

### Recommendation

History는 hidden valve/delay state의 aliasing을 줄이는 정보로, \(\dot P\)와 \(P_{ch}-P_{act}\)는 새 정보가 아닌 physics-derived coordinate로 구분한다. History-only의 plateau와 rate 효과를 convergence control 및 lambda discrepancy로 재검증한다.

### Introduction

- 피드백: \(\dot P\)와 \(P_{ch}-P_{act}\)는 pressure history로 계산 가능하므로 History-only도 제어되어야 함.
- 이론적으로 맞음: augmented MLP의 derived-feature weight는 history first layer로 정확히 folding 가능.
- 따라서 질문을 “rate가 필수인가?”에서 “explicit difference가 finite SAC의 학습·damping·robustness를 개선하는가?”로 수정.
- Current pressure는 valve hysteresis/dynamic state, pump state와 0.1 s delay를 보지 못하므로 non-Markov.

### 이번 주 진행한 내용

- History-only seed-0 training curve 1000 episode 재분석.
- 601–1000 block slope와 reference-cycle-adjusted slope 계산.
- Current/history/action-history future-pressure predictor 및 matched-pulse 근거 정리.
- NeurIPS 2023 PIDE/GPIDE와 explicit \(\dot P\)의 inductive-bias 관계 정리.
- NeurIPS 2024 lambda discrepancy를 이용한 Markovness 진단 설계.

### 결과

- H는 701–800에서 최저 combined RMSE 3.144 후 901–1000에서 3.480으로 개선 없음.
- 601–1000 combined-RMSE slope의 신뢰구간이 0을 포함 → 관측 budget 내 plateau.
- R50은 101–200에 RMSE 2.492로 H(8.187)보다 빠르게 학습.
- H 대비 R50은 seed-0에서 pressure/action HF oscillation을 크게 감소시켰으나 약 100배 scale emphasis가 있음.
- 결론: History-only가 원리적으로 불가능한 것이 아니라 vanilla SAC의 derivative-direction 학습 실패 가능성. Multi-seed/scale/placebo/continuation 필요.

### 다음 단계

1. First-layer folding으로 H+rate와 동일한 History-only actor 구성.
2. H/R50/R5000/zero4 동일 constraint multi-seed.
3. Current/H/H+rate/privileged-state lambda discrepancy 비교.
4. H/R50 2000 episode continuation + 50-episode fixed-bank evaluation.

---

## 14. 교수님 피드백 반영: observer/POMDP와 reward를 분리한 재정의

### 14.1 POMDP는 “센서에 동특성이 있어야” 성립하는 개념이 아니다

표준 제어의 선형 근사에서는

\[
x_{t+1}=Ax_t+Bu_t+w_t,\qquad y_t=Cx_t+v_t
\]

로 쓴다. 여기서 MDP인지 여부는 **policy가 받는 (y_t)가 plant의 Markov state (x_t)를 충분히 나타내는가**로 판단한다. (C=I)이고 (x_t)의 모든 성분이 측정된다면 (y_t=x_t)이므로 observation도 Markov state다. 반대로 (C\)가 pressure 성분만 고르는 행렬이면, measurement equation이 정적이어도 (y_t\neq x_t)이고 POMDP일 수 있다. 센서 delay, filtering, noise는 그 불완전성을 더하는 요인일 뿐 POMDP의 필요조건이 아니다.

delay가 있을 때는 관측 버퍼까지 포함한 augmented plant state를

\[
\bar x_t=(x_t,y_{t-1},\ldots,y_{t-d})
\]

로 두면 전체 simulator는 다시 Markov이다. 하지만 policy가 그 중 일부 delayed pressure만 받으면 policy observation은 여전히 Markov state가 아니다.

### 14.2 우리 lib3에 적용하면 무엇이 (x), 무엇이 (y)인가

우리의 simulator full state는 최소한

\[
x_t=(P_{ch,+},P_{ch,-},P_{act,+},P_{act,-},\;
\{z_i,x_{1,i},x_{2,i},I_{\mathrm{prev},i},s_{\mathrm{prev},i}\}_{i=1}^6,
\text{pump/crank phase},\text{sensor-delay buffer},\ldots)
\]

를 포함한다. 반면 base policy의 현재 sensor observation은 delayed 4-pressure뿐이며, history policy도 이 pressure sequence와 과거 action을 받을 뿐 valve internal state나 pump/crank state를 직접 보지 못한다. 실제 valve model에서는 같은 pressure/action에서도 (x_1,x_2,z,I_{\mathrm{prev}})가 다르면 predicted flow와 다음 pressure가 달라진다. 따라서 **full simulator state 기준으로 (C\neq I)** 이며, base policy interface는 POMDP다.

다만 0.20 s pressure/action history를 “state observer”라고 단정하면 안 된다. 그것은 model-based Luenberger/Kalman observer가 아니라, 과거 (y,u)에서 hidden state의 일부를 추론하도록 SAC에 준 **finite-window history encoder**다. history가 approximate information state인지 여부는 held-out next-observation prediction, matched-pulse aliasing, lambda discrepancy와 privileged-state ablation으로 검증해야 한다. 충분한 history가 full state를 복원한다는 증명은 아직 없다.

### 14.3 NeurIPS 2023 PIDE/GPIDE와의 정확한 관계

Char와 Schneider는 POMDP를 “policy observation이 true state가 아닌 경우”로 넓게 정의하고, 특히 unmeasured state variable의 경우를 다룬다. 따라서 partial measurement라는 큰 틀에서는 우리와 같다. 그러나 논문의 기여는 PID의 error sum/difference를 history encoder의 inductive bias로 사용하는 것이며, (y=Cx)의 observability rank, pneumatic valve state estimator, 또는 우리 simulator의 delayed sensor model을 분석하지는 않는다.

그러므로 발표 문장은 다음처럼 제한한다.

> NeurIPS 2023은 partially observed control에서 PID-inspired history encoder가 유용할 수 있다는 architecture precedent다. 우리 시스템이 왜 partially observed인지는 별도로 full simulator state와 policy observation의 차이로 보이고, history가 충분한 observer인지는 실험적으로 검증한다.

또한 ​\(\dot P\)와 reserve는 현재 history/raw pressure에서 결정적으로 계산된다. 따라서 이들은 observer가 새 hidden state를 복원한 결과도, Markov information을 추가한 것도 아니다. finite MLP/SAC가 difference/headroom 좌표를 쓰기 쉽게 만든 coordinate/conditioning bias다.

### 14.4 Reward 문제: 현재 결과가 말하는 것과 말하지 않는 것

세 기존 ablation config의 reward는

\[
r_t=-0.3\left(\lvert P^{ref}_{act,+}-P_{act,+}\rvert+
\lvert P^{ref}_{act,-}-P_{act,-}\rvert\right)
\]

뿐이다. rate tracking, action difference, second action difference, valve conflict, chamber reserve/bound coefficient는 모두 0이다. 따라서 이 reward는 pressure ripple이 actuator pressure error를 키우는 범위에서는 **간접적으로** 진동을 벌하지만, 다음을 직접 구분하지 못한다.

- 같은 mean/MAE를 가지는 smooth trajectory와 high-frequency ripple,
- valve action chattering과 smooth action,
- reference가 움직이는 동안 필요한 pressure rate와 reference 주변의 불필요한 oscillation.

따라서 History-only의 training return/MAE가 좋아졌다고 “진동도 해결됐다”고 말할 수 없다. 반대로 History-only가 진동했다고 해서 POMDP 때문에 제어 불가능하다고도 말할 수 없다. 기존 objective가 그 정책을 충분히 배제하지 않았을 가능성이 실제로 있다.

### 14.5 해야 할 reward ablation: 2000 episode는 필요하지만 2×2로 한다

History-only만 reward를 바꾸어 2000 episode로 가면, 좋아졌을 때 observation 효과인지 objective 효과인지 분리할 수 없다. 다음 factorial design으로 간다.

| Observation | Base reward | Phase-aware smooth reward |
|---|---|---|
| H: pressure/action history | H-base | H-smooth |
| H+\(\dot P\): same history + explicit rate | HR-base | HR-smooth |

smooth reward는 raw \(\lvert\dot P\rvert\) penalty가 아니라 reference motion을 보존하는 rate-tracking term으로 둔다.

\[
r_t^{\mathrm{smooth}}
=-\lambda_r\sum_{c\in\{+,-\}}
\ell\!\left(\frac{\dot P_{act,c}-\dot r_c}{s_c}\right)
-\lambda_1\lVert u_t-u_{t-1}\rVert_2^2
-\lambda_2\lVert u_t-2u_{t-1}+u_{t-2}\rVert_2^2.
\]

이미 환경에는 첫 rate-error term, action-delta, second-action-delta가 구현되어 있고 기존 모델에서 coefficient만 0이었다. rate term은 plateau에서 \(\dot r=0\)이므로 pressure ripple을 직접 벌하지만, ramp에서는 필요한 pressure 변화까지 억제하지 않는다. \(\lambda\)는 seed-0 development sweep에서 사전에 고정한 뒤 평가 seed에는 재조정하지 않는다. chamber bound/termination은 네 조건 모두 동일하게 적용한다.

이 rate reward는 action을 고르는 시점에 policy에 \(\dot P\)를 직접 주는 것이 아니라, transition 뒤에 rate error를 scalar reward로 주는 **objective/supervision 변화**다. 따라서 H-smooth의 성공은 observation이 Markov해졌다는 증거가 아니며, 반드시 HR-smooth와의 같은-objective 비교가 필요하다.

권장 순서는 (i) H-smooth seed-0을 2000 episode까지 먼저 screening해 reward가 진동 축을 실제로 바꾸는지 확인하고, (ii) coefficient를 lock한 뒤 4 조건 × 3 seeds × 2000 episode를 실행하는 것이다. fixed reference bank에서 RMSE, plateau HF ripple/PSD, action HF RMS/total variation, overshoot/settling, constraint violation을 함께 보고 paired confidence interval을 계산한다.

해석은 명확하다.

- **H-smooth가 H-base보다 진동을 줄이고 HR-base 수준에 접근**: 기존 차이의 상당 부분은 objective/optimization 문제.
- **H-smooth도 HR-smooth보다 반복적으로 나쁨**: 같은 objective에서 explicit rate coordinate의 learning/conditioning contribution.
- **H-smooth와 HR-smooth가 모두 좋음**: rate observation의 “필수” 주장은 철회하고, rate는 sample-efficiency bias로만 주장.
- **둘 다 나쁨**: reward보다 hidden-state/observation 또는 actuator constraint 모델을 다시 확인.
