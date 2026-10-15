# Observer, POMDP, 그리고 우리 pneumatic system의 observability audit

## 결론 먼저

현재 시스템에 대해 바로 말할 수 있는 것은 다음 세 문장이다.

> 수식 렌더러가 없는 Markdown viewer에서는 LaTeX 명령이 그대로 보일 수
> 있습니다. 핵심 기호는 [plain 수식 요약](formula_reference_plain.md)에
> ASCII/Unicode로 정리되어 있습니다.

1. `lib3`와 `lib4`의 **instantaneous sensor vector**는 simulator의 augmented execution state 전체를 직접 측정하지 않는다.
2. 그러나 이것만으로 plant가 제어이론적으로 **unobservable**하거나, history policy도 여전히 control-POMDP라고 결론낼 수는 없다.
3. 진짜 연구 질문은 “어떤 길이의 pressure/action history가 control-relevant state를 충분히 reconstruct하는가?”이며, 이는 nonlinear observability/output-map test와 privileged-state prediction으로 검사해야 한다.

따라서 발표에서 “우리 시스템은 POMDP다”라고 단정하지 않는다. 아래처럼 말한다.

> 현재 sensor만으로는 full simulator state를 직접 측정하지 않는다. 다만 measured pressure/action history가 control-relevant state를 충분히 복원하는지는 observability analysis와 held-out prediction으로 판정한다. History가 충분하면 history policy는 approximate fully observed controller로 취급할 수 있고, explicit rate는 새 information이 아니라 learning bias다.

---

## 1. 세 개념을 엄밀히 분리

### 1.1 Full-state measurement

이산시간 plant를

\[
x_{t+1}=F(x_t,u_t),\qquad y_t=h(x_t)
\]

로 둔다. 여기서 (x_t)는 Markov transition을 결정하는 full (또는 augmented) state다.

- **full-state measured**: (y_t)가 (x_t)를 즉시 uniquely determine한다. 선형 local model이면 (y_t=Cx_t), (C=I)가 대표적인 경우다.
- **not full-state measured**: (h(x)=h(x'))인 (x\ne x')가 존재한다. 이것은 현재 measurement 하나만으로 state를 구별 못 한다는 뜻이다.

### 1.2 Control-theoretic observability

full-state measurement가 아니어도 known input/output history로 state를 복원할 수 있다. 선형 LTI model에서는

\[
\mathcal O=\begin{bmatrix}C\\CA\\\vdots\\CA^{n-1}\end{bmatrix},
\qquad \operatorname{rank}(\mathcal O)=n
\]

이면 observable이다. 따라서 (C\ne I\)는 **full-state measurement가 아님**을 뜻할 뿐, unobservable의 증명은 아니다.

### 1.3 RL의 partial observation

policy가 사용하는 observation을 (o_t)라고 하자. 아래가 성립하면 (o_t)는 Markov state다.

\[
p(o_{t+1},r_t\mid o_{0:t},u_{0:t})
=p(o_{t+1},r_t\mid o_t,u_t).
\]

instantaneous (o_t=y_t)가 이 조건을 못 만족할 수 있어도,

\[
H_t=(y_{t-k:t},u_{t-k:t-1})
\]

가 만족하면 history controller는 control-relevant하게 fully observed일 수 있다. 이것이 교수님의 “우리 시스템은 full observed 아닌가?”가 겨냥하는 가능성이다.

---

## 2. Source audit: lib3와 lib4는 반드시 분리한다

### 2.1 lib3: 현재 observation-ablation에 사용한 pressure plant

`lib3`의 RK ODE vector는 (X_{\rm DIM}=8)이다.

\[
x^{\rm ODE}_{\rm lib3}=
[t,P_{ch+},P_{ch-},\theta,P_{pis1},P_{pis2},P_{act+},P_{act-}]^\top.
\]

기본 sensor는 4 pressure다. 하지만 현재 source에는 ODE vector 밖에서 갱신되는 6개 valve의 runtime state도 있다.

\[
x^{\rm valve}_i=(z_i,x_{1,i},x_{2,i},I_{\rm prev,i},s_{\rm prev,i}),\quad i=1,\ldots,6.
\]

또한 Python wrapper는 delay가 0이 아니면 observation deque를 유지한다. 그러므로 executable simulator의 augmented state는

\[
X^{\rm aug}_{\rm lib3}=(x^{\rm ODE}_{\rm lib3},x^{\rm valve}_{1:6},b^{\rm delay})
\]

로 써야 한다. `X_DIM=8`만을 full state라고 쓰면 valve memory와 delay를 빠뜨린다.

관련 구현: [lib3 ODE dimension](../pneu_env/src/pneu_env/lib3/pneumatic_simulator.h), [valve runtime state와 dynamics](../pneu_env/src/pneu_env/lib3/pneumatic_CT.cpp), [sensor delay buffer](../pneu_env/src/pneu_env/sim3.py).

### 2.2 lib4: soft-actuator dynamics를 포함한 main plant 후보

`lib4`의 RK ODE vector는 (X_{\rm DIM}=10)이다.

\[
x^{\rm ODE}_{\rm lib4}=
[t,P_{ch+},P_{ch-},\theta,P_{pis1},P_{pis2},P_{act+},P_{act-},L,v]^\top.
\]

`lib4` sensor wrapper는 4 pressure, length (L), velocity (v)를 제공할 수 있다. 단, policy가 실제로 velocity까지 받는지는 해당 experiment config의 `include_displacement_velocity_obs`를 확인해야 한다. `lib4`에도 현재 source 기준으로 6개 valve runtime state와 optional delay buffer가 존재한다. 즉

\[
X^{\rm aug}_{\rm lib4}=(x^{\rm ODE}_{\rm lib4},x^{\rm valve}_{1:6},b^{\rm delay})
\]

가 source-level full execution state다.

중요한 정정: “lib4에는 (X_{\rm DIM}=10)만 있고 valve internal state가 없다”는 설명은 현재 repository source와 맞지 않는다. `ValveRuntimeState`와 `solenoid_valve()`의 (z,x_1,x_2) update가 실제로 구현돼 있다.

관련 구현: [lib4 state/valve 정의](../pneu_env/src/pneu_env/lib4/pneumatic_simulator.h), [soft actuator pressure equation 및 state update](../pneu_env/src/pneu_env/lib4/pneumatic_CT.cpp), [6-sensor/delay wrapper](../pneu_env/src/pneu_env/sim4.py), [policy observation assembly](../pneu_env/src/pneu_env/env4.py).

---

## 3. instantaneous sensor가 부족할 수 있는 수학적 이유

### 3.1 lib4의 variable-volume coupling

actuator positive pressure는 source model에서 다음 꼴이다.

\[
\dot P_{act+}=
\frac{-P_{act+}V_+'(L)v+RT\,\dot m_{act+}}{V_+(L)}.
\]

policy가 (L,v)를 보지 않는 config라면, 같은 four-pressure와 action에서도 (L) 또는 (v)가 다르면 \(\dot P_{act+}\)가 달라지는 즉시 counterexample이 된다.

policy가 (L,v)를 모두 보더라도 아직 자동으로 full observed는 아니다. 같은 measured

\[
y=(P_{ch+},P_{ch-},P_{act+},P_{act-},L,v)
\]

와 같은 action에서 valve (x_{1,i})가 다르면 valve mass flow가 달라진다.

\[
\dot m_i \propto \max(x_{1,i},0),
\]

이므로 \(\dot P\)와 다음 measurement가 달라질 수 있다. pump piston pressure와 crank phase도 chamber pressure dynamics에 들어간다. 이것은 **instantaneous sensor가 augmented simulator state를 directly reveal하지 않는다**는 source-level argument다.

### 3.2 그러나 이것은 unobservability proof가 아니다

위 hidden state는 recent (y,u) history에서 reconstruct될 수 있다. 즉

\[
(y_{t-k:t},u_{t-k:t-1})\mapsto\hat X_t
\]

가 가능한지를 아직 확인하지 않았다. History가 충분하면 H10/H20 policy를 approximate information-state controller로 부를 수 있다. 이때 explicit rate는 state를 새로 관측하는 것이 아니다.

\[
\dot P_t=\frac{P_t-P_{t-1}}{\Delta t}=L H_t
\quad\Rightarrow\quad
\sigma(H_t,\dot P_t)=\sigma(H_t).
\]

따라서 ​\(\dot P\)의 효과가 있더라도 “POMDP를 MDP로 만들었다”는 주장은 불가하다. 가능한 주장은 finite SAC/MLP가 trend coordinate를 더 쉽게 최적화했다는 inductive-bias/conditioning effect다.

---

## 4. 수학적으로 먼저 해야 할 observability analysis

### 4.1 Local nonlinear observability rank: model이 smooth한 구간에서

switching, clipping, contact를 고정한 local operating region에서는

\[
\dot x=f(x)+\sum_j g_j(x)u_j,\qquad y=h(x)
\]

의 Lie-derivative observability codistribution을 계산할 수 있다.

\[
\mathcal D_o(x)=\operatorname{span}\{dh,dL_fh,dL_f^2h,\ldots\}.
\]

\(\operatorname{rank}\mathcal D_o=n\)이면 그 operating point 주변에서 locally weakly observable이다. 단 valve direction switching, saturation, contact는 non-smooth/hybrid이므로 global rank proof를 주장하지 않는다. 각 mode와 plateau/ramp operating point에서 별도로 계산한다.

### 4.2 현재 project에 더 실용적인 finite-horizon output-map Jacobian

known action sequence (U=(u_0,\ldots,u_{N-1}))에 대해

\[
G_N(X_0;U)=
[h(X_0),h(X_1),\ldots,h(X_N)]
\]

를 정의한다. finite difference 또는 automatic differentiation으로

\[
J_N=\frac{\partial G_N}{\partial X_0}
\]

를 계산한다.

- hidden-state column이 거의 0: 그 hidden direction은 해당 input/length에서 output history로 구별되지 않음.
- \(J_N\)의 smallest singular value가 충분히 큼: local reconstruction이 잘 conditioned됨.
- window (N)을 늘려 singular spectrum이 포화되는 지점: 필요한 history length의 물리적 근거.

이 test는 `current`, H5, H10, H20 중 어느 history가 충분한지에 가장 직접적인 수학적 근거를 준다. 다만 full rank는 local/nominal claim이며 noise와 model mismatch 아래 real observability를 보장하지 않는다.

### 4.3 data-level corroboration: observability proof와 역할이 다름

동일 capacity predictor로 다음을 held-out trajectory에서 비교한다.

\[
\hat y_{t+\tau}=g_{\rm current}(y_t,u_t),
\qquad
\hat y_{t+\tau}=g_H(H_t,u_t),
\qquad
\hat y_{t+\tau}=g_{\rm priv}(X_t^{\rm aug},u_t).
\]

- current → H의 error 감소: instantaneous observation이 부족하다는 evidence.
- H → privileged의 gap: 현재 history가 아직 복원하지 못한 control-relevant hidden information의 upper-bound evidence.
- prediction이 좋다고 full observability를 증명하는 것은 아니다. 이는 chosen horizon/distribution에서의 empirical sufficiency다.

lambda discrepancy는 이와 별개로 “현재 reward/policy에서 value prediction이 얼마나 Markov-like한가”를 본다. observability rank test의 대체물이 아니다.

---

## 5. Reward는 반드시 별도 축으로 검사한다

기존 lib3 H/H+rate ablation의 base reward는 current actuator absolute tracking error뿐이다.

\[
r_t^{\rm base}=-0.3(|e_{+,t}|+|e_{-,t}|).
\]

따라서 trajectory ripple은 error를 키우는 범위에서만 간접 벌점을 받고 action chatter 또는 derivative mismatch를 독립적으로 벌하지 않는다. History-only가 oscillate한 원인은 (a) observation/history reconstruction, (b) finite optimization coordinate, (c) reward objective 중 하나 또는 조합일 수 있다.

reward를 바꾸는 것은 필요하지만 observability test의 대체가 아니다. 첫 rate reward는 raw \(|\dot P|\)가 아니라 phase-aware form을 쓴다.

\[
r_t^{\rm smooth}=r_t^{\rm base}
-\lambda_r\sum_c\ell\left(\frac{\dot P_{act,c}-\dot r_c}{s_c}\right)
-\lambda_1\lVert u_t-u_{t-1}\rVert^2
-\lambda_2\lVert u_t-2u_{t-1}+u_{t-2}\rVert^2.
\]

plateau에서 \(\dot r=0\)이라 pressure rate를 벌하고, ramp에서는 목표 변화에 필요한 rate를 보존한다. 이 reward는 action 시점에 policy에 rate를 직접 주는 것이 아니라 transition 뒤 scalar supervision을 바꾸는 것이므로 observation ablation과 분리해야 한다.

| Observation | Base reward | Smooth reward |
|---|---:|---:|
| H: pressure/action history | H-base | H-smooth |
| HR: H + explicit \(\dot P\) | HR-base | HR-smooth |

각 조건을 2000 episode까지, locked coefficient와 같은 chamber constraint/reference bank로 실행한다. H-smooth seed-0의 2000-episode run은 tuning/screening으로는 유용하지만 논문 결론은 2×2 × 최소 3 seeds로 낸다.

---

## 6. 실행 우선순위

1. **Source/config freeze**: 논문에 사용할 lib3/lib4 binary build hash, actual policy observation vector, delay/noise, action scaling, reference generator를 한 장으로 고정한다.
2. **Output-map observability audit**: lib4 main plant에서 (J_N) singular spectrum을 current/H5/H10/H20 window별, plateau/ramp별로 계산한다. lib3 결과는 pressure-only precursor로 분리한다.
3. **Empirical sufficiency**: current/H/H+rate/privileged predictor와 matched-history counterfactual을 같은 held-out bank에서 수행한다.
4. **Representation test**: H+rate actor의 first-layer folding으로 rate가 expressivity를 추가하지 않음을 확인한다.
5. **Reward test**: H-smooth seed-0 2000 episode screening 후, coefficient lock → 2×2 × 3 seeds × 2000 episodes.
6. **Primary outcome**: fixed-bank tracking RMSE, HF pressure PSD/ripple, action HF RMS/total variation, settling/overshoot, constraint violations. Training return은 보조 로그다.

이 순서가 필요한 이유는 분명하다. reward만 바꾸어 H가 안정화되면 “reward가 부족했다”는 결론은 가능하지만, history가 state를 충분히 reconstruct하는지 또는 explicit rate가 finite optimization을 돕는지는 알 수 없다. 반대로 observability/source audit 없이 POMDP를 전면 novelty로 내세우면 교수님 질문에 방어할 수 없다.
