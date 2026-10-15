# lib3 제어공학적 Observability 분석

## 결론 먼저

lib3 source와 한 비포화 operating point에서 numerical nonlinear
observability를 계산한 결과는 다음과 같다.

1. **현재 4-pressure snapshot은 25차원 smooth-core simulator state를 fully
   observable하게 만들지 않는다.** 순간 측정의 local rank는 4다.
2. **Pressure/action history에는 실제 state-disambiguation 정보가 있다.** 실제
   0.1 s measurement delay를 포함한 4-pressure output-map rank는 H1의 4에서
   H10의 19--20, H20의 20--21까지 증가했다.
3. 그래도 H20에서 25차원 smooth-core full rank에 도달하지 않았고 retained condition
   number도 약 5.7e4--8.3e4였다. 따라서 이 operating point에서 lib3는
   **완전히 observable하다고 결론낼 수 없고, 여러 방향이 weakly observable**하다.
4. Actuator pressure 2개만 사용하면 H20 rank가 12--13에 머문다. Chamber
   pressure 2개는 target이 아니더라도 pump/valve dynamics를 구분하는 데
   실질적으로 기여한다.
5. `[p, p_dot]`과 pressure H2는 rank와 row space가 동일했다. 따라서 `p_dot`은
   **current pressure-only observation보다는 더 많은 temporal information을
   드러내지만, 이미 pressure history가 있다면 새로운 information은 아니다.**

이 결과로 가장 정확하게 할 수 있는 주장은 다음이다.

> lib3에서는 현재 pressure만으로 full simulator state를 구별할 수 없으며,
> pressure/action history가 hidden dynamics의 구별 가능성을 크게 높인다.
> Explicit pressure rate는 two-sample history의 first-difference coordinate로,
> history-only policy가 차분을 내부에서 학습하는 부담을 줄이는 inductive bias다.

![lib3 finite-horizon output-map rank](../../../pneu_rl/exp/lib3_nonlinear_observability/rank_vs_history.png)

---

## 1. Observability와 POMDP는 다르지만 직접 연결된다

### 1.1 제어공학의 질문

Deterministic plant를 다음처럼 쓴다.

```text
x[k+1] = F(x[k], u[k])
y[k]   = h(x[k])
```

Observability는 known input과 output history로 서로 다른 initial state를
구별하거나 복원할 수 있는지를 묻는다.

```text
(y[k-L+1:k], u[k-L+1:k-1])  ->  x[k]
```

`O = [C; CA; ...; CA^(n-1)]` rank test는 LTI system용 공식이다. Observability
개념 자체가 선형시스템에만 한정되는 것은 아니다. Nonlinear system에서는
Lie-derivative rank condition 또는 finite-horizon output map의 Jacobian을 쓴다.
고전적인 nonlinear formulation은 Hermann과 Krener의
[Nonlinear Controllability and Observability](https://doi.org/10.1109/TAC.1977.1101601)에
정리돼 있다.

### 1.1.1 LTI observability matrix와 lib3 계산의 차이

LTI(discrete-time linear time-invariant) system은

```text
x[k+1] = A x[k] + B u[k]
y[k]   = C x[k]
```

처럼 하나의 고정된 `A`, `B`, `C`로 모든 operating point와 time step을 표현한다.
이때 미래 output을 전개하면

```text
y[k]     = C x[k] + known input term
y[k+1]   = C A x[k] + known input term
y[k+2]   = C A^2 x[k] + known input term
...
```

이고, initial state에 의존하는 부분만 모으면

```text
O_L = [ C       ]
      [ C A     ]
      [ C A^2   ]
      [ ...     ]
      [ C A^(L-1)]
```

가 된다. 따라서 `rank(O_L)=n`이면 모든 state 방향이 이 LTI 모델의 output
history에 locally 나타난다고 판정한다. 입력 `u`는 state를 직접 관측하는 행이
아니라 known input term과 state excitation을 만든다.

lib3에는 다음 이유로 하나의 `A`, `C`를 그대로 사용할 수 없다.

1. pressure equation과 valve flow equation이 곱셈, 압력차, volume 변화와 함께
   nonlinear이다.
2. `z`, `q`, `q_dot`이 valve마다 별도로 update되고, `branch`와 saturation에
   따라 dynamics가 바뀐다.
3. simulator는 RK integration, valve runtime update, measurement delay를 포함한다.
4. state sensitivity는 operating point와 실제로 가한 input trajectory에 따라
   달라진다.

따라서 lib3에서는 `A`와 `C`를 임의로 선형화해서 하나의 global observability
결론을 내리지 않고, 실제 simulator transition을 그대로 사용해 **해당 operating
point와 해당 known input sequence에서의 local finite-horizon observability**를
계산했다. 이는 LTI matrix test의 대체 개념이라기보다, LTI observability matrix가
표현하던 “초기 state perturbation이 output history에 나타나는 방향의 수”를
nonlinear plant에 맞게 수치적으로 일반화한 것이다.

### 1.2 RL/POMDP의 질문

POMDP는 stochastic decision problem의 정보 구조다. Environment에는 Markov
state `s[k]`가 있지만 policy에는 observation `o[k]`만 주어진다. 같은
observation을 만드는 두 hidden state에서 동일 action 뒤의 future distribution이
다르면 `o[k]`는 Markov state가 아니다.

```text
s1 != s2,  O(s1) = O(s2),
but P(s[k+1] | s1, a) != P(s[k+1] | s2, a)
```

Control observability가 성립해 input/output history로 state를 복원할 수 있다면,
그 history 또는 observer estimate는 RL의 information state 역할을 할 수 있다.
이 연결은 measured input/output로 output-feedback RL을 구성한
[Lewis & Vamvoudakis (2011)](https://pubmed.ncbi.nlm.nih.gov/20350860/)와 noisy
finite batch를 분석한
[Yaghmaie et al. (IEEE TAC, 2024)](https://doi.org/10.1109/TAC.2024.3385680)에서도
명시적으로 다뤄진다.

따라서 두 개념의 관계는 다음 한 문장으로 정리한다.

> Observability는 state reconstruction의 plant/sensor 조건이고, POMDP는 policy가
> 받은 정보가 decision-sufficient한지를 표현하는 framework다. Observable한
> input/output history는 POMDP의 state ambiguity를 줄이는 한 방법이다.

---

## 2. lib3의 실제 state, input, measurement

### 2.1 Input과 pressure measurement

lib3의 valve input은 6개다.

```text
u = [u_ch+, u_ch-, u_act+in, u_act+out, u_act-in, u_act-out] ∈ R^6
```

Simulator와 real controller가 공통으로 사용하는 pressure measurement는 4개다.

```text
p = [P_ch+, P_ch-, P_act+, P_act-] ∈ R^4
```

현재 lib3 RL reward가 직접 tracking하는 reference는 actuator pressure 2개다.
그러나 policy는 chamber pressure 2개도 관측한다. 따라서 “6-input/4-pressure
measurement/2-tracking-target task”가 현재 코드에 가장 정확한 표현이다.

Input 수가 output/state 수보다 많은 것은 observability 문제가 아니다. 이는
over-actuated control allocation 문제다. Observability는 state dynamics,
sensor map, 그리고 주어진 input trajectory가 state를 구별시키는가의 문제다.

### 2.2 RK state만 보면 8차원이지만, 그것이 full execution state는 아니다

lib3 RK vector는 다음 8개다.

```text
x_RK = [time, P_ch+, P_ch-, theta, P_pump1, P_pump2, P_act+, P_act-]
```

하지만 각 valve에는 RK vector 밖에서 update되는 runtime state가 5개씩 있다.

```text
xi_i = [z_i, q_i, q_dot_i, I_prev_i, branch_i],  i = 1,...,6
```

- `z`: Bouc-Wen 형태 hysteresis memory
- `q`, `q_dot`: second-order valve-flow lag state (`x1`, `x2` in source)
- `I_prev`: 이전 coil current
- `branch`: opening/closing branch memory

따라서 clock과 Python delay buffer를 제외한 C++ execution state는 37차원이다.

```text
7 ODE states excluding time + 6 valves × 5 runtime states = 37
```

여기서 중요한 점은 `I_prev`와 `branch`의 성격이 서로 다르다는 것이다.

- `I_prev`는 마지막 coil current를 저장하는 **실수형 memory state**다.
- `branch`는 current가 증가하는지 감소하는지를 나타내는 **이산 mode**이며,
  source의 실제 변수명은 `state_prev`다. 값은 0 또는 1이다.

따라서 전체 scalar 저장값 37개는 엄밀하게 말하면

```text
31 continuous coordinates
  = 7 ODE states excluding time
  + 6 valves × (z, q, q_dot, I_prev)
6 binary branch modes
```

로 나뉜다. 이번 Jacobian 분석은 `I_prev`와 6개 branch mode를 고정한 뒤, 다음
25개 **smooth-core coordinate**만 perturb한 조건부 분석이다.

```text
x_core = [4 pressures, theta, 2 pump pressures,
           6 × (z, q, q_dot)] ∈ R^25
```

그러므로 표의 rank 25는 전체 37개 scalar execution state가 observable하다는
뜻이 아니다. 정확히는 “고정된 `I_prev`와 고정된 branch mode에서 25개 smooth
state direction이 얼마나 pressure history에 나타나는가”를 측정한 것이다. `I_prev`
까지 perturb하는 31차원 continuous analysis와 branch별 mode comparison은 별도의
추가 실험으로 남아 있다.

### 2.3 Hybrid branch란 무엇인가?

lib3의 각 valve는 현재 command의 절대값만 사용하는 것이 아니라, 직전 current와
비교해 command가 증가하는지 감소하는지를 기억한다. source의 로직은 다음과 같다.

```text
if current > I_prev + eps:
    branch = 1       # current increasing / opening-side branch
elif current < I_prev - eps:
    branch = 0       # current decreasing / closing-side branch
else:
    branch = previous branch
```

이후 branch는 단순한 label이 아니라 실제 dynamics 식을 선택한다.

```text
dI = |current - I_prev| * (2*branch - 1)
dz = A*dI - beta*|dI|*z - gamma*dI*|z|

wn    = wn_up    if branch == 1 else wn_down
zeta  = zeta_up  if branch == 1 else zeta_down
```

즉 branch가 1인 경우와 0인 경우에 hysteresis update의 방향과 flow-lag의
natural frequency/damping이 달라진다. 수학적으로는

```text
x[k+1] = F_branch(x[k], u[k])
branch[k+1] = Gamma(branch[k], I_prev[k], u[k])
```

형태의 hybrid system이다. 한 branch 안에서는 `F_0` 또는 `F_1`을 smooth하게
미분할 수 있지만, `current - I_prev = ±eps` 경계를 넘는 순간에는 다음 mode가
바뀌어 하나의 smooth Jacobian으로 전체를 표현할 수 없다.

따라서 이번 분석에서는 branch를 고정하고 각 mode 안에서만 finite-difference
Jacobian을 계산했다. branch 전환을 포함한 전체 시스템의 observability를 보이려면
각 branch 조합을 따로 계산하거나, discrete-mode observer/hybrid automaton 분석을
추가해야 한다.

Source 근거:

- [state와 valve runtime 정의](../../../pneu_env/src/pneu_env/lib3/pneumatic_simulator.h)
- [pressure/flow dynamics](../../../pneu_env/src/pneu_env/lib3/pneumatic_CT.cpp)
- [RK integration과 pressure output](../../../pneu_env/src/pneu_env/lib3/pneumatic_simulator.cpp)
- [0.1 s sensor delay wrapper](../../../pneu_env/src/pneu_env/sim3.py)

### 2.4 Reduced pressure model이 fully observed라는 말과 모순되지 않는다

제어용 reduced model을 처음부터 `x_ctrl = p ∈ R^4`로 정의하면 `y = x_ctrl`이고
`C = I4`다. 이 모델은 정의상 fully measured/observable하다.

그러나 이 결론은 다음 closure 가정을 필요로 한다.

```text
p[k+1] ≈ F_ctrl(p[k], u[k])
```

lib3 source에서는 같은 4 pressure와 command라도 pump pressure, crank phase,
valve flow/hysteresis state가 다르면 다음 pressure가 달라질 수 있다. 따라서
`x_ctrl=p`는 유용한 reduced approximation일 수는 있지만, full simulator의 exact
Markov state와 동일하지 않다.

### 2.5 MCK 비유로 보는 lib3의 state, observation, input

MCK(mass--spring--damper)에서는

```text
m x_ddot + c x_dot + k x = u
```

이고, 최소 state는

```text
x_MCK = [position, velocity]
```

이다. 위치만 알아서는 같은 위치에서 오른쪽으로 움직이는지 왼쪽으로 움직이는지
구분할 수 없기 때문이다. `u`는 질량에 가하는 외력이므로 state가 아니라 input이다.

lib3도 같은 기준으로 읽으면 된다.

| MCK 개념 | lib3 대응 | 역할 |
|---|---|---|
| position/velocity | pressure, pump/phase, valve `z/q/q_dot` | plant가 현재 기억하는 physical state |
| position sensor | 4 pressure measurement | 실제 policy가 받는 observation |
| external force `u` | 6 valve commands | plant transition을 바꾸는 known input |
| unmeasured velocity | valve flow/hysteresis/memory | pressure만으로는 직접 보이지 않는 hidden state |

lib3의 개념적 관계는 다음과 같다.

```text
physical state x_phys
        │
        ├── measurement map h(·) ──> y = [P_ch+, P_ch-, P_act+, P_act-]
        │                              (0.1 s delayed in the environment)
        │
controller/policy ──> u = 6 valve commands ──> pneumatic plant transition
```

여기서 full simulator state는 대략

```text
x_phys = [4 pressures, theta, 2 pump pressures,
           6 × (z, q, q_dot, I_prev, branch)]
```

이고, policy가 직접 보는 값은

```text
y = [P_ch+, P_ch-, P_act+, P_act-]
```

뿐이다. 따라서 `y`와 `x_phys`는 같은 개념이 아니다.

#### `P_dot`은 MCK의 velocity와 비슷하지만, physical state로 자동 추가되지는 않는다

공압 chamber의 pressure derivative는 이상기체식에서 다음처럼 계산된다.

```text
P_dot = (R T / V) * m_dot - (P / V) * V_dot
```

여기서 mass flow는 밸브 command, pressure difference, hysteresis와 flow-lag에
의존한다.

```text
m_dot = g(P, u, z, q)
```

따라서 full physical state와 현재 input을 알고 있다면 `P_dot`은

```text
P_dot = f(P, z, q, u, theta, pump pressure)
```

로 계산되는 종속 변수다. MCK에서

```text
x_ddot = -(k/m)x - (c/m)x_dot + (1/m)u
```

를 state에 별도로 넣지 않는 것과 같다.

다만 현재 pressure만 observation으로 받는 RL policy에는 `P_dot`이 유용할 수 있다.
왜냐하면 policy가 직접 알 수 없는 pressure 변화 방향을 history로부터 계산해 주기
때문이다. 이때 `P_dot`은 새로운 physical state가 아니라 **관측 history의 derived
feature**다.

```text
P_dot[k] ≈ (P[k] - P[k-1]) / dt
```

그러므로 `[P[k], P_dot[k]]`는 `[P[k-1], P[k]]`와 invertible한 좌표변환이고,
pressure H2보다 새로운 physical information을 추가하지 않는다. 그러나 H1의
current pressure-only observation과 비교하면 velocity-like temporal information을
추가한다.

#### `q`, `z`, `I_prev`, `branch`는 왜 state인가?

이 변수들은 현재 pressure와 current command만으로 즉시 계산되는 값이 아니라,
과거 command와 valve trajectory를 기억한다. 같은 pressure와 같은 현재 command라도
이 값들이 다르면 다음 flow와 pressure가 달라진다.

```text
same [pressure, current command]
but different [z, q, I_prev, branch]
        └──> different next flow and next pressure
```

따라서 이들은 plant의 Markov state를 구성하는 memory다. 반대로 current command
`u[k]`는 plant에 가해지는 input이며, 그 자체를 state vector에 추가하지 않는다.

#### MCK와 lib3의 observability 대응

MCK에서 위치 `x`만 측정하고 속도 `x_dot`을 측정하지 않아도 위치 history로 속도를
추정할 수 있듯이, lib3에서도 pressure만 측정하고 `q/z`를 직접 측정하지 않아도
pressure/action history로 hidden valve dynamics를 추정할 수 있다.

```text
MCK:   position history  -> velocity-like hidden state
lib3:  pressure/action history -> q, z, branch-related hidden dynamics
```

차이는 lib3가 단순한 smooth second-order system이 아니라 hysteresis와 branch
switching을 포함한다는 점이다. 그래서 history가 hidden state를 완전히 복원한다고
바로 가정하지 않고, branch를 고정한 local Jacobian과 finite-horizon rank를 먼저
계산했다.

---

## 3. 왜 p_dot이 observer 관점에서 의미가 있는가

### 3.1 최소 pressure-flow 예제

한 pressure channel의 local state를 `x=[p,q]`라고 두자. `q`는 valve mass flow나
flow-lag를 대표하는 hidden state다.

```text
[p[k+1]]   [a11 a12] [p[k]]   [b1] u[k]
[q[k+1]] = [a21 a22] [q[k]] + [b2]

y[k] = [1 0] x[k] = p[k]
```

LTI observability matrix는 다음과 같다.

```text
O = [ C  ] = [  1    0  ]
    [ CA ]   [ a11  a12 ]
```

`a12 != 0`이면 rank는 2다. 즉 pressure history 두 시점으로 hidden flow direction을
구별할 수 있다. 연속시간 표현에서는 다음과 같다.

```text
p_dot = a11 p + a12 q + b1 u
q = (p_dot - a11 p - b1 u) / a12
```

따라서 independently measured한 `p_dot`이 있다면 `[p,p_dot,u]`에서 hidden
`q`를 algebraically reconstruct할 수 있다. 이 의미에서 `p_dot`은 첫 번째
Lie derivative/output-dynamics coordinate다.

### 3.2 하지만 현재 p_dot은 새 sensor가 아니라 H2의 좌표변환이다

현재 구현은 pressure sensor 두 sample의 finite difference다.

```text
p_dot[k] = (p[k] - p[k-1]) / dt
```

두 표현을 놓으면

```text
H2 = [p[k-1], p[k]]
z  = [p[k], p_dot[k]] = T H2

T = [ 0       I    ]
    [-I/dt   I/dt  ]
```

`dt != 0`이면 `T`는 invertible이다. 그러므로

```text
rank(dz/dx) = rank(dH2/dx)
sigma(H2, p_dot) = sigma(H2)
```

즉 다음 두 문장이 동시에 참이다.

- Current pressure H1과 비교하면 `[p,p_dot]`은 temporal information을 추가해
  더 많은 hidden direction을 구별할 수 있다.
- Pressure history H2 이상과 비교하면 `p_dot`은 information/rank를 추가하지 않고
  같은 정보를 dynamics-aligned coordinate로 재표현한다.

이번 numerical calculation에서도 all-pressure 기준 H2와 `[p,p_dot]`은 둘 다
rank 8이었고, row-space projector 차이는 `2e-15`--`3e-14`였다. 이는 numerical
precision 수준의 동일성이다.

### 3.3 “Observer 한 단계 + controller 한 단계 = 2차”는 정식 표현이 아니다

Observer dynamics와 controller dynamics를 결합하면 closed-loop augmented system의
order가 커질 수는 있다. 그러나 observability index와 controller order를 단순히
더해 “총 2차”라고 정의하지는 않는다.

더 정확한 표현은 다음이다.

> Pressure-only controller는 hidden flow/trend를 history 또는 observer dynamics로
> 추정해야 한다. Explicit rate coordinate는 그 추정에서 가장 먼저 필요한
> output derivative를 직접 제공하므로 policy의 representation/optimization
> burden과 reaction latency를 줄일 수 있다. 단 finite-difference rate는 한 sample의
> memory가 필요하며 physical observer order를 없애는 것은 아니다.

`p_ddot`도 가능하지만 세 pressure sample의 second-difference일 뿐이다.

```text
p_ddot[k] ≈ (p[k] - 2p[k-1] + p[k-2]) / dt^2
```

이는 H3에서 계산 가능하고, hidden state가 second output derivative에서 처음
나타나는 relative-degree 구조일 때만 유용하다. Noise amplification은 `p_dot`보다
더 크므로 feature를 추가한다고 항상 성능이나 observability conditioning이
좋아지지는 않는다.

---

## 4. Numerical nonlinear observability 방법

### 4.1 Nonlinear finite-horizon output map

Known action sequence `U = {u[0], ..., u[L-2]}`와 initial state `x0`에 대해
simulator transition을 반복한다.

```text
x[0] = x0
x[i+1] = F(x[i], u[i])
y[i]   = h(x[i])
```

그 결과의 finite-horizon output map을 다음처럼 정의했다.

```text
G_L(x0; U) = [y0, y1, ..., y(L-1)]
J_L        = dG_L / dx0
```

LTI의 `O_L`가 `C A^i`를 행으로 쌓는 것과 달리, 여기서는 `F`를 실제로 rollout한
뒤 `G_L`을 만들고 그 Jacobian을 계산한다. 따라서 `J_L`의 column `j`는 initial
state coordinate `x_j`를 조금 바꿨을 때 전체 pressure history가 얼마나 바뀌는지를
나타낸다.

### 4.2 Jacobian을 실제로 만든 방법

분석 script의 계산 순서는 다음과 같다.

1. simulator를 reset하고 10-step multisine preparation을 수행한다.
2. preparation 뒤의 augmented simulator snapshot을 `x0`으로 저장한다.
3. 각 continuous state coordinate `x_j`에 대해 characteristic scale `s_j`를
   정하고, `delta_j = 2e-4 s_j`를 만든다.
4. `x0 + delta_j e_j`와 `x0 - delta_j e_j`를 각각 복원한다.
5. 동일한 known input sequence를 두 trajectory에 넣고 output history를 rollout한다.
6. central finite difference로 Jacobian의 `j`번째 column을 계산한다.

```text
J_L[:,j] ≈
  (G_L(x0 + delta_j e_j; U) - G_L(x0 - delta_j e_j; U))
  / (2 delta_j)
```

여기서 `U`는 constant input 또는 time-varying multisine input이다. 같은 `U`를
plus/minus trajectory에 적용하는 이유는 input의 효과와 initial-state sensitivity를
분리하기 위해서다.

### 4.3 delay와 output-history window

실제 lib3 observation에는 0.1 s delay가 있으므로, 50 Hz에서 5 samples delay를
반영했다. `delay=d`, history window `L`인 경우 실제로 사용하는 map은

```text
G_(d,L)(x0; U) = [y[d], y[d+1], ..., y[d+L-1]]
```

이다. 따라서 H1은 한 시점 pressure snapshot, H2는 두 시점 pressure history,
H10은 10개 delayed pressure sample을 의미한다. 이것이 LTI observability matrix의
row block 개수를 늘리는 것과 대응한다.

#### H1/H2/H5/H10/H20를 읽는 법

여기서 `H`는 episode horizon이나 학습 episode 수가 아니라 **output-history sample
개수**다. lib3 simulator frequency가 50 Hz이므로 sample interval은

```text
dt = 1 / 50 = 0.02 s
```

이다. 실제 delay를 5 samples로 설정한 표에서는 다음과 같이 읽는다.

| 표기 | 사용한 delayed pressure samples | output row 수 (4 pressure 기준) | 첫 sample과 마지막 sample 사이 |
|---|---|---:|---:|
| H1 | `[p[k+5]]` | 4 | 0 s |
| H2 | `[p[k+5], p[k+6]]` | 8 | 0.02 s |
| H5 | `[p[k+5], ..., p[k+9]]` | 20 | 0.08 s |
| H10 | `[p[k+5], ..., p[k+14]]` | 40 | 0.18 s |
| H20 | `[p[k+5], ..., p[k+24]]` | 80 | 0.38 s |

즉 `H20`은 20개의 pressure vector를 사용한다는 뜻이지 20초를 사용한다는
뜻이 아니다. 첫 output이 실제 state보다 0.1 s 늦게 시작하는 것은 별도의
measurement delay이고, H 값은 delay 이후 몇 개의 sample을 쌓는지를 나타낸다.

4개의 pressure를 사용하므로 H`L`의 output map은 최대 `4L`개의 row를 갖는다.
하지만 row 수가 곧 rank는 아니다. 같은 hidden-state 방향에 대한 row가 서로
종속적이면 rank는 `4L`보다 작을 수 있다. 예를 들어 pressure-only H10은 40개
output row를 사용하지만 numerical effective rank는 약 19--21에 머문다.

### 4.4 Scaling, SVD, effective rank

Pressure(kPa), phase, hysteresis, flow는 단위와 크기가 서로 다르므로 Jacobian을
그대로 SVD하면 큰 단위의 state가 rank를 지배할 수 있다. 그래서 state/output
characteristic scale을 적용했다.

```text
J_scaled = S_y^(-1) J_L S_x
J_scaled = U Sigma V^T
rank_eff = number of singular values > tolerance
```

기존 pressure-only 분석은 상대 tolerance `1e-6 × sigma_max`를 사용했고, 후보
state 비교 분석은 scale 이후 고정 absolute tolerance `1e-7`을 사용했다. 따라서
표의 rank는 symbolic exact rank가 아니라 **numerical effective local rank**이다.
이 기준은 singular spectrum과 함께 봐야 하며, rank만으로 conditioning이나 RL
성능을 직접 보장하지 않는다.

### 4.5 rank의 의미와 한계

- `rank(J_L)=r`라는 것은 이 operating point와 이 input sequence에서 작은 initial
  state perturbation 중 `r`개의 독립적인 방향이 output history에 나타난다는 뜻이다.
- rank가 25이면 `I_prev`와 branch가 고정된 25차원 smooth-core state에 대해 local
  inverse가 가능하다는 강한 증거지만, global observability theorem은 아니다.
- rank가 23이면 최소 두 개 방향은 해당 finite horizon과 tolerance에서 output에
  충분히 나타나지 않았다는 뜻이다.
- `branch`는 discrete mode이고 `I_prev`는 이번 분석에서 고정한 continuous memory다.
  둘을 25차원 Jacobian에 포함하지 않았으므로 전체 execution state 37개 scalar가
  항상 observable하다는 결론은 내릴 수 없다.
- branch transition이나 saturation을 가로지르면 `F`가 미분 가능하지 않을 수
  있다. 그래서 preparation이 clamp에 닿은 구간은 main table에서 제외하고,
  branch별 separate analysis가 필요하다.

기존 numerical rank 표에 사용한 분석 설정은 이 절의 마지막 표에 정리했다.

### 4.6 Exact symbolic rank를 계산할 수 있는가?

#### MCK에서는 가능하다

이상적인 LTI MCK 모델은

```text
x_dot = A x + B u
y     = C x
```

에서 `A`, `C`가 고정된 유한 행렬이다. 따라서

```text
O_L = [C; C A; C A^2; ...]
```

를 기호식으로 만들고, 어떤 `n × n` minor의 determinant가 0이 아닌지 확인할 수
있다. 예를 들어 position만 측정하는 2차 MCK에서

```text
x = [position, velocity]
C = [1, 0]
A = [0, 1; -k/m, -c/m]
```

이면

```text
O_2 = [1, 0]
      [0, 1]
```

이므로 `rank(O_2)=2`가 parameter 값과 무관하게 exact하게 성립한다.

#### 현재 lib3 전체 모델에서는 바로 적용하기 어렵다

lib3는 다음 연산을 포함한다.

- `if/else`로 선택되는 opening/decreasing branch
- `max`, `min`, `clamp` saturation
- `exp`, `logaddexp`, compressible-flow 함수
- valve 내부 sub-step RK/Euler update
- pressure와 volume이 결합된 nonlinear transition
- `I_prev`와 current threshold에 따른 mode switching

따라서 finite-horizon map은 하나의 전역 analytic expression이 아니라 mode와
saturation region마다 다른 piecewise map이다.

```text
G_L(x; U) = G_L^(mode, saturation region)(x; U)
```

각 region 안에서는 symbolic Jacobian을 만들 수 있지만, H20처럼 여러 step을
전개하면 식의 크기가 급격히 증가한다. branch 경계를 통과하면 미분식 자체가
바뀌므로 하나의 global symbolic Jacobian과 global exact rank는 정의되지 않는다.

또한 실제 parameter가 decimal floating-point 값이고 `exp`나 `log`가 포함되므로,
어떤 minor가 수학적으로 정확히 0인지 판단하는 것도 일반적인 정수/유리수
행렬처럼 간단하지 않다. 매우 작은 값이 진짜 0인지, 단지 conditioning이 나쁜
것인지 구분해야 한다.

#### 가능한 exact 또는 semi-exact 분석

다음 조건을 고정하면 제한적인 symbolic 분석은 가능하다.

1. 특정 branch 조합을 고정한다.
2. force/area/flow saturation이 inactive하다고 가정한다.
3. `I_prev`와 mode를 고정하거나, 필요하면 `I_prev`를 symbolic continuous state로
   포함한다.
4. H1 또는 H2처럼 짧은 horizon을 선택한다.
5. 실제 simulator 식을 smooth symbolic function으로 다시 작성한다.

그 뒤 candidate minor determinant를 계산해 해당 mode cell에서 nonzero인지 확인할
수 있다. 이 결과는 “그 branch와 operating region에서 generic rank가 얼마인가”를
말해 주지만, 모든 branch와 saturation을 포함하는 global theorem은 아니다.

#### 현재 분석을 더 엄밀하게 만드는 방법

현재의 numerical effective rank를 다음 방식으로 보강할 수 있다.

- finite-difference step과 rank tolerance sweep
- arbitrary-precision SVD
- interval arithmetic으로 smallest singular value가 0에서 떨어져 있는지 검증
- branch별 rank와 branch-switching trajectory 분리
- `I_prev`까지 포함한 31차원 continuous Jacobian 계산
- symbolic/automatic differentiation으로 finite-difference 결과 교차검증

따라서 현재 논문에서 “exact symbolic rank가 21이다”라고 쓰면 안 된다. 정확한
표현은 다음이다.

> lib3의 특정 operating point, fixed hybrid mode, fixed input sequence에서 실제
> simulator output-map Jacobian의 numerical effective rank를 계산했다. Exact global
> symbolic rank는 piecewise nonlinear/hybrid 구조 때문에 별도의 mode-restricted
> analysis 없이는 정의하거나 계산하기 어렵다.

| 항목 | 설정 |
|---|---|
| Perturbed state coordinates | 25 smooth-core states |
| Held valve memory | 6 continuous `I_prev` + 6 binary branch modes |
| Sensor sets | actuator pressure 2개 / 전체 pressure 4개 |
| Output window | H1, H2, H5, H10, H20 |
| Known future input | constant / time-varying multisine |
| Measurement delay | 0 / 실제 설정 5 samples = 0.1 s |
| Finite difference | central, characteristic scale의 `2e-4` |
| Operating point | 10-step excitation 후의 non-saturated state |

120-step preparation에서는 `ch_neg.z`가 source clamp `+/-1e6`에 닿았다. 이
구간은 smooth Jacobian 가정이 성립하지 않으므로 main rank table에서 제외했다.
이는 valve model이 hybrid/saturated operating region을 가진다는 별도의 중요한
한계이며, 각 branch/saturation mode를 따로 검사해야 한다.

분석 코드는 production `.so`를 덮어쓰지 않고 별도의 analysis library를 만든다.

- [numerical observability script](../../../pneu_rl/scripts/analyze_lib3_nonlinear_observability.py)
- [state snapshot C ABI](../../../pneu_env/src/pneu_env/lib3/pneumatic_simulator.cpp)
- [raw summary CSV](../../../pneu_rl/exp/lib3_nonlinear_observability/observability_summary.csv)
- [p_dot/H2 equivalence CSV](../../../pneu_rl/exp/lib3_nonlinear_observability/rate_history_equivalence.csv)

재현 명령:

```bash
conda run -n env_1 python pneu_rl/scripts/analyze_lib3_nonlinear_observability.py
```

---

## 5. Numerical 결과

### 5.1 실제 0.1 s delay 조건의 effective local rank

| Known input | Sensors | H1 | H2 | H5 | H10 | H20 |
|---|---|---:|---:|---:|---:|---:|
| Constant | actuator 2 | 2 | 4 | 8 | 11 | 12 |
| Time-varying | actuator 2 | 2 | 4 | 8 | 10 | 13 |
| Constant | all pressure 4 | 4 | 8 | 16 | 20 | 21 |
| Time-varying | all pressure 4 | 4 | 8 | 17 | 19 | 20 |

점선 25가 `I_prev`와 branch를 고정한 smooth-core state의 full rank다. 이 operating point와
threshold에서는 어떤 조건도 full rank에 도달하지 않았다.

주요 해석은 다음과 같다.

- H1에서 H2로 갈 때 all-pressure rank가 4에서 8로 증가한다. Pressure trend에
  hidden dynamics 정보가 있다는 직접적인 수치 근거다.
- H5/H10까지 history를 늘리면 rank가 크게 증가한다. 현재 History SAC가 단순히
  network input을 늘린 것이 아니라 state ambiguity를 완화하는 dynamic
  output-feedback 구조라는 근거다.
- H10 이후 증가량은 작아지며 H20에서도 full rank가 아니다. 현재 10-step history가
  충분한 exact information state라고 증명되지 않는다.
- Chamber pressure를 제외하면 rank가 크게 낮아진다. Chamber pressure는 reward의
  직접 target이 아니더라도 control authority와 hidden pump/valve state를
  distinguish하는 measurement다.
- 이 한 sequence에서는 time-varying input이 rank를 항상 높이지 않았다.
  Nonlinear/hybrid observability는 단순히 “입력을 많이 흔들면 무조건 증가”하는
  것이 아니며, operating mode와 excitation design에 의존한다.

### 5.2 Rank보다 conditioning이 더 큰 문제다

![lib3 observability singular spectrum](../../../pneu_rl/exp/lib3_nonlinear_observability/singular_spectrum_h20.png)

Actual-delay/all-pressure/H20에서 retained condition number는 다음과 같다.

| Known input | Effective rank | Retained condition number |
|---|---:|---:|
| Constant | 21 / 25 smooth-core | `8.31e4` |
| Time-varying | 20 / 25 smooth-core | `5.70e4` |

즉 일부 state direction은 이론적으로 output에 흔적을 남기더라도 압력 noise,
model mismatch, finite precision 아래에서 복원하기 어렵다. `act_pos_in.z`는 이
operating point에서 H20 pressure sensitivity가 사실상 0이었고,
`act_pos_out.z`, `ch_neg.z`도 매우 약했다. 이는 global unobservability proof가
아니라 **해당 valve mode/operating point에서의 weak or zero local sensitivity**다.

따라서 binary한 “observable/unobservable” 한 단어보다 다음 세 수치를 함께
보고해야 한다.

```text
effective rank + smallest retained singular value + condition number
```

---

## 6. RL observation에 대한 판정

### 6.1 Current pressure-only SAC

Full lib3 execution state를 environment state로 보면 current 4-pressure observation은
`I_prev`와 branch를 고정한 25개 smooth-core direction 중 4개만 순간적으로 구별한다.
5-step delay까지 있으므로
current policy observation은 current physical state와도 일치하지 않는다.

따라서 full simulator를 기준으로 current-pressure SAC는 POMDP setting으로 보는
것이 타당하다.

### 6.2 History SAC

Pressure/action history는 hidden dynamics를 구별하는 정보를 실제로 늘린다. 그러나
H10 numerical rank가 19--20/25이고 6개의 `I_prev`와 6개의 branch mode도 rank
계산에서 고정했으므로,
현재 결과로 H10을 exact MDP state 또는 exact belief state라고 부를 수는 없다.

가장 정확한 표현은 다음이다.

> History SAC는 known input/output history를 이용해 control-relevant hidden state를
> 근사 복원하는 finite-memory output-feedback policy이며, partial observability를
> 완화하지만 exact MDP 전환을 보장하지 않는다.

### 6.3 History SAC + rate

Rate는 history로부터 결정론적으로 계산된다. 따라서 H10에 rate를 추가해도
information set이나 theoretical observability rank는 늘지 않는다. 그럼에도
learning이 빨라질 수 있는 이유는 다음과 같다.

- first difference를 actor/critic이 별도로 학습할 필요가 없음
- pressure level과 motion direction/speed가 좌표상 분리됨
- tracking error가 같은 두 상태에서 approaching/retreating phase를 쉽게 구분
- finite data와 imperfect optimization에서 더 좋은 conditioning 제공 가능

이 해석은 history encoding에 summation/differencing bias를 넣은
[Char & Schneider, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ba1c5356d9164bb64c446a4b690226b0-Abstract-Conference.html)과
방향이 맞는다. 다만 이번 분석은 GPIDE architecture 재현이 아니라 lib3의
pressure-rate coordinate에 대한 system-specific observability/representation
분석이다.

---

## 7. 교수님께 보고할 문장

> Observability는 선형시스템에만 있는 개념은 아니고, 선형에서는 observability
> matrix를, lib3 같은 nonlinear/hybrid system에서는 output-history Jacobian을
> 사용합니다. lib3의 4 pressure를 reduced control state로 정의하면 C=I라 fully
> measured입니다. 하지만 실제 simulator는 pump pressure, crank phase, valve
> hysteresis와 second-order flow state를 추가로 가지므로 4 pressure snapshot은
> full state가 아닙니다.
>
> 실제 source를 기준으로 `I_prev`와 hybrid mode를 고정한 25차원 smooth-core state의 numerical
> observability를 계산했습니다. 0.1초 delay 조건에서 4-pressure rank는 H1 4,
> H2 8, H5 16--17, H10 19--20, H20 20--21로 증가했습니다. 따라서 history가
> hidden dynamics를 distinguish하는 것은 확인됐지만, 현재 H10이 full state를
> 완전히 복원한다고 말할 수는 없습니다.
>
> p_dot은 current pressure-only와 비교하면 첫 temporal derivative를 제공해 state
> ambiguity를 줄입니다. 그러나 finite difference p_dot은 H2와 invertible한
> 좌표변환이라 기존 history보다 새로운 information은 아닙니다. 그래서 우리의
> RL 결과는 observability를 새로 만든 것보다, observable history information을
> control-aligned coordinate로 주어 finite-budget learning을 빠르게 한 inductive
> bias로 해석하는 것이 맞습니다.

---

## 8. 다음 수학/수치 검증

당장 추가 RL training 없이 다음 분석을 진행할 수 있다.

1. **Operating-point sweep**: low/high chamber reserve, actuator plateau/ramp,
   opening/closing branch별로 같은 singular spectrum을 계산한다.
2. **Numerical robustness**: finite-difference size와 rank threshold sweep으로
   rank plateau를 확인한다.
3. **Excitation design**: random multisine이 아니라 state-sensitivity 또는 minimum
   singular value를 최대화하는 input sequence를 찾는다.
4. **Reduced observer test**: full 25-state reconstruction보다 control-relevant
   latent state인 net mass flow/pressure trend를 EKF, moving-horizon estimator,
   supervised predictor로 복원 가능한지 평가한다.
5. **Real-data corroboration**: 같은 `(p,u)` neighborhood에서 history가 다른
   sample들의 next-pressure distribution을 비교해 reduced Markov closure를
   검증한다.
6. **p_dot vs filtered p_dot**: finite difference의 noise amplification을 고려해
   raw rate, low-pass rate, learned difference를 동일 정보 조건에서 비교한다.

밸브 내부 후보 state(`z`, `q`, `q_dot`), pump/phase state, 현재·이전 command의
역할을 같은 output-map 기준으로 비교한 확장 분석은
[lib3 observation-set observability analysis](./lib3_observation_set_observability_analysis.md)에
정리했다.

논문에서는 global observability theorem보다 다음 결론이 더 방어 가능하다.

> Source-level nonlinear analysis shows that temporal pressure/action information
> materially increases local state distinguishability in the valve-dynamic
> pneumatic plant. An explicit pressure-rate coordinate does not add information
> beyond history, but provides a dynamics-aligned inductive bias that improves
> finite-budget SAC learning.
