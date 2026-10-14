# lib3 내부 state 후보와 Observability 분석

## 결론 먼저

이번 분석은 기존의 `4-pressure` 관측에 simulator 내부의 밸브 state를 직접
노출했을 때, finite-horizon nonlinear output map의 local rank가 어떻게 달라지는지
확인한 것이다. 결과는 다음처럼 해석해야 한다.

1. `z`(hysteresis), `q`(flow lag), `q_dot`(flow-lag rate)를 직접 보여주면
   pressure-only보다 hidden state 방향을 더 많이 구별할 수 있다.
2. 특히 `q` 또는 `q_dot` 하나만 추가해도 H2에서 rank가 8에서 19로 증가하고,
   `q+q_dot`를 함께 주면 H2에서 20--24까지 증가한다. 이는 밸브 내부 flow
   dynamics가 pressure evolution과 직접 연결된다는 수식적 증거다.
3. `z`와 simulator 내부 `q`, `q_dot`는 real policy가 직접 받는 값과 동일하지
   않다. 하지만 real system에서 valve flow/mass-flow를 측정할 수 있다면, 그
   measurement는 deployable observation 후보가 된다. 단 real flow sensor 값은
   simulator 내부 `q=x1`와 반드시 같은 값이 아니므로 sensor map, unit, sign,
   delay, filtering을 맞춰야 한다.
4. 현재 command `u_k`는 plant state가 아니라 known input이다. 고정된 입력열을
   state perturbation에 대해 미분하면 0이므로, `u_k`를 output에 덧붙여도
   observability rank는 증가하지 않는다.
5. 반면 이전 command `u_{k-1}`와 action history는 valve branch/hysteresis를
   해석하는 데 필요하다. 이것은 새로운 plant-state sensor가 아니라 hybrid mode를
   조건화하는 information history다. 실제 policy에는 넣을 수 있지만, rank 표에서
   state dimension을 늘린 것으로 세면 안 된다.

따라서 교수님께는 다음처럼 말하는 것이 가장 정확하다.

> 6개의 valve command는 state가 아니라 known input이다. simulator 내부 `z`, `q`,
> `q_dot`을 추가한 결과는 hidden flow dynamics의 upper bound를 보여준다. 한편
> real system에서는 valve flow/mass-flow를 실제로 측정할 수 있으므로, simulator의
> privileged `q`를 그대로 복사하는 대신 실제 sensor map을 반영한 `flow-aware`
> observation을 별도 검증할 수 있다. 이때 pressure-rate는 history에서 유도되는
> coordinate이고, flow measurement는 추가적인 physical output이다.

분석 산출물:

- [rank summary CSV](../../../pneu_rl/exp/lib3_observability_observation_sets/observation_set_rank_summary.csv)
- [rank heatmap](../../../pneu_rl/exp/lib3_observability_observation_sets/observation_set_rank_heatmap.png)
- [singular spectrum](../../../pneu_rl/exp/lib3_observability_observation_sets/observation_set_singular_spectrum_h20.png)
- [분석 script](../../../pneu_rl/scripts/analyze_lib3_observability_observation_sets.py)

---

## 1. State, input, observation을 먼저 분리해야 한다

lib3의 plant를 hybrid nonlinear discrete-time system으로 쓰면

```text
x[k+1] = F(x[k], u[k], m[k])
y[k]   = h(x[k])
```

이다. 여기서

- `x[k]`: simulator가 적분·업데이트하는 physical state
- `u[k] ∈ R^6`: policy가 보내는 6개 valve command
- `m[k]`: opening/closing branch 같은 discrete hybrid mode
- `y[k] ∈ R^4`: 측정 pressure

현재 task는 **6-control / 4-pressure measurement / 2-actuator-pressure target**로
부르는 것이 정확하다. 4개 pressure 중 chamber pressure 2개는 tracking target은
아니지만 plant dynamics를 구분하는 데 사용된다.

### 1.1 현재 command는 state가 아니다

현재 action을 output에 함께 쓴다고 하자.

```text
y_aug[k] = [h(x[k]), u[k]]
```

known command sequence를 고정하고 initial state에 대한 Jacobian을 계산하면

```text
∂y_aug/∂x = [ ∂h/∂x ]
             [    0   ]
```

이다. 따라서 command row는 local state observability rank를 증가시키지 않는다.
입력은 `F(x,u)`를 통해 이후 pressure output을 변화시키므로 observability를
**excite**할 수는 있지만, 센서가 state를 직접 측정하는 것과는 다른 역할이다.

다음 예외는 구분해야 한다.

- 명령값 `u_k`: known input, plant state 아님
- 이전 명령 `u_{k-1}, u_{k-2}`: history feature이며 hybrid mode 해석에 도움
- valve spool/current filter의 실제 내부 변수: actuator dynamics를 모델링한다면
  plant state
- policy가 출력한 action과 실제 valve에 도달한 action이 다르고 그 차이를
  actuator state로 모델링한다면, 그 actuator lag state는 state에 포함해야 함

즉 `u`를 state vector에 억지로 추가하는 것이 아니라, `u`가 actuator internal
dynamics를 거쳐 state를 만든다면 그 **내부 lag state**를 state로 정의한다.

---

## 2. lib3 execution state와 후보 변수

현재 C++ simulator의 execution state는 clock을 제외하면 31개의 continuous
coordinate와 6개의 binary branch mode를 포함한다. 이번 output-set rank 계산에서
perturb한 smooth-core state는 다음 25개다.

```text
x_core = [P_ch+, P_ch-, theta, P_pump1, P_pump2, P_act+, P_act-,
           6 × (z_i, q_i, q_dot_i)] ∈ R^25
```

각 valve에는 아래 hybrid/runtime memory도 있다.

```text
xi_i = [z_i, q_i, q_dot_i, I_prev_i, branch_i]
```

### 2.1 `z`: valve hysteresis state

`z_i`는 coil input의 방향과 과거 trajectory에 따라 valve opening/closing
response가 달라지게 하는 memory state다. 같은 pressure와 같은 현재 command라도
`z_i`가 다르면 effective flow가 달라질 수 있다.

예를 들어 local pressure equation이

```text
P_dot = (R T / V) (m_dot_in - m_dot_out) - (P/V) V_dot
m_dot_i = G_i(u_i, P_i, z_i, q_i)
```

라면

```text
∂P_dot/∂z_i = (R T / V) ∂m_dot_i/∂z_i
```

가 일반적으로 0이 아니다. 따라서 `z_i`는 pressure future를 바꾸는 hidden
direction이다. `z`를 직접 측정하면 그 memory ambiguity를 줄일 수 있다.

주의할 점은 source의 calibrated hysteresis가 일부 operating point에서 매우 큰
값 또는 clamp에 접근할 수 있다는 것이다. 이 경우 smooth Jacobian rank는
“global hysteresis observability”가 아니라 해당 local branch에서의 수치 결과다.

### 2.2 `q`: valve flow lag state

lib3는 valve flow를 즉시 algebraic하게 계산하지 않고 second-order lag state를
업데이트한다. `q_i`는 그 lag의 position/flow-like state이고 최종 mass flow에
직접 영향을 준다.

```text
q_dot_i      = v_i
v_dot_i      = f_i(q_i, v_i, u_i, P_i, z_i)
m_dot_i      = g_i(q_i, P_i, z_i)
```

따라서

```text
P_dot = f_P(P, q, z, u)
```

의 `q` 민감도는 보통 크다. `q`를 직접 output으로 추가한 결과가 `z`보다 빠르게
rank를 높이는 것은 이 physical connection과 일치한다.

단, simulator의 `q=x1`와 real flow sensor의 측정값을 동일시하면 안 된다. real
sensor는 일반적으로 다음과 같은 output으로 모델링한다.

```text
f_meas[k] = h_flow(x[k], u[k]) + sensor_noise
```

이번 real system의 측정 flow는 정확히 6개 channel이다.

```text
f_meas = [
  f_pump→ch+, f_pump→ch-,
  f_ch+→act+, f_ch-→act-,
  f_act+→atm, f_act-→atm
] ∈ R^6
```

이는 각각 `pump → chamber` 2개, `chamber → actuator` 2개,
`actuator → atmosphere` 2개다. 실제 센서의 unit, polarity, sample delay,
low-pass filter를 simulator observation에도 동일하게 적용해야 한다. 이 조건이
맞으면 `f_meas`는 simulator-only privileged feature가 아니라 real에 배치 가능한
physical output이다.

### 2.3 `q_dot`: flow-rate state

`q_dot_i`는 flow lag가 증가하는지 감소하는지를 나타낸다. 이는 pressure의 다음
변화 방향을 예측하는 데 유리하다.

```text
q[k+1] ≈ q[k] + dt q_dot[k]
P[k+1] ≈ P[k] + dt f_P(P[k], q[k], z[k], u[k])
```

따라서 `q_dot`은 transient와 damping/phase 정보를 직접 드러내는 후보다. 하지만
`q_dot` 자체는 noisy derivative 성격이므로, rank가 증가해도 RL 성능이 자동으로
증가한다는 뜻은 아니다. 센서 noise와 normalization을 포함한 control 실험이
별도로 필요하다.

### 2.4 `I_prev`, `branch`: continuous memory와 discrete branch

두 변수는 같은 종류가 아니다. `I_prev`는 실수형 last-current memory이고,
`branch`는 source의 `state_prev`에 해당하는 0/1 discrete mode다.

- `I_prev`: previous coil current 또는 command filtering memory
- `branch`: valve current가 증가하는지 감소하는지를 나타내는 discrete branch

source에서는 다음 조건으로 branch를 갱신한다.

```text
current > I_prev + eps  -> branch = 1
current < I_prev - eps  -> branch = 0
그 사이                 -> 이전 branch 유지
```

branch에 따라 hysteresis의 `dI` 부호와 flow-lag의 `wn`, `zeta`가 바뀐다. 따라서
한 branch 안에서는 smooth map `F_0` 또는 `F_1`을 사용할 수 있지만, threshold를
넘는 순간 dynamics mode가 바뀐다. 이번 rank는 branch와 `I_prev`를 고정한 25차원
smooth-core slice의 결과다. 엄밀히 말해 전체 scalar execution state는 31개의
continuous coordinate와 6개의 binary mode로 구성되며, `I_prev`까지 perturb하는
31차원 분석은 별도 실험이다.

이 값들은 같은 `(p,u)`에서 서로 다른 future를 만드는 원인이 될 수 있으므로,
policy 관점에서는 action history가 중요하다. 그러나 `branch`는 ordinary Jacobian
rank에 나타나지 않는 discrete ambiguity이므로, branch별로 separate rank를
계산하거나 hybrid automaton/observer 분석을 해야 한다.

---

## 3. 후보 output set과 계산 방법

이 문서의 rank는 LTI observability matrix를 직접 계산한 결과가 아니다. lib3에는
operating point마다 달라지는 nonlinear/hybrid transition이 있으므로, 실제
simulator를 rollout해 만든 finite-horizon output map의 state Jacobian을 SVD한
결과다. LTI의 `O=[C;CA;...]`와 lib3의 `J_L`이 어떻게 대응하고, finite difference와
delay/window를 어떻게 적용했는지는
[lib3 nonlinear observability analysis 문서](./lib3_nonlinear_observability_analysis.md)에
상세히 적었다. 특히 MCK의 `position/velocity`와 lib3의 `pressure/valve memory`가
어떻게 대응하는지는 그 문서의 MCK analogy 절을 참고하면 된다.

각 output set에 대해 동일한 operating point, 동일한 known input sequence, 동일한
central finite difference를 사용했다. 0.1 s delay는 50 Hz simulator에서 5 samples로
반영했다. 각 candidate output map을

```text
H_L(x0; u) = [ y[delay], y[delay+1], ..., y[delay+L-1] ]
```

여기서 `H1`, `H2`, `H5`, `H10`, `H20`은 각각 `L=1,2,5,10,20`인 output-history
window다. 50 Hz에서 sample interval은 0.02 s이고, 표의 delay=5는 첫 output이
0.1 s 뒤에 시작한다는 뜻이다. 예를 들어 H10은 10개의 delayed pressure vector,
즉 4-pressure output 기준 40개 row를 사용한다. H10이라는 표기는 10초나 10개
episode를 의미하지 않는다.

이 map과 smooth-core 25차원 initial state에 대한 Jacobian

```text
J_L = ∂H_L / ∂x0
```

을 계산했다. State/output characteristic scale로 정규화한 뒤 singular value가
`1e-7`보다 큰 개수를 effective local rank로 기록했다. 이는 exact symbolic rank가
아니라 finite-difference와 scaling에 의존하는 **numerical effective rank**이다.

비교한 output set은 다음과 같다.

| Output set | 직접 노출하는 값 | 해석 |
|---|---|---|
| `pressure_4` | 4 pressures | 현재 deployable baseline |
| `pressure_pump_phase` | pressure + pump pressure + `theta` | simulator internal ODE state |
| `pressure_valve_z` | pressure + 6 hysteresis `z` | valve memory privileged output |
| `pressure_valve_q` | pressure + 6 flow `q` | flow lag privileged output |
| `pressure_valve_qdot` | pressure + 6 flow-rate `q_dot` | transient privileged output |
| `pressure_valve_flow_pair` | pressure + `q` + `q_dot` | flow dynamics privileged output |
| `pressure_all_continuous` | all 25 smooth-core states | upper-bound diagnostic |
| `pressure_plus_current_action` | pressure + current 6-command input | known-input diagnostic |

### 3.1 0.1 s delay에서의 effective rank

아래는 `delay=5` samples 결과다. `constant`는 마지막 preparation command를
유지했고, `time_varying`은 multisine input을 사용했다.

#### Constant input

| output set | H1 | H2 | H5 | H10 | H20 |
|---|---:|---:|---:|---:|---:|
| 4 pressures | 4 | 8 | 18 | 23 | 23 |
| pressure + pump + phase | 7 | 13 | 19 | 23 | 23 |
| pressure + hysteresis `z` | 10 | 14 | 23 | 25 | 25 |
| pressure + flow `q` | 10 | 19 | 24 | 24 | 24 |
| pressure + flow-rate `q_dot` | 10 | 19 | 24 | 24 | 24 |
| pressure + `q` + `q_dot` | 16 | 24 | 24 | 24 | 24 |
| 25 smooth-core states | 25 | 25 | 25 | 25 | 25 |

`pressure_plus_current_action`은 H1/H2/H5/H10/H20에서 `4/8/18/23/24`였다.
이는 추가한 6개 command row의 initial-state Jacobian이 0이어서 pressure-only와
동일한 rank가 나온 것이다.

#### Time-varying input

| output set | H1 | H2 | H5 | H10 | H20 |
|---|---:|---:|---:|---:|---:|
| 4 pressures | 4 | 8 | 18 | 21 | 22 |
| pressure + pump + phase | 7 | 13 | 19 | 21 | 22 |
| pressure + hysteresis `z` | 9 | 13 | 22 | 22 | 22 |
| pressure + flow `q` | 10 | 19 | 23 | 24 | 24 |
| pressure + flow-rate `q_dot` | 10 | 19 | 23 | 24 | 24 |
| pressure + `q` + `q_dot` | 16 | 20 | 23 | 24 | 24 |
| 25 smooth-core states | 24 | 24 | 24 | 24 | 24 |

`pressure_plus_current_action`은 time-varying case에서도 `4/8/18/22/22`로
pressure-only와 동일했다. 다만 time-varying command sequence 자체는 plant를
excite하므로 rank와 conditioning을 바꿀 수 있다. 이는 command를 state로 추가한
효과가 아니라 **input excitation을 바꾼 효과**다.

이 표의 핵심은 절대적인 rank 숫자 자체보다 **어떤 내부 state를 노출했을 때
pressure-only의 missing direction이 줄어드는지**다.

- H1에서 `q` 또는 `q_dot`를 추가하면 4 → 10이 된다. 한 시점에서도 valve
  transient coordinate를 직접 보여주기 때문이다.
- H2에서 `q`/`q_dot`는 8 → 19, `q+q_dot`는 8 → 20--24로 올라간다.
- `z`도 rank를 증가시키지만, time-varying case에서는 hysteresis branch와
  operating-point sensitivity 때문에 flow state보다 증가폭이 안정적이지 않다.
- pump pressure와 phase는 4-pressure보다 정보를 추가하지만, valve flow
  coordinate만큼 빠르게 full continuous state를 분리하지는 못한다.
- smooth-core output이 24가 되는 time-varying case는 “25개 state가 항상
  globally observable”이라는 뜻이 아니다. 해당 local trajectory에서 한 방향이
  singular tolerance 아래였다는 뜻이며, operating point/branch를 바꾸면 달라질
  수 있다.

### 3.2 기존 `q` rank와 실제 6-channel flow rank의 차이

현재 표의 `pressure_valve_q`와 `pressure_valve_flow_pair`는 simulator 내부
`x1=q`, `x2=q_dot`를 직접 노출한 **state-coordinate experiment**다. Real sensor를
반영하는 실험은 다음 output map으로 다시 정의해야 한다.

```text
y_flow[k] = [p[k], f_meas[k]]
f_meas[k] = h_flow(x[k], u[k]) + noise[k]
```

여기서 `f_meas ∈ R^6`은 `pump→chamber` 2개, `chamber→actuator` 2개,
`actuator→atmosphere` 2개를 포함한다. Flow output은 현재 command에 직접적인
feedthrough가 있을 수 있지만, command 자체를 state로 추가하는 것과는 다르다.
Jacobian에는 다음처럼 flow sensor의 state sensitivity가 들어간다.

```text
∂[p, f_meas] / ∂x = [ ∂p/∂x       ]
                     [ ∂h_flow/∂x ]
```

따라서 기존 `q` rank 결과를 real flow sensor rank라고 그대로 인용하면 안 된다.
현재 lib3 C++의 `mass_flowrate` 배열은 simulator 내부 진단을 위해 10개 항을
보관한다. 이 10개 배열을 앞에서부터 잘라서 real의 6개 센서라고 간주하면 안
된다. 실제 실험에서는 배관 위치에 맞는 `pump→chamber` 2개,
`chamber→actuator` 2개, `actuator→atmosphere` 2개를 명시적으로 매핑하고,
각 채널의 sign과 단위를 calibration해야 한다.
다음 분석에서는 six-channel flow sensor의 gain, delay, filtering, noise model을
동일하게 적용한 `pressure + measured flow` output map을 추가해야 한다.

---

## 4. 제어 명령과 action history를 어디에 넣는가

### 4.1 현재 명령 `u_k`

현재 command는 다음처럼 모델에 들어간다.

```text
x[k+1] = F(x[k], u[k])
```

그러므로 state vector를

```text
[x[k], u[k]]
```

로 늘리는 것은 일반적으로 올바른 state-space 정의가 아니다. `u[k]`가 이미
controller에 알려져 있고, 다음 transition을 계산하기 위한 input이기 때문이다.

다만 output history와 함께 command sequence를 조건화하는 것은 필요하다.

```text
H_L(x0; u[0:L-1])
```

여기서 입력은 state를 직접 관측하는 센서가 아니지만, 서로 다른 hidden state가
다른 pressure response를 보이도록 excitation을 제공한다. 따라서 “command가
observability에 영향을 준다”와 “command가 state dimension이다”는 구분해야 한다.

### 4.2 이전 command `u_{k-1}`

밸브 branch가 이전 current/direction에 의존한다면 같은 current `u_k`라도
`u_{k-1}`가 다르면 opening/closing branch가 달라질 수 있다. 따라서

```text
policy input = [pressure history, action history]
```

는 POMDP의 ambiguity를 완화할 수 있다. 하지만 이것은 직접 state sensor를
추가한 것이 아니라 hidden mode를 추정할 단서를 제공한 것이다.

정확한 표현은 다음이다.

> Action history는 hybrid plant의 mode-conditioned information state를 근사한다.
> finite action history가 모든 branch를 정확히 복원한다고 단정할 수는 없지만,
> current action만 사용하는 것보다 branch ambiguity를 줄인다.

---

## 5. 이 결과가 real observation 설계에 주는 것

### 직접 넣기 어려운 privileged candidate

- `z`: hysteresis latent state라 직접 sensor가 없다.
- simulator `q`, `q_dot`: real flow sensor와 동일한 state coordinate가 아니며,
  `q_dot`은 특히 직접 측정값이 아니다.
- `P_pump1`, `P_pump2`: pump-side sensor가 없으면 simulator-only다.
- `theta`: encoder/clock가 reset과 동기화되어 알려진다면 hidden state가 아니다.

### real에서 사용 가능한 policy observation 후보

1. 기존 4-pressure와 짧은 pressure/action history
2. 실제 측정 flow `f_meas ∈ R^6`와 pressure/action history
3. `p_dot` 또는 filtered pressure-rate
4. 이전 action/current와 action difference
5. 필요할 때만 observer가 추정한 valve mode 또는 confidence

`p_dot`은 H2의 invertible coordinate라 rank를 새로 만드는 feature는 아니다.
그럼에도 policy가 finite data에서 history differencing을 직접 학습하는 부담을
줄이고, reference change와 damping direction을 즉시 제공하는 representation bias로
사용할 수 있다. 반대로 직접 측정한 flow `f_meas`는 pressure와 중복되지 않는
추가 physical output이 될 수 있다. 다만 기존 `q`/`q_dot` rank 표는 simulator
state coordinate를 노출한 upper bound이고, real flow sensor를 사용한 rank는
`h_flow(x,u)`를 넣어 다시 계산해야 한다.

### 유량 측정 반영 후의 RL 실험 계획

#### 이미 완료된 비교

`history SAC`와 `history + p_dot SAC`의 matched-reward 비교는 이미 수행했다.
[3-way learning screen](./history_rate_inductive_bias_screen.md)에 정리된 것처럼,
SAC / History SAC / History SAC + rate를 동일 lib3 plant, base reward, 512-step
episode, 1,000 episode, seed 0에서 비교했다. 따라서 이 비교를 같은 조건으로
다시 반복하는 것은 현재 flow 연구의 1차 우선순위가 아니다. HR 결과는 새 실험의
reference checkpoint/baseline으로 사용한다.

#### 새로 할 최소 flow screen

reward, reference, episode length를 고정하고 다음 네 arm만 비교한다.

| Arm | Observation | 목적 |
|---|---|---|
| H | 기존 history-action SAC | 기존 baseline 재사용 |
| HR | H + explicit pressure-rate | 이미 완료된 inductive-bias baseline |
| HF | H + 6-channel measured flow `f_meas` | flow measurement의 추가 physical information |
| HFR | H + 6-channel measured flow + pressure-rate | flow와 rate의 complementary effect |

screen은 우선 seed 0, 1,000 episode로 실행한다. `f_meas`는 simulator의 `z/q`
privileged state가 아니라 위 topology와 동일한 6개 flow output으로 구성해야 한다.
현재 `include_valve_diagnostics`는 `z/q/mdot` 18개를 함께 넣는 privileged 묶음이므로,
flow 실험에서는 이를 그대로 사용하지 말고 **6개 flow만 반환하는 전용 observation
mode**를 사용해야 한다. sensor delay, filtering, noise, normalization도 real
logging protocol과 맞춘다.

#### 추가 진단 arm

HF가 H보다 확실히 좋을 때만 다음 arm을 추가한다.

| Arm | Observation | 답하는 질문 |
|---|---|---|
| PF | 현재 4-pressure + current flow만 | flow가 history 없이도 Markov ambiguity를 줄이는가? |
| HF+reserve | H + flow + chamber reserve | flow와 chamber headroom의 중복/보완 관계 |

PF는 primary result가 아니라 flow의 state-sufficiency를 확인하는 진단이다.
HFR이 HR보다 개선되지 않으면 `p_dot`을 추가로 주장하지 않고, flow 자체의
physical output 효과만 보고한다.

#### 평가 지표와 후속 순서

각 arm에서 return만 비교하지 않고 다음을 함께 저장한다.

- combined actuator-pressure RMSE/MAE와 2 kPa success episode
- pressure oscillation RMS 및 high-frequency ratio
- action delta와 valve command switching rate
- measured-flow lag/peak/error 및 pressure-flow correlation
- held-out past reference와 OOD reference 성능

seed-0 screen에서 HF 또는 HFR이 finalist가 되면 그 arm과 H/HR만 seed 1,2를
추가해 mean ± std를 계산한다. 그 뒤 동일 policy를 과거 reference와 OOD reference,
마지막으로 real closed-loop에서 검증한다. 이렇게 하면 이미 끝난 `history →
history+rate` 실험을 반복하지 않고, 이번 논문의 새 축을 **measurable flow
observation이 sim-to-real 및 hidden flow dynamics를 얼마나 줄이는가**로 분리할 수
있다.

---

## 6. 교수님께 설명할 짧은 버전

> lib3를 6-input/4-pressure-output nonlinear hybrid plant로 분해했습니다. 4개의
> pressure만 센싱하면 reduced pressure model에서는 fully measured이지만, simulator가
> 실제로 적분하는 pump pressure, phase, valve hysteresis, flow-lag state까지 포함하면
> current pressure snapshot은 full physical state가 아닙니다. simulator 내부 `z`, `q`,
> `q_dot`을 추가한 rank 결과는 hidden flow dynamics의 upper bound를 보여줍니다.
> 한편 real에서는 `pump→chamber` 2개, `chamber→actuator` 2개,
> `actuator→atmosphere` 2개로 총 6개의 flow를 실제 측정할 수 있으므로, 동일한
> sensor map을 반영한 flow-aware observation을 별도 검증할 수 있습니다. 현재 command는
> known input이고, 이전 command는 valve branch를 조건화하는 information history로
> 분리했습니다.

---

## 재현 명령

```bash
/home/rise/miniconda3/envs/env_1/bin/python \
  pneu_rl/scripts/analyze_lib3_observability_observation_sets.py
```

현재 계산은 production dynamics를 변경하지 않고 analysis-only snapshot/setter
ABI를 사용한다. `rank_absolute_tolerance_after_scaling=1e-7`인 numerical effective
rank이므로, 논문에는 tolerance와 operating point, input excitation, branch를
반드시 함께 적어야 한다.
