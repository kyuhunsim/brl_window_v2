# Pneumatic RL 실험 정리

## Exploration, Ours 비교, lib3 4-control, Ablation 및 Reference 설계

작성일: 2026-07-26  
대상 시스템: `lib2`, `lib3` pneumatic simulator  
목표: **PID/I-term 없이 RL만으로 충분한 제어 성능을 확보하고, 실패 원인을 exploration·reference·환경 동역학·architecture로 분리한다.**

---

## 1. 비교 기준: pure SAC가 아니라 과거 Ours

이번 실험의 기준은 단순히 “최근 pure SAC가 잘 됐는가”가 아니다. 과거에 성능이 좋았던

`260517_15_57_47_051lib2_Ours_Simulation`

과 최근 ablation 모델을 비교해야 한다.

다만 두 모델은 같은 조건이 아니다. 올바른 비교 문장은 다음과 같다.

> 과거 Ours 구성은 history, multi-step action, prediction/future reward, CAPS 등의 요소를 함께 사용했을 때 좋은 성능을 보였다. 최근의 단순화된 ablation 구성은 이 요소들을 제거하거나 변경했기 때문에, 성능 저하가 SAC 자체의 문제인지, 제거된 보조 요소의 문제인지, reference와 episode 조건의 문제인지 분리해서 확인해야 한다.

### 1.1 과거 Ours 구성

과거 Ours는 단순 SAC가 아니다.

- `num_prev=10`: observation history
- `num_pred=15`: 미래 상태 prediction
- `num_act=5`: multi-step action
- future/prediction reward
- CAPS policy regularization
- ReplayBuffer observation perturbation
- `horizon=2048`
- 과거 random reference 범위
- 저장된 설정에 PID `Ki` 값이 포함되어 있어 training/validation에서 I-term이 실제 적용됐는지 별도 확인 필요

저장 metric은 다음과 같다.

| 모델 | POS RMSE | NEG RMSE | 의미 |
|---|---:|---:|---|
| `051lib2_Ours` | 약 12.37 kPa | 약 4.75 kPa | 과거 성능 기준 |

이것을 “pure SAC보다 좋았다”고 표현하면 부정확하다. 과거 Ours의 성공을 재현한 뒤, 구성 요소를 하나씩 제거해야 한다.

### 1.2 최근 ablation 구성

최근 screening은 다음 네 가지를 비교했다.

1. SAC
2. SAC + CAPS
3. SAC + MPObs + CAPS
4. SAC + MPObs + CAPS + I-term

현재 MPObs는 observation만 바꾸지 않는다. `num_act=5` 때문에 action dimension도 증가한다.

| library | pure SAC action dimension | MPObs action dimension |
|---|---:|---:|
| lib2 | 2 | 10 = 2 × 5 |
| lib3 | 6 | 30 = 6 × 5 |

따라서 `SAC`와 `SAC+MPObs+CAPS`의 차이는 MPObs 하나가 아니다. history, prediction, multi-step action, action dimension, CAPS가 동시에 변한다.

---

## 2. Policy action과 실제 valve command

Policy 출력과 simulator가 받는 physical command를 구분해야 한다.

정규화된 policy action은

$$
a_{RL}\in[0,1]^d
$$

이고, 환경의 physical scaling은

$$
u_{physical}
=u_{min}+(u_{max}-u_{min})a_{RL}
$$

이다. 보통 `u_max=1.0`이다.

### 2.1 `scale_low=0.8`일 때

| RL action | physical command |
|---:|---:|
| 0.0 | 0.80 |
| 0.5 | 0.90 |
| 1.0 | 1.00 |

### 2.2 `scale_low=0.85`일 때

| RL action | physical command |
|---:|---:|
| 0.0 | 0.85 |
| 0.5 | 0.925 |
| 1.0 | 1.00 |

따라서 “policy가 0.5를 냈다”만으로 valve가 실제로 얼마나 열렸는지는 알 수 없다. normalized action과 physical action을 함께 기록해야 한다.

### 2.3 lib2의 0.8과 0.85

`0.8`과 `0.85` 중 하나가 항상 정답인 것은 아니다. lib2 valve model은 cracking region에서 command 변화가 유량에 직접 반영되지 않거나, 기대한 pressure direction과 다른 거동을 보일 수 있다.

현재 screening 기본값은 lib2 `0.85`, lib3 `0.8`이다. 반면 feasible reference profile이 `[0.8,1.0]`을 가정하면 학습도 반드시

```bash
--scale-low 0.8
```

로 실행해야 한다. reference와 training scale이 다르면 같은 reference라도 다른 plant task가 된다. 최종 선택은 open-loop valve sweep으로 해야 한다.

---

## 3. SAC exploration의 전체 구조

현재 SAC exploration은 하나의 noise가 아니라 다음 요소의 조합이다.

1. 초기 warm-up action
2. Gaussian stochastic policy sampling
3. 자동 entropy coefficient `alpha`
4. 선택적인 Pink noise
5. replay buffer에 축적되는 state-action transition

각 요소는 다른 역할을 한다. CAPS는 직접적인 exploration noise가 아니라 policy regularization이다.

---

## 4. Gaussian policy sampling

Policy network는 state에서 action mean과 log standard deviation를 출력한다.

$$
(mu_\theta(s),\log\sigma_\theta(s))=\pi_\theta(s)
$$

$$
\sigma_\theta(s)=\exp(\log\sigma_\theta(s))
$$

그 다음

$$
z=\mu_\theta(s)+\sigma_\theta(s)\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I)
$$

를 계산하고 `tanh`를 통과시킨다.

$$
a=\tanh(z)
$$

현재 action space가 `[0,1]`이므로 사실상

$$
a_{RL}=\frac{\tanh(z)+1}{2}
$$

와 같은 형태로 정규화된다.

같은 state에서 매번 다른 command가 나오는 이유는 mean이 매번 달라서가 아니라, 같은 mean 주변에서 Gaussian sampling을 하기 때문이다.

---

## 5. Warm-up: 처음 10 episode

### 5.1 과거 코드

구버전 `train.py`/SAC에서는 첫 `start_epi=10` episode 동안

$$
a_i\sim\mathcal{U}(0,1)
$$

형태의 uniform random action을 replay buffer에 넣었다.

lib2에서는

$$
(a_{pos},a_{neg})\sim\mathcal{U}(0,1)^2
$$

이다.

### 5.2 horizon과 warm-up transition

50 Hz에서

$$
\Delta t=1/50=0.02\ \mathrm{s}
$$

이다.

| 설정 | episode당 step | warm-up episode | 총 transition | 연속 물리시간 |
|---|---:|---:|---:|---:|
| carry off | 2048 | 10 | 20,480 | episode별 40.96 s |
| carry on | 512 | 10 | 5,120 | 연속 102.4 s |

carry off에서는 매 episode plant가 초기화된다. carry on에서는 10개 episode가 독립 rollout이 아니라 하나의 긴 physical trajectory가 된다. 따라서 `start_epi=10`의 의미는 carry에 따라 완전히 달라진다.

### 5.3 현재 working tree

현재 코드는 uniform warm-up 대신 lib2 open-loop characterization으로 얻은 nominal physical command를 사용하도록 바뀌었다.

$$
u_{warmup,physical}\approx(0.96,0.98)
$$

normalized warm-up action은

$$
a_{warmup}
=
\frac{u_{warmup,physical}-u_{min}}
{u_{max}-u_{min}}
$$

으로 변환한다.

단, `260722`, `260723` 모델은 이 변경 전 코드로 학습됐을 가능성이 높다. 과거 모델 결과에 현재 warm-up 정책을 소급해서 해석하면 안 된다.

---

## 6. SAC entropy와 alpha

SAC policy objective는

$$
J_\pi
=
\mathbb{E}_{s\sim D,a\sim\pi}
\left[
\alpha\log\pi(a|s)
-\min(Q_1(s,a),Q_2(s,a))
\right]
$$

이다.

- `Q` 항: 높은 value action 선택
- `alpha log pi` 항: 너무 빠른 policy collapse 방지

alpha가 크면 entropy를 유지하는 압력이 크고, 작으면 reward가 좋다고 판단한 action으로 수렴하는 압력이 커진다.

### 6.1 자동 alpha tuning

현재 code는 자동 entropy tuning을 사용할 때 다음을 실행한다.

```python
self.target_ent = -action_dim
self.log_alpha = torch.zeros(1, requires_grad=True)
self.alpha = self.log_alpha.exp()
```

따라서 자동 tuning이 켜져 있으면 config에 적힌 `alpha=0.5`가 실제 초기 alpha로 직접 사용되지 않는다.

$$
\log\alpha=0
\quad\Rightarrow\quad
\alpha=1
$$

alpha loss는

$$
L_\alpha
=
-\mathbb{E}
\left[
\log\alpha
\left(\log\pi(a|s)+\mathcal{H}_{target}\right)
\right]
$$

이다. 현재 target entropy는

$$
\mathcal{H}_{target}=-d
$$

이며 `d`는 action dimension이다.

| 구성 | action dimension | target entropy |
|---|---:|---:|
| lib2 pure SAC | 2 | -2 |
| lib3 pure SAC | 6 | -6 |
| lib2 MPObs | 10 | -10 |
| lib3 MPObs | 30 | -30 |

그러므로 lib2와 lib3, pure SAC와 MPObs의 alpha를 숫자만으로 비교하면 안 된다. alpha와 함께 action std, saturation, joint action coverage를 봐야 한다.

### 6.2 alpha가 말해주지 않는 것

alpha는 SAC objective가 entropy를 얼마나 중요하게 보는지를 말해준다. 하지만 다음을 보장하지 않는다.

- 실제 valve combination의 다양성
- cracking region 방문
- pressure rate 방향 coverage
- 특정 action 조합 편향 없음
- 유용한 transition 탐색

alpha가 큰데도 physical action이 `[0.85,0.85]` 부근에만 몰릴 수 있다.

---

### 6.3 한 step에서 네트워크가 업데이트되는 전체 과정

SAC는 episode가 끝난 뒤에만 한 번 업데이트하는 구조가 아니다. 현재 설정에서는 replay buffer에 transition이 충분히 쌓인 뒤, **환경 step마다 네트워크를 업데이트**한다.

한 step의 흐름은 다음과 같다.

1. 현재 state \(s_t\)에서 action \(a_t\) 선택
2. simulator가 action 실행
3. 다음 state \(s_{t+1}\)와 reward \(r_t\) 수신
4. \((s_t,a_t,r_t,s_{t+1},done)\)를 replay buffer에 저장
5. buffer 크기가 batch size보다 크면 critic, actor, alpha 업데이트
6. critic target network를 soft update

transition은 다음처럼 저장된다.

$$
(s_t,a_t,r_t,s_{t+1},m_t)\in D
$$

여기서 \(m_t\)는 terminal이 아니면 1, terminal이면 0에 가까운 mask다.

현재 주요 설정이 batch size 128, epoch 1, update every 1이면 buffer에 129개 정도 transition이 쌓인 뒤부터 대략 environment step 1회당 gradient update 1회가 일어난다. 따라서 start epi 10 동안 policy action을 선택하지 않더라도, warm-up transition이 buffer에 충분히 쌓이면 network update는 진행될 수 있다.

> “처음 10 episode는 random/safe action만 실행한다”와 “처음 10 episode 동안 network가 전혀 학습하지 않는다”는 같은 말이 아니다. 현재 구현에서는 warm-up action으로 buffer를 채우면서 critic, actor, alpha update가 진행될 수 있다.

#### 1단계: Critic target 계산

다음 state에서 policy가 새 action을 sampling한다.

$$
a_{t+1}\sim\pi_\theta(\cdot|s_{t+1})
$$

두 target critic 중 작은 값을 사용한다.

$$
Q_{target}(s_{t+1},a_{t+1})
=
\min(Q'_1,Q'_2)
-\alpha\log\pi_\theta(a_{t+1}|s_{t+1})
$$

replay buffer의 reward를 앞에 붙여 Bellman target을 만든다.

$$
y_t
=
r_t
+m_t\gamma
\left[
\min(Q'_1,Q'_2)
-\alpha\log\pi_\theta(a_{t+1}|s_{t+1})
\right]
$$

현재 critic loss는

$$
L_Q
=
\operatorname{Huber}(Q_1(s_t,a_t),y_t)
+
\operatorname{Huber}(Q_2(s_t,a_t),y_t)
$$

이다. 즉 reward는 먼저 critic이 “이 state에서 이 action을 했을 때 앞으로 얼마나 좋은가”를 학습하는 target으로 사용된다.

#### 2단계: Actor update

현재 state에서 새 action을 sampling한다.

$$
\tilde a_t\sim\pi_\theta(\cdot|s_t)
$$

현재 critic이 그 action의 value를 평가한다.

$$
Q_{min}(s_t,\tilde a_t)
=
\min(Q_1(s_t,\tilde a_t),Q_2(s_t,\tilde a_t))
$$

pure SAC policy loss는

$$
L_{policy}^{SAC}
=
\mathbb{E}
\left[
\alpha\log\pi_\theta(\tilde a_t|s_t)
-Q_{min}(s_t,\tilde a_t)
\right]
$$

이다.

- \(Q\)가 높은 action은 loss를 낮추므로 더 자주 선택된다.
- log probability가 지나치게 집중되면 alpha 항이 이를 완화한다.
- reward가 actor에 직접 들어가는 것이 아니라, reward로 학습된 critic의 Q를 통해 actor에 전달된다.

CAPS를 켜면

$$
L_{policy}
=
L_{policy}^{SAC}
+L_{temporal}
+L_{spatial}
$$

가 된다.

#### 3단계: Alpha update

alpha는 reward를 직접 입력으로 받지 않는다. alpha update는 policy의 log probability와 target entropy를 사용한다.

$$
L_\alpha
=
-\mathbb{E}
\left[
\log\alpha
\left(
\log\pi_\theta(a_t|s_t)
+\mathcal{H}_{target}
\right)
\right]
$$

그러나 reward는 alpha에 간접적으로 영향을 준다.

$$
r_t
\rightarrow
Q(s,a)\text{ 변화}
\rightarrow
\pi_\theta\text{ 변화}
\rightarrow
\log\pi\text{ 변화}
\rightarrow
\alpha\text{ 변화}
$$

따라서 reward가 나쁘다고 alpha가 즉시 커지는 것은 아니다. reward가 나쁘면 critic이 action을 낮게 평가하고, actor가 action distribution을 바꾸며, 그 distribution의 entropy가 target과 달라졌을 때 alpha가 변한다.

#### 4단계: Target critic soft update

마지막으로 target critic은 current critic을 조금씩 따라간다.

$$
\phi'
\leftarrow
\tau\phi+(1-\tau)\phi'
$$

현재 tau는 0.005이므로 target network는 한 번의 update에서 current critic을 약 0.5%만 따라간다. 이것은 Q target이 갑자기 크게 변해 학습이 불안정해지는 것을 줄인다.

### 6.4 reward가 네트워크에 미치는 숫자 예시

한 transition에서

$$
r_t=-100,
\qquad
\gamma=0.9,
\qquad
\min(Q'_1,Q'_2)=-20,
\qquad
\log\pi(a_{t+1}|s_{t+1})=-1.5,
\qquad
\alpha=0.5
$$

라고 하자. entropy 항을 포함한 다음 value는

$$
-20-0.5(-1.5)=-19.25
$$

이고 critic target은

$$
y_t=-100+0.9(-19.25)=-117.325
$$

가 된다.

같은 state/action에서 reward가 -20으로 개선되면

$$
y_t=-20+0.9(-19.25)=-37.325
$$

가 된다. critic은 두 transition을 서로 다른 value로 학습하고, actor는 높은 Q를 주는 action 쪽으로 이동한다.

반대로 pressure error 제곱 때문에 reward가 -10000처럼 매우 커지면 critic target scale도 매우 커진다. 그러면 Q value가 큰 음수 범위로 이동하고, critic gradient가 reward scale에 지배되며, actor가 Q 추정 오차에 민감해질 수 있다. 따라서 reward는 단순할 뿐 아니라 scale도 확인해야 한다.

### 6.5 Episode가 끝날 때 실제로 일어나는 일

현재 training loop에서 episode는 horizon만큼 환경을 실행하는 학습 구간이다.

Episode 중에는 매 step마다 transition 저장, buffer sampling, critic/actor/alpha update, pressure/action diagnostics 기록이 진행된다.

Episode 종료 시에는 다음이 수행된다.

- episode reward 합산 출력
- model checkpoint 저장
- train diagnostics 저장
- exploration diagnostics 저장
- 100 episode마다 replay buffer 저장
- 다음 episode에서 env.reset() 호출

중요한 점은 episode reward 합계가 network update에 한 번에 직접 들어가지 않는다는 것이다.

$$
R_{episode}=\sum_{t=0}^{H-1}r_t
$$

는 성능 기록과 logging에 사용되고, 실제 update는 각 transition의 \(r_t\)를 minibatch로 sampling해서 수행한다.

#### carry false

다음 episode에서 simulator pressure와 reference가 초기화된다.

$$
P^{k+1}(0)=P_{init}
$$

따라서 episode reward가 좋아졌다는 것은 같은 초기 조건에서 해당 rollout tracking이 좋아졌다는 의미에 가깝다.

#### carry true

다음 episode가 이전 physical state에서 이어진다.

$$
P^{k+1}(0)=P^k(T)
$$

따라서 episode reward가 갑자기 나빠져도 policy가 갑자기 바뀐 것이 아니라, 이전 episode에서 plant가 이미 불리한 pressure 영역으로 이동했기 때문일 수 있다. carry에서는 episode별 reward보다 전체 physical time trajectory와 state drift를 함께 봐야 한다.

#### Episode 경계와 terminal state

horizon이 끝났다는 이유만으로 physical system이 terminal state에 도달한 것은 아니다. 단순히 학습 구간을 끊은 것일 수 있다. carry true에서는 다음 episode가 이어지므로 horizon 경계가 실제 terminal transition이 되면 안 된다. done, replay mask, environment reset semantics를 반드시 일치시켜야 한다.

---

## 7. gamma와 delayed pressure response

현재 screening config는 `gamma=0.9`이다. SAC constructor default는 별도로 `0.8`이므로 실행된 `cfg.yaml`의 실제 값을 확인해야 한다.

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
$$

`gamma=0.9`이면

$$
0.9^{50}\approx0.00515,
\qquad
0.9^{100}\approx2.65\times10^{-5}
$$

이다. 50 Hz에서 50 step은 1초다. pneumatic pressure가 의미 있게 변하는 데 수 초가 걸리면, valve command의 장기 효과가 policy에 약하게 전달될 수 있다.

이것은 exploration 문제와 별개로 중요한 credit assignment 문제다.

> 필요한 valve combination을 실제로 시도했지만 pressure response가 늦게 나타나고, 그 뒤늦은 개선이 원인 action에 충분히 credit되지 않았을 가능성이 있다.

과거 Ours의 future/prediction reward와 multi-step action은 이 지연을 일부 보완했을 가능성이 있다.

---

## 8. CAPS는 exploration인가?

CAPS는 직접적인 action noise가 아니라 policy regularization이다.

Temporal term:

$$
L_{temporal}
=
w_T\|\pi(s_t)-\pi(s_{t+1})\|_2
$$

Spatial term:

$$
L_{spatial}
=
w_S\|\pi(s_t)-\pi(\tilde{s}_t)\|_2
$$

`\tilde{s}_t`는 observation perturbation state다. 최종 policy loss는

$$
J_{policy}=J_\pi+L_{temporal}+L_{spatial}
$$

이다.

현재 screening의 대표 설정:

| 항목 | pure SAC | CAPS |
|---|---:|---:|
| temporal weight | 0 | 0.5 |
| spatial weight | 0 | 0.4 |
| replay noise std | 0 | 1.0 |

CAPS가 진동을 줄여도 필요한 valve switching까지 억제할 수 있다. 따라서 CAPS는

$$
\text{탐색 다양성}
\leftrightarrow
\text{action smoothness}
$$

의 trade-off를 가진다.

---

## 9. Pink noise

white noise는 매 step 독립적인 noise다.

$$
\eta_t\sim\mathcal{N}(0,\sigma^2I)
$$

하지만 pneumatic valve는 매 20 ms마다 독립적으로 크게 움직이는 것보다, 일정 시간 같은 방향으로 유지했다가 천천히 바꾸는 탐색이 의미 있을 수 있다.

Pink noise는 시간 상관을 갖는 noise다.

$$
u_t
=
\operatorname{clip}
\left(
\mu_\theta(s_t)+\sigma_{pink}\eta_t^{pink},
0,1
\right)
$$

현재 구현은 Voss–McCartney 방식의 online approximation을 사용한다. 여러 octave random process를 합쳐 대략적인 1/f 특성을 만든다.

실행 예시는 다음과 같다.

    --exploration pink
    --pink-sigma 0.05
    --pink-octaves 8

현재 screening에서 Pink noise를 켜면 action 선택은 다음처럼 된다.

    none:
        policy Gaussian sample

    pink:
        policy mean + temporally correlated pink noise

Pink noise가 항상 정답은 아니다. amplitude가 너무 크면 operating region을 벗어나고, 너무 작으면 새로운 valve combination을 거의 방문하지 못한다. 반드시 policy mean, final action, pink noise, saturation, pressure rate를 함께 기록해야 한다.

---

## 10. 탐색이 잘 되었는지 확인하는 방법

reward 하나만 봐서는 exploration이 충분했는지 알 수 없다.

### 10.1 Policy distribution

exploration_diagnostics.csv에서 다음을 본다.

#### log std와 action std

$$
\sigma_i(s)=\exp(\log\sigma_i(s))
$$

특정 action dimension의 std가 계속 작으면 해당 valve가 거의 고정된 것이다. 예를 들어 lib3에서

$$
std(u_0)=0.25,
\qquad
std(u_1)=0.03
$$

이면 u1 방향을 거의 탐색하지 않았을 가능성이 있다.

#### Saturation ratio

$$
R_{sat}
=
\frac{
\#\{a_i\le\epsilon\ \lor\ a_i\ge1-\epsilon\}
}{
N\times d
}
$$

sat가 높으면 policy가 action boundary에 붙어 있다는 뜻이다. physical boundary는 normalized action의 boundary인 scale_low와 1.0이다.

### 10.2 Action histogram

각 valve별 histogram으로 다음을 확인한다.

- 모든 valve가 여러 command 구간을 방문했는가
- [0,0], [1,1] 또는 특정 corner에 몰렸는가
- inlet/outlet이 동시에 열리는 조합이 반복되는가
- 특정 action dimension의 분산이 거의 0인가

### 10.3 Joint action distribution

lib2에서는

$$
(u_{pos},u_{neg})
$$

의 2차원 scatter를 그린다. lib3에서는 주요 valve pair를 선택한다.

또한

$$
\operatorname{corr}(u_i,u_j)
$$

를 본다. 상관이 +1 또는 -1에 가까우면 물리적으로 필요한 coupling일 수도 있지만, policy가 특정 1차원 경로에 갇힌 것일 수도 있다.

### 10.4 Physical pressure-rate coverage

각 transition에서

$$
\dot P_{pos,t}
\approx
\frac{P_{pos,t+1}-P_{pos,t}}{\Delta t}
$$

$$
\dot P_{neg,t}
\approx
\frac{P_{neg,t+1}-P_{neg,t}}{\Delta t}
$$

를 계산한다. 그리고

$$
(u_{pos},u_{neg})
\rightarrow
(\dot P_{pos},\dot P_{neg})
$$

를 시각화한다.

확인해야 할 질문은 다음이다.

- positive pressure를 올리는 방향과 내리는 방향을 모두 경험했는가
- negative pressure를 올리는 방향과 내리는 방향을 모두 경험했는가
- target tracking에 필요한 rate 영역을 실제로 방문했는가
- 특정 command 영역에서만 pressure rate가 관측됐는가

이 지표가 있어야 alpha가 높았다는 것과 필요한 physical direction을 탐색했다는 것을 구분할 수 있다.

---

## 11. 지금까지 lib2 ablation 결과

### 11.1 512-step reset SAC

260722_screen_lib2_sac_seed0은 horizon 512, carry false다.

| 시점 | alpha | log std mean | action std mean | saturation |
|---|---:|---:|---:|---:|
| episode 11 | 약 1.74 | 약 0.54 | 약 0.38 | 약 0.25 |
| episode 1000 | 약 0.042 | 약 -0.43 | 약 0.35 | 약 0.06 |

마지막 metric은 대략 POS RMSE 23.7 kPa, NEG RMSE 11.9 kPa다. 그러나 512 step은 10.24초이므로 과거 Ours의 40.96초 rollout과 직접 비교할 수 없다.

### 11.2 2048-step reset SAC

260723_screen_lib2_h2048_reset_sac_seed0은 carry false, horizon 2048이다.

- reward: 약 -17,238
- POS RMSE: 약 34.5 kPa
- NEG RMSE: 약 7.2 kPa
- action std mean: 약 0.37
- saturation: 약 0.059
- physical command mean: POS 약 0.913, NEG 약 0.909

완전히 학습이 안 된 것은 아니다. NEG는 비교적 따라가지만 POS tracking이 약하다.

> lib2에서 positive pressure를 필요한 방향으로 움직이는 valve command region과 그 결과를 policy가 안정적으로 발견·유지하지 못했다.

### 11.3 512-step carry SAC

260723_screen_lib2_h512_carry_sac_seed0은 carry true, horizon 512이다.

- reward: 약 -622,573
- POS RMSE: 약 3,977 kPa
- NEG RMSE: 약 76.2 kPa
- action std mean: 약 0.163
- physical command mean: POS 약 0.814, NEG 약 0.864

이것은 단순 Gaussian exploration failure만으로 설명하기 어렵다. carry rollout에서 plant가 이전 episode의 압력과 내부 상태를 이어가면서 잘못된 영역으로 이동한 뒤 회복하지 못한 가능성이 크다.

과거 carry 처리에서는 다음 문제가 있었다.

- chamber pressure만 carry되고 pump/valve internal state 보존이 불명확
- episode 시작 때 강제 action이 simulator에 들어갈 수 있었음
- normalized low action이 physical [0.8,0.8] 또는 [0.85,0.85]가 될 수 있었음
- 시작 command가 hidden reset action처럼 작용할 가능성

현재 working tree에서는 reset 시 simulator를 advance시키는 hidden reset action을 제거하고, carry reset에서 simulator·reference·observation history를 보존하도록 수정되어 있다. 따라서 260723 결과를 현재 carry semantics의 결과로 해석하면 안 된다.

### 11.4 중간 결론

최근 pure SAC 실패를 곧바로 exploration 부족이라고 단정할 수 없다. 현재 타당한 원인 후보는 다음과 같다.

1. carry/reset semantics
2. reference-horizon 불일치
3. valve operating region
4. delayed response에 대한 credit assignment
5. Ours 요소를 동시에 제거한 architecture ablation
6. 필요한 joint valve combination을 충분히 방문하지 못한 exploration

---

## 12. 과거 Ours가 최근 ablation보다 좋아 보였던 이유

### 12.1 Ours는 더 많은 정보를 사용했다

pure SAC는 현재 observation과 reference에서 1-step action을 낸다. Ours의 history는 다음을 제공한다.

- pressure가 올라가는 중인지 내려가는 중인지
- 이전 valve command에 response가 있었는지
- pressure slope가 어느 방향인지
- 같은 pressure error라도 plant 내부 상태가 다른지

따라서 Ours의 성능 향상을 단순히 exploration이 더 좋았다고만 볼 수 없다. 관측 가능성과 credit assignment가 좋아졌을 수 있다.

### 12.2 Multi-step action

num_act=5는 policy가 여러 step control sequence를 제공하는 구조다. 이 구조는 매 20 ms마다 command가 바뀌는 현상을 줄이고, valve를 일정 시간 유지해 pressure response를 관측하게 한다.

### 12.3 Future/prediction reward

pure SAC의 current reward는 현재 pressure error에 주로 의존한다. Ours는 미래 prediction과 future error를 사용해 delayed pressure response에 대한 signal을 앞쪽 action에 전달했을 가능성이 있다.

### 12.4 Horizon과 reference

과거 Ours는 2048 step, 약 40.96초 rollout이었다. 최근 512-step reset 모델은 약 10.24초다. reference segment가 10초보다 긴 시간 scale이면 512-step episode는 transition을 완성하기 전에 끝난다.

과거 Ours의 성공은 다음의 결합 결과일 수 있다.

$$
\text{Ours architecture}
+\text{history}
+\text{multi-step action}
+\text{future/pred reward}
+\text{CAPS}
+\text{horizon 2048}
+\text{legacy reference}
+\text{특정 reset/carry semantics}
$$

### 12.5 PID/I-term 재현성

과거 Ours cfg에는 Ki_pos=0.01, Ki_neg=0.01이 저장되어 있다. 반면 train2.py의 Ours branch는 pid=None으로 보이며, 실제 training에서 env.set_pid()가 호출되지 않았을 가능성이 있다.

확인 전에는 다음을 단정하면 안 된다.

- Ours는 I-term 없이 성공했다
- Ours는 I-term 덕분에 성공했다

확인할 것은 training 중 set_pid 호출, trajectory의 ctrl_input, validation PID 설정, cfg와 runtime state 일치 여부다.

---

## 13. Reference가 중요한 이유

간단한 current tracking reward는

$$
r_t
=
-w_p(P_{pos,t}-P^{ref}_{pos,t})^2
-w_n(P_{neg,t}-P^{ref}_{neg,t})^2
$$

형태다.

reference가 plant가 따라갈 수 없는 속도로 변하면 policy가 최선의 action을 내도 error가 남는다.

$$
\text{low reward}
\neq
\text{항상 policy 실패}
$$

일 수 있으며,

$$
\text{low reward}
=
\text{policy error}
+\text{reference infeasibility}
+\text{state/carry mismatch}
$$

로 분해해야 한다.

### 13.1 Legacy random reference

기존 random reference가 pressure min/max 범위에 target을 만들더라도 다음은 자동 보장되지 않는다.

- transition time이 충분한가
- pressure rate가 plant capacity 이하인가
- positive/negative 동시 변화가 가능한가
- carry 상태에서 다음 segment로 이동 가능한가
- horizon 안에 transition이 끝나는가

### 13.2 Feasible reference

reference rate를

$$
|\dot P_i^{ref}(t)|\le\rho_i^{safe}
$$

처럼 제한한다. 여기서 robust rate는 theoretical maximum valve rate가 아니라 여러 operating point에서 반복해도 안정적인 pressure rate다.

그러나 feasible reference는 rate만으로 완성되지 않는다.

$$
\text{feasible reference}
=
\text{pressure range}
+\text{rate limit}
+\text{transition duration}
+\text{initial state}
+\text{carry/reset policy}
$$

### 13.3 Reset과 carry

carry off:

$$
P_{plant}^{(k+1)}(0)=P_{init}
$$

각 episode가 독립 task다.

carry on:

$$
P_{plant}^{(k+1)}(0)
=
P_{plant}^{(k)}(T)
$$

episode는 physical reset이 아니라 RL update boundary가 된다. reference도 계속 진행된다.

---

## 14. lib3에서 가능한 4-control 실험

lib3 simulator는 물리적으로 6 input을 받는다.

    u[0] : positive chamber valve
    u[1] : negative chamber valve
    u[2] : actuator positive inlet
    u[3] : actuator positive outlet
    u[4] : actuator negative inlet
    u[5] : actuator negative outlet

C++ simulator를 진짜 4 input으로 다시 만드는 것은 mass-flow equation, C ABI, Python wrapper, indexing을 모두 확인해야 하는 큰 변경이다.

### 14.1 권장하는 첫 4-control 방법

현재 PneuEnv3에는 chamber control을 고정하고 policy가 actuator 4개만 출력하는 기능이 있다.

    fixed_chamber_ctrl=(u_ch_pos, u_ch_neg)

policy action은

$$
a_{policy}\in[0,1]^4
$$

이고 simulator에는

$$
u_{sim}
=
[u_{ch,pos}^{fixed},u_{ch,neg}^{fixed},a_0,a_1,a_2,a_3]
$$

가 전달된다. C++ 수정 없이 “6 policy controls가 필요한가?”를 확인할 수 있다.

### 14.2 첫 4-control 조건

- pure SAC
- num_act=1
- MPObs off
- CAPS off
- I-term off
- 동일 reference, initial state, reward, seed
- chamber fixed command는 open-loop sweep으로 결정

steady_chamber_ctrl은 내부 PI-like controller이므로 pure RL 능력을 보는 첫 실험에는 사용하지 않는다.

### 14.3 해석

4-control이 성공하면 6-valve redundancy, chamber-actuator coupling, action dimension이 탐색을 어렵게 했을 가능성이 있다. 4-control도 실패하면 reference, valve operating region, gamma/horizon, observation, reward scale을 먼저 의심한다.

---

## 15. 권장 실험 순서

### 15.1 Open-loop plant 확인

다음 command를 일정하게 유지해 pressure trajectory와 dP/dt를 기록한다.

    (0.80,0.80)
    (0.85,0.85)
    (0.92,0.96)
    (0.96,0.98)
    (0.98,1.00)

40.96초 동안 bounded인지 확인하고, carry/reset에서 episode 시작 action이 다르게 들어가는지도 확인한다.

### 15.2 Reference 검증

reference에 대해

$$
\max_t|\dot P_{pos}^{ref}(t)|,
\qquad
\max_t|\dot P_{neg}^{ref}(t)|
$$

를 계산하고 open-loop robust capacity와 비교한다. 512-step에서 transition이 완성되는지도 확인한다.

### 15.3 과거 Ours 재현

다음을 먼저 맞춘다.

- horizon 2048
- reference 범위와 seed
- scale low/high
- reset/carry
- num_prev, num_pred, num_act
- future/prediction reward
- CAPS와 replay noise
- PID/I-term runtime state

### 15.4 요소별 ablation

| 단계 | 구성 | 질문 |
|---|---|---|
| A | Ours | 기준 성능 재현 여부 |
| B | Ours - future/pred reward | delayed reward가 중요한가 |
| C | Ours - CAPS | 진동 억제 효과가 있는가 |
| D | Ours - MPObs/history | memory가 중요한가 |
| E | pure SAC | 보조 요소 없이 가능한가 |
| F | pure SAC + Pink | temporal exploration이 도움이 되는가 |
| G | pure SAC + history | memory만으로 개선되는가 |

현재처럼 pure SAC와 MPObs+CAPS+I-term을 바로 비교하면 한 번에 너무 많은 요소가 변한다.

### 15.5 lib3

1. lib3 6-control pure SAC
2. lib3 6-control pure SAC + Pink
3. lib3 4-policy-control, chamber fixed
4. 6-control과 4-control의 action/pressure-rate coverage 비교
5. 그 후 CAPS, MPObs, I-term 추가

---

## 16. 교수님께 보고할 수 있는 결론

### 16.1 확정적으로 말할 수 있는 것

- 과거 Ours는 pure SAC와 같은 조건이 아니었다.
- 최근 MPObs ablation은 action dimension도 증가시킨다.
- 512 step은 10.24초, 2048 step은 40.96초다.
- carry on은 independent reset이 아니라 physical continuation이다.
- alpha는 exploration pressure이지 physical coverage 자체가 아니다.
- lib3에는 C++ 수정 없이 chamber를 고정해 4-policy-control을 시험할 경로가 있다.
- 최근 lib2 carry 결과는 reset보다 크게 발산했다.

### 16.2 아직 가설인 것

- 주원인이 exploration 부족인지
- lib2 lower bound 0.8이 항상 잘못인지
- 과거 Ours training에서 실제 I-term이 켜졌는지
- reference rate가 지나치게 보수적인지
- gamma=0.9가 delayed response에 충분한지
- 6-valve redundancy가 lib3 실패의 주원인인지

### 16.3 랩미팅 권장 표현

> 과거 Ours는 높은 성능을 보였지만 pure SAC와 같은 조건이 아니었고, history·multi-step action·future/prediction reward·CAPS가 함께 사용되었다. 최근 ablation에서는 architecture, episode horizon, carry semantics, reference profile, action scaling이 동시에 달라졌기 때문에 성능 저하를 exploration 부족 하나로 결론내릴 수 없다. 먼저 open-loop plant capacity와 reference를 검증하고, carry/reset semantics를 고정한 뒤 과거 Ours를 동일 조건에서 재현한다. 그 다음 요소를 하나씩 제거하고, lib3에서는 simulator를 수정하지 않는 4-policy-control 실험으로 redundancy와 exploration을 분리한다.

---

## 17. 앞으로 저장할 로그

### 학습 설정

- cfg.yaml
- git commit hash
- lib version
- train/reference seed
- reference profile source
- scale low/high
- initial state
- horizon/carry
- warm-up mode/action
- alpha tuning 여부
- gamma
- action dimension
- PID/I-term 여부

### Episode metric

- reward
- POS/NEG RMSE와 bias
- pressure sign-change rate
- physical action mean/min/max
- normalized action mean/min/max
- action delta
- alpha
- log std min/mean/max
- action std per dimension
- saturation ratio

### Exploration metric

- action histogram
- joint action scatter
- per-dimension correlation
- policy mean vs sampled action
- Pink noise value
- dP/dt histogram
- action과 pressure-rate joint plot
- reference slope histogram

### Validation metric

- 동일 reference인지
- 동일 initial state인지
- validation PID/I-term이 training과 동일한지
- deterministic/stochastic policy 구분
- reference/plant time axis 일치 여부

---

## 18. 최종 요약

최근 pure SAC가 못난 이유가 단순히 exploration 부족이라고 아직 단정할 수 없다. 과거 Ours와 최근 ablation 사이에서 architecture, reward, action dimension, horizon, carry/reset, reference, action scaling이 동시에 변했기 때문이다.

권장 순서는

$$
\text{open-loop plant 확인}
\rightarrow
\text{reference 검증}
\rightarrow
\text{carry/reset 고정}
\rightarrow
\text{과거 Ours 재현}
\rightarrow
\text{요소별 ablation}
\rightarrow
\text{pure SAC exploration 개선}
\rightarrow
\text{lib3 6/4-control 비교}
$$

이다.

exploration 성공 여부는 alpha 하나가 아니라

$$
\text{policy distribution}
+\text{joint action coverage}
+\text{pressure-rate coverage}
+\text{delayed-response credit assignment}
$$

를 함께 봐야 판단할 수 있다.

---

이 문서는 현재 코드와 저장된 diagnostic을 기준으로 작성했으며, 검증하지 않은 내용은 가설로 구분했다. 로컬 PC로 옮겨도 별도 파일 링크 없이 문서 자체만으로 SAC update, exploration, ablation, reference 설계를 이해할 수 있도록 구성했다.
