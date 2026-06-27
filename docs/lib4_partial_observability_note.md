# lib4 Partial Observability Note

## 한 줄 요약

`lib4`는 단순한 fully observed MDP라기보다,  
**delay + coupling + hidden valve dynamics + redundant action**이 섞인  
**partially observed control problem (부분관측 제어 문제)** 로 보는 게 더 맞다.

즉 문제의 핵심은 단순히 "valve nonlinearity가 있어서 어렵다"가 아니라,

- 내부적으로 안 보이는 상태(hidden state)가 있고
- 현재 관측만으로는 다음 반응을 완전히 알기 어렵고
- action도 6D direct valve라서 너무 redundant하다는 점이다.

---

## 1. 우리가 지금 policy에 주는 관측 (observed state)

현재 `env4` policy가 직접 보는 정보는 대략 다음이다.

- chamber pressure
- actuator pressure
- displacement
- displacement velocity
- reference trajectory
- optional history window

겉으로 보면 pressure와 displacement를 다 보니까 state가 충분해 보일 수 있다.  
하지만 실제 lib4 dynamics는 이 정보만으로 완전히 결정되지 않을 가능성이 크다.

---

## 2. lib4 내부에서 policy가 직접 못 보는 것들 (hidden state)

lib4 valve model은 밸브마다 내부 runtime state를 가진다.

- hysteresis state `z`
- internal dynamic states `x1`, `x2`
- previous current `I_prev`
- previous switching state `state_prev`

이게 6개 밸브 각각에 존재한다.

즉 같은 시점에:

- chamber pressure가 같고
- actuator pressure가 같고
- displacement가 같아도

밸브 내부 상태가 다르면 다음 스텝 반응이 달라질 수 있다.

이건 RL 관점에서 보면,

> `observation o_t`는 비슷한데 `next state s_{t+1}`가 다르게 나오는 상황

이 생긴다는 뜻이다.  
즉 strict한 의미의 Markov state가 아닐 가능성이 높다.

추가적인 hidden structure도 있다.

- observation delay buffer
- chamber regulator integral state
- chamber-actuator-displacement coupling
- action scaling
- valve saturation

---

## 3. 왜 이게 RL에 치명적인가

현재 RL은 direct 6D valve control을 배운다.

즉 agent는 동시에 배워야 한다.

1. 어느 valve 방향이 맞는지
2. 얼마나 열어야 하는지
3. 반대 방향 valve를 얼마나 억제해야 하는지
4. 지금 action이 몇 step 뒤 pressure/displacement에 어떻게 반영되는지

그런데 state가 partially observed이면 critic 입장에서는:

- 비슷한 observation
- 비슷한 action
- 하지만 다른 outcome

이 반복될 수 있다.

그러면 value estimate가 noisy해지고, actor는 깔끔한 방향성을 배우기보다
서로 상쇄되는 애매한 조합을 학습하기 쉽다.

예:

- `act_neg_in`도 크게 열고
- `act_neg_out`도 크게 여는 식

이런 정책은 "압력을 조절하려는 척은 하지만 방향성이 분명하지 않은 정책"이다.

---

## 4. 왜 이게 PID에도 불리한가

PID는 RL보다 단순하지만, 그렇다고 해서 이 문제에서 자동으로 강한 건 아니다.

일반 PID는 기본적으로:

- 현재 error
- 과거 적분
- error derivative

정도만 본다.

하지만 lib4에서는 그걸로 충분하지 않을 수 있다.

이유:

- valve hysteresis가 있음
- transport delay가 있음
- chamber와 actuator가 강하게 coupled됨
- displacement는 pressure 2개의 결과라서 간접 제어임

그래서 PID가 안 된다고 해서 곧바로 "plant가 불가능"이라고 해석하면 안 된다.  
더 정확히는:

> 현재의 low-level control structure가 hidden dynamics와 coupling을 감당하기에 너무 약할 수 있다.

---

## 5. real system에서는 hidden state를 넣을 수 있나?

### 짧은 답

직접은 못 넣는다.

simulator 내부 hidden state는 real robot에서 측정할 수 없기 때문이다.

예를 들어:

- valve hysteresis internal state
- valve internal dynamic state
- simulator 내부 buffer state

이런 건 real에서 센서로 바로 읽을 수 없다.

---

## 6. 그럼 real에서는 어떻게 해야 하나?

정답은 **hidden state를 직접 넣는 게 아니라, hidden state를 추정할 수 있는 proxy를 넣는 것**이다.

real-friendly한 방법은 보통 이렇다.

- short observation history
- previous action history
- pressure derivative
- pressure-difference derivative
- command memory
- recurrent policy / latent state estimator

즉 real-world control에서는:

> "진짜 hidden state"를 주는 대신,  
> 그것을 간접적으로 추정할 수 있는 temporal context를 준다.

이게 partial observability를 다루는 일반적인 방식이다.

---

## 7. 이 프로젝트에서 현실적인 해법

lib4에서 제일 현실적인 첫 단계는:

1. physically measurable observation만 유지하고
2. temporal feature를 조금 추가하고
3. action ambiguity를 줄이는 것이다

그래서 최소 수정으로 넣어볼 만한 것이:

- previous applied action
- pressure-difference velocity

이다.

이 둘의 장점:

- sim에서 바로 쓸 수 있음
- real에서도 똑같이 구성 가능
- hidden valve state를 직접 쓰지 않음
- policy가 delay/hysteresis를 간접 추정하는 데 도움됨

즉 sim-only shortcut이 아니라, **real transfer 가능한 state augmentation**이라고 볼 수 있다.

---

## 8. 지금 연구 질문으로 다시 쓰면

현재 질문은 사실 이렇게 바꿔 쓰는 게 더 정확하다.

### Q1. RL이 원리적으로 안 되는가?

그건 아니다.

하지만 현재 formulation은 RL에게 매우 불리하다.

- 6D direct valve action
- partial observability
- delay
- coupling
- multi-objective reward

이 한꺼번에 걸려 있기 때문이다.

### Q2. PID가 안 되면 RL도 안 되는가?

그것도 아니다.

하지만 PID도 잘 안 되면 적어도 다음은 말할 수 있다.

> 문제는 SAC만의 문제가 아니라, system/control formulation 자체가 쉽지 않다.

### Q3. 그럼 지금 제일 의심되는 건 무엇인가?

현재로서는 다음 순서가 가장 설득력 있다.

1. action representation issue
2. partial observability / weak Markov property
3. redundant 6D valve action
4. delayed coupled pneumatic dynamics

---

## 9. paper/novelty 관점에서의 문장

논문 스타일로 가장 깔끔한 문장은 이쪽이다.

> The main difficulty is not merely valve nonlinearity, but the combination of delayed coupled pneumatic dynamics, redundant low-level valve actuation, and partial observability.

한글로 풀면:

> 핵심 난점은 단순한 밸브 비선형성 자체가 아니라,  
> 지연이 있는 coupled pneumatic dynamics, redundant한 저수준 밸브 제어입력,  
> 그리고 partial observability가 동시에 존재한다는 점이다.

이 framing은 꽤 중요하다.  
왜냐하면 이러면 "state representation 개선"이나 "control abstraction"이  
단순 튜닝이 아니라 **문제 구조를 겨냥한 방법론적 개선**으로 설명되기 때문이다.

---

## 10. 지금 단계에서 practical takeaway

지금 바로 실험적으로 해볼 만한 것은:

- `previous action` 추가
- `pressure diff derivative` 추가
- 필요하면 `pressure derivative`도 추가
- 이후 action abstraction 또는 scale/action range 재설계 검토

즉 방향은:

> hidden state를 직접 넣는 방향이 아니라,  
> real에서도 쓸 수 있는 temporal proxy를 state에 추가하는 방향

으로 가는 게 맞다.

---

## 11. 현재 모델들의 state 구성 상세 정리

이 섹션은 "지금 실제로 policy input이 몇 차원인지"를 헷갈리지 않게
아주 구체적으로 적어둔 메모다.

중요한 포인트는:

- `control 6개`는 원래 모든 모델 input에 직접 들어가는 것이 아니었다
- `num_prev`, `num_pred`, `num_act`가 state 크기를 크게 바꾼다
- `pred`가 있으면 predictor rollout block이 state에 추가된다
- 최근에 추가한 `previous action`, `pressure diff velocity`는 **new Ours**에서만 직접 들어간다

### 11.1 공통 정의

`env4`에서 쓰는 기본 단위는 다음과 같다.

#### (A) 1-step base observation

기본 관측값 `base_obs_dim = 6`

구성:

1. chamber positive pressure
2. chamber negative pressure
3. actuator positive pressure
4. actuator negative pressure
5. displacement
6. displacement velocity

즉:

```text
base_obs_dim = 6
```

#### (B) 1-step reference

reference `dim_ref = 3`

구성:

1. reference actuator positive pressure
2. reference actuator negative pressure
3. reference displacement

즉:

```text
dim_ref = 3
```

#### (C) action dimension

full action `dim_act = 6`

구성:

1. chamber positive valve
2. chamber negative valve
3. actuator positive in
4. actuator positive out
5. actuator negative in
6. actuator negative out

즉:

```text
dim_act = 6
```

#### (D) history 관련 정의

`env4`는 다음 규칙을 쓴다.

```text
num_obs = num_prev + 1
num_ref = num_prev + num_pred + 1
```

설명:

- `num_prev = 10`이면 과거 10개 + 현재 1개를 보므로 `num_obs = 11`
- `num_pred = 15`이면 reference는 과거/현재/미래까지 모두 포함해서 `26`

즉:

```text
num_prev = 10  ->  num_obs = 11
num_prev = 0   ->  num_obs = 1

num_prev = 10, num_pred = 15  ->  num_ref = 26
num_prev = 0,  num_pred = 1   ->  num_ref = 2
```

#### (E) 최종 state 계산 공식

`env4` 최종 state 크기는 아래처럼 계산된다.

```text
dim_obs_traj = num_obs * dim_obs
dim_fut_traj = num_act * dim_obs        # pred != None 일 때만
dim_ref_traj = num_ref * dim_ref
dim_state    = dim_obs_traj + dim_fut_traj + dim_ref_traj
```

여기서:

- `dim_obs`는 "1-step observation 크기"
- old 모델은 `dim_obs = 6`
- new 모델은 `disp velocity`를 포함하면 `13`, 제외하면 `12`

이다.

---

### 11.2 SAC baseline state 구성

현재 `train4.py`의 SAC mode는:

```text
num_prev = 0
num_pred = 1
num_act  = 1
pred     = None
include_displacement_velocity_obs = False
include_prev_action_obs = True
include_pressure_diff_velocity_obs = True
```

#### (A) 1-step observation 크기

new feature가 켜져 있으므로:

- base observation: `5`
- previous applied action: `6`
- pressure diff velocity: `1`

합치면:

```text
dim_obs = 5 + 6 + 1 = 12
```

#### (B) observation trajectory block

`num_prev = 0` 이므로:

```text
num_obs = num_prev + 1 = 1
dim_obs_traj = 1 * 12 = 12
```

#### (C) predictor block

SAC baseline은 `pred = None` 이므로:

```text
dim_fut_traj = 0
```

#### (D) reference trajectory block

```text
num_ref = num_prev + num_pred + 1 = 0 + 1 + 1 = 2
dim_ref_traj = 2 * 3 = 6
```

#### (E) 최종 state 크기

```text
dim_state = 12 + 0 + 6 = 18
```

즉 현재 `train4.py` 기준 **SAC baseline input dimension은 18차원**이다.

#### (F) 해석

SAC baseline은 아주 짧은 state를 쓴다.

- 현재 observation 1개
- predictor 없음
- reference 2 step

그래서 정보량은 적지만, 그만큼 학습 공간은 비교적 작다.

---

### 11.3 기존 Ours state 구성 (old Ours)

여기서 말하는 "기존 Ours"는
[0625_lib4_Ours/cfg.yaml](/home/rise/test/brl_window_v2/pneu_rl/models/0625_lib4_Ours/cfg.yaml)
같은 저장된 모델이 사용한 구조를 뜻한다.

이 cfg에서는:

```text
num_prev = 10
num_pred = 15
num_act  = 5
pred != None

include_prev_action_obs = False
include_pressure_diff_velocity_obs = False
```

즉 **old Ours는 history는 길지만, previous action과 pressure diff velocity는 직접 안 넣었다.**

#### (A) 1-step observation 크기

old Ours는 base observation만 사용:

```text
dim_obs = 6
```

구성:

1. chamber pos pressure
2. chamber neg pressure
3. actuator pos pressure
4. actuator neg pressure
5. displacement
6. displacement velocity

여기서 주의:

- `제어 6개(control 6)`는 old Ours input에 직접 포함되지 않았다
- 즉 "압력 4개 + 제어 6개 + 변위 1개" 구조가 아니었다
- 실제로는 `압력 4 + 변위 + 변위속도 = 6`

#### (B) observation trajectory block

```text
num_obs = num_prev + 1 = 10 + 1 = 11
dim_obs_traj = 11 * 6 = 66
```

#### (C) predictor trajectory block

`pred != None`, `num_act = 5` 이므로:

```text
dim_fut_traj = 5 * 6 = 30
```

이 block은 predictor rollout으로 생성된 미래 observation 5개를 붙인 것이다.

#### (D) reference trajectory block

```text
num_ref = num_prev + num_pred + 1 = 10 + 15 + 1 = 26
dim_ref_traj = 26 * 3 = 78
```

즉 reference는:

- 과거 reference
- 현재 reference
- 미래 reference

를 다 포함한다.

#### (E) 최종 state 크기

```text
dim_state = 66 + 30 + 78 = 174
```

즉 **기존 Ours input dimension은 174차원**이다.

#### (F) 해석

old Ours는:

- predictor가 있고
- reference horizon도 길고
- observation history도 길다

그래서 input이 이미 꽤 크다.

하지만 여전히 직접 없는 정보가 있었다.

- previous applied action
- pressure diff velocity

즉 history는 있었지만, **action-aware temporal clue**는 약했다.

---

### 11.4 새 Ours state 구성 (new Ours)

현재 `train4.py`에서 새로 학습하는 Ours는:

```text
num_prev = 10
num_pred = 15
num_act  = 5
pred != None

include_displacement_velocity_obs = False
include_prev_action_obs = True
include_pressure_diff_velocity_obs = True
```

즉 old Ours와 비교해서 바뀐 것은:

- previous applied action 6개 추가
- pressure diff velocity 1개 추가
 - displacement velocity 1개 제거

뿐이다.

#### (A) 1-step observation 크기

new Ours는:

- base observation: `5`
- previous applied action: `6`
- pressure diff velocity: `1`

이므로:

```text
dim_obs = 5 + 6 + 1 = 12
```

구성은 정확히 다음이다.

1. chamber pos pressure
2. chamber neg pressure
3. actuator pos pressure
4. actuator neg pressure
5. displacement
6. previous chamber pos command
7. previous chamber neg command
8. previous actuator pos in command
9. previous actuator pos out command
10. previous actuator neg in command
11. previous actuator neg out command
12. pressure difference velocity  
   (`d/dt (act_pos_pressure - act_neg_pressure)`)

#### (B) observation trajectory block

```text
num_obs = 10 + 1 = 11
dim_obs_traj = 11 * 12 = 132
```

#### (C) predictor trajectory block

`pred != None`, `num_act = 5` 이므로:

```text
dim_fut_traj = 5 * 12 = 60
```

즉 predictor rollout도 old Ours의 `5 x 6`이 아니라  
new Ours에서는 `5 x 12`로 커진다.

#### (D) reference trajectory block

reference 쪽은 old Ours와 동일:

```text
num_ref = 26
dim_ref_traj = 26 * 3 = 78
```

#### (E) 최종 state 크기

```text
dim_state = 132 + 60 + 78 = 270
```

즉 **새 Ours input dimension은 270차원**이다.

#### (F) old Ours와 new Ours 차이

old Ours:

```text
174 dims
```

new Ours:

```text
270 dims
```

증가량:

```text
270 - 174 = 96
```

이 `96`은 어디서 생기나?

- observation trajectory 쪽 증가:
  - old: `11 * 6 = 66`
  - new: `11 * 12 = 132`
  - 증가: `66`

- predictor trajectory 쪽 증가:
  - old: `5 * 6 = 30`
  - new: `5 * 12 = 60`
  - 증가: `30`

합치면:

```text
66 + 30 = 96
```

즉 new Ours는 old Ours보다

- 현재/과거 observation window 전체에 대해 richer feature를 넣고
- predictor rollout block에도 같은 feature 확장을 적용한 것

이다.

---

### 11.5 왜 "num_prev=10이 있는데도 new feature를 추가하냐?"

이 질문이 제일 중요하다.

`num_prev=10`이 있으면 분명히 과거 정보는 본다.
하지만 그건 **observation history**다.

old Ours가 보던 것은:

- 과거 pressure
- 과거 displacement
- 과거 displacement velocity

였다.

그런데 old Ours가 직접 못 보던 것은:

- 직전에 어떤 action을 넣었는지
- pressure difference가 지금 어느 방향으로 얼마나 빨리 움직이고 있는지

였다.

즉:

- `num_prev=10`은 "과거 상태"를 보는 것
- `previous action`은 "과거 입력"을 보는 것
- `pressure diff velocity`는 "상태 변화 방향"을 명시적으로 주는 것

이다.

이 셋은 겹치는 듯 보여도 역할이 다르다.

그래서 new Ours는 history를 새로 만드는 게 아니라,

> 기존 history 위에 action-aware, derivative-aware clue를 추가하는 것

이라고 이해하는 게 맞다.

---

### 11.6 한눈에 보는 비교표

#### SAC baseline

- `dim_obs = 12`
- `dim_obs_traj = 1 * 12 = 12`
- `dim_fut_traj = 0`
- `dim_ref_traj = 2 * 3 = 6`
- `dim_state = 18`

#### old Ours

- `dim_obs = 6`
- `dim_obs_traj = 11 * 6 = 66`
- `dim_fut_traj = 5 * 6 = 30`
- `dim_ref_traj = 26 * 3 = 78`
- `dim_state = 174`

#### new Ours

- `dim_obs = 12`
- `dim_obs_traj = 11 * 12 = 132`
- `dim_fut_traj = 5 * 12 = 60`
- `dim_ref_traj = 26 * 3 = 78`
- `dim_state = 270`

---

### 11.7 practical takeaway

이 숫자들이 말해주는 건 단순하다.

- SAC baseline은 state가 작다
- old Ours는 history/pred/reference가 길다
- new Ours는 old Ours에 temporal proxy를 붙였지만, `disp velocity`는 빼서 현재 기본값 기준으로는 `270차원`이다

그래서 new Ours가 초반 학습이 느릴 수 있는 건 자연스럽다.  
반대로, 만약 new Ours가 나아진다면 그건 단순히 "history가 더 길어서"가 아니라

- previous action
- pressure diff velocity

같은 추가 정보가 실제로 partial observability를 완화했을 가능성을 뒷받침한다.
