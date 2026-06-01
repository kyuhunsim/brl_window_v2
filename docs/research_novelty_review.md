# 연구 방향 및 Novelty 점검

작성 기준일: 2026-05-31

## 1. 결론

이 연구는 충분히 진행할 가치가 있다. 다만 현재 상태 그대로 논문을 구성하면
`기존 시스템을 복잡하게 확장하고 RL을 적용한 연구`로 보일 위험이 있다.

이 연구의 핵심 novelty는 RL, SAC, Runge-Kutta integrator 자체가 아니다.
가장 강한 중심축 후보는 다음과 같다.

> Static pressure matching만으로는 soft pneumatic actuator의 closed-loop 성능을
> 보장할 수 없다. 주파수 의존 밸브 동특성, chamber reserve, actuator geometry가
> 결합되면 작은 mismatch가 drift와 valve conflict로 증폭된다. 본 연구는 이 실패를
> 측정하고, 모델링하며, 구조적으로 줄이는 방법을 제시한다.

좋은 논문으로 만들기 위해서는 `더 복잡한 simulator를 만들었다`가 아니라
`기존 평가 방식이 놓치던 control-relevant mismatch를 정의하고 검증했다`를
주장해야 한다.

## 2. 선행 연구와 현재 연구의 차이

가장 가까운 선행 연구는 다음 논문이다.

- `Modeling and Reinforcement Learning-Based Control of Simultaneous Positive and Negative Pressure Generation in Pneumatic Systems`
- 파일: `docs/Modeling_and_Reinforcement_Learning-Based_Control_of_Simultaneous_Positive_and_Negative_Pressure_Generation_in_Pneumatic_Systems.pdf`

선행 논문은 단일 펌프 기반 양압/음압 chamber를 2개 밸브로 제어한다.
또한 이미 다음 요소를 사용한다.

- 4th-order Runge-Kutta 기반 simulator
- SAC 기반 RL controller
- Model-Predicted Observation (MPObs)
- CAPS 기반 action smoothing
- 양압/음압 coupled dynamics 모델링

현재 연구는 다음 방향으로 확장되고 있다.

| 구분 | 선행 연구 | 현재 연구 |
|---|---|---|
| 제어 대상 | 양압/음압 chamber | chamber + soft actuator |
| 밸브 수 | 2 | 6 |
| 밸브 모델 | 비교적 단순한 정적 모델 | hysteresis + nonlinear area + dynamic response |
| actuator 모델 | 부하 실험 중심 | soft actuator geometry, volume, displacement 모델링 |
| 제어 목표 | pressure tracking | actuator pressure tracking, 최종적으로 displacement tracking |
| 중요 내부 상태 | chamber pressure | chamber reserve, actuator pressure, displacement, valve state |

즉, 현재 연구는 단순한 채널 추가가 아니라 `pressure source control`에서
`soft actuator closed-loop control`로 문제 자체가 확장되는 과정이다.

## 3. 현재까지 잘한 부분

### 3.1 실험 인프라를 실제로 구축했다

아이디어 수준이 아니라 다음 파이프라인이 존재한다.

1. RT/FPGA 및 TCP/IP 기반 장비 제어
2. real CSV 데이터 취득
3. 6채널 유량 데이터 수집
4. 밸브 및 펌프 파라미터 튜닝
5. simulator replay를 통한 sim-real 비교
6. simulator 기반 RL 학습
7. quick visualization 및 PID baseline 비교

운영 흐름은 `docs/goal.md`에 정리되어 있다.

### 3.2 밸브 모델을 의미 있게 고도화했다

기존 선형 정적 밸브 모델 대신 다음 요소를 추가했다.

- Bouc-Wen hysteresis
- nonlinear S-curve effective area
- compressible flow
- opening/closing 비대칭 2차 dynamic response

관련 정리는 `docs/solvalve.md`, 구현은
`pneu_env/src/pneu_env/lib3/pneumatic_CT.cpp`에 있다.

### 3.3 실패를 단순히 성능 저하로 보지 않았다

`pneu_rl/exp/260528_rl_vs_pid_failure_analysis.md`에는 다음 문제가 분리되어 있다.

- train/eval PID mismatch
- chamber drift
- raw 6D action ambiguity
- in/out valve simultaneous opening conflict
- nonlinear valve opening range
- short-horizon training과 long rollout 사이의 차이

이 분석은 중요하다. 좋은 논문은 `성능이 낮았다`가 아니라
`어떤 조건에서 왜 실패하는가`를 설명할 수 있어야 한다.

### 3.4 실패해도 남는 자산이 있다

최종 알고리즘이 예상보다 강하지 않더라도 다음 결과는 남는다.

- 6-valve pneumatic soft actuator dataset
- dynamic valve identification protocol
- simulator 및 replay pipeline
- failure taxonomy
- PID/RL baseline
- frequency-aware validation protocol

이는 `docs/do_not_forget.html`에서 말한
`hypothesis risk를 infrastructure value로 hedge한다`는 기준에 부합한다.

## 4. Novelty로 보기 어려운 요소

다음 요소는 필요하지만 논문의 중심 기여로 제시하기 어렵다.

- SAC 또는 RL을 사용했다.
- Runge-Kutta integrator를 사용했다.
- 밸브 수를 2개에서 6개로 늘렸다.
- optimizer 또는 튜닝 방법을 바꿨다.
- Hyper-VAM 수식을 simulator에 추가했다.
- PID보다 RMSE가 조금 낮아졌다.

특히 현재 C++ simulator의 적분기는 RK 3/8-rule 형태이며 구현상 의미는 있지만,
numerical integrator 선택 자체는 novelty가 아니다.

핵심은 기존 도구를 사용했는지가 아니라 다음 질문에 답하는 것이다.

> 기존 방식은 정확히 어떤 실패를 보지 못했으며, 왜 soft actuator closed-loop에서
> 그 실패가 중요해지는가?

## 5. 가장 강한 연구 질문 후보

### 5.1 Control-relevant fidelity

단순 pressure replay RMSE가 낮은 simulator라도 실제 controller 학습에는 충분하지
않을 수 있다. 작은 mismatch가 closed-loop에서 누적되거나 증폭될 수 있기 때문이다.

검증할 질문:

- static fitting이 좋아도 chirp 응답이 틀리면 RL policy가 실패하는가?
- pressure RMSE가 비슷한 두 simulator가 서로 다른 policy failure mode를 만드는가?
- valve hysteresis를 제거하면 어떤 주파수와 trajectory에서 성능이 깨지는가?
- actuator variable volume을 제거하면 displacement tracking에서 어떤 오차가 생기는가?

### 5.2 Reserve-aware control

RL은 actuator pressure tracking만 최적화하면 chamber pressure를 비용 없는 내부 자원처럼
사용할 수 있다. 실제 분석에서도 positive chamber가 과도하게 상승하는 문제가 확인되었다.

검증할 질문:

- actuator RMSE가 낮아도 chamber reserve가 붕괴하는가?
- 짧은 episode에서는 좋아 보이지만 긴 rollout에서는 drift가 생기는가?
- reserve constraint를 추가하면 tracking, smoothness, safety 사이의 trade-off가 어떻게 바뀌는가?

### 5.3 Structured action allocation

raw 6D SAC는 pressure control과 valve allocation을 동시에 학습해야 한다.
이때 in/out 밸브를 함께 여는 conflict가 생긴다.

검증할 질문:

- raw 6D action에서 conflict가 구조적으로 반복되는가?
- conflict-free allocator를 사용하면 더 작은 action space로 같은 tracking 성능을 얻는가?
- policy가 직접 6개 밸브를 제어하는 대신 pressure-rate 또는 reserve command를 내리면
  sim-to-real robustness가 좋아지는가?

## 6. 현재 구현 상태와 부족한 부분

### 6.1 구현된 부분

현재 추적 가능한 `lib3`에는 다음 요소가 있다.

- 6개 밸브 입력
- chamber 및 actuator pressure state
- Bouc-Wen hysteresis
- nonlinear effective area
- 2차 valve dynamics
- mass flow debug 정보
- MPObs 기반 예측 경로
- PID baseline
- RL reward의 action delta, valve conflict, chamber reserve 항

### 6.2 아직 구현되지 않은 핵심

`docs/5538b4a2-5c1f-441f-b8da-f2f264f83734_Modeling.pdf`에는 다음 모델이 정리되어 있다.

- displacement에 따른 `V_pos(L)`, `V_neg(L)`
- `dV/dL`
- pressure-volume coupling
- pressure에 따른 force
- rod displacement dynamics

그러나 현재 `lib3` 구현에서 actuator volume은 고정값이며 displacement state는 없다.
따라서 최종 목표인 displacement control은 아직 닫히지 않았다.

### 6.3 simulator 계보가 정리되어야 한다

`docs/goal.md`에는 `lib9` 기반 메인 트랙이 적혀 있지만 현재 repository에는
`sim9.py`, `env9.py`, `pred9.py`, `lib9`가 없다. 반면 `gen_sim9_from_real_csv.py`,
`real9.py`는 존재한다.

논문 실험 전에는 simulator 버전, 튜닝 결과, 빌드 결과, RL 모델의 대응관계를
명시적으로 고정해야 한다.

권장 기록 형식:

```text
(real CSV)
-> (identification config/result)
-> (simulator model version)
-> (shared library build hash)
-> (RL config/model)
-> (evaluation result)
```

### 6.4 RK integration과 valve internal state를 점검해야 한다

현재 밸브 내부 상태 `z`, `x1`, `x2`는 C++ 밸브 함수 내부에서 갱신된다.
논문용 simulator로 사용하려면 다음 중 하나가 필요하다.

1. valve state를 전체 ODE state에 명시적으로 포함한다.
2. multirate approximation으로 유지하되 근사 이유와 timestep convergence를 검증한다.

## 7. 추천하는 실험 구조

### 7.1 Simulator ablation

모델 계보를 명확하게 나눈다.

| 모델 | 구성 |
|---|---|
| M0 | linear static valve |
| M1 | nonlinear static valve |
| M2 | M1 + hysteresis |
| M3 | M2 + dynamic response |
| M4 | M3 + actuator `V(L)` and displacement dynamics |

각 모델을 동일한 held-out 데이터에서 비교한다.

### 7.2 평가 trajectory

- static step
- opening/closing hysteresis sweep
- chirp input
- random control replay
- long-horizon rollout
- 다양한 payload
- displacement reference tracking

### 7.3 평가 metric

압력 RMSE 하나로 끝내지 않는다.

- pressure RMSE
- net-flow RMSE
- frequency response gain error
- frequency response phase error
- long-horizon drift
- reserve violation duration
- reserve violation magnitude
- in/out valve conflict
- action variation
- displacement RMSE
- sim-to-real policy degradation

### 7.4 Controller baseline

- PID
- DOB 또는 기존 선행 방식
- raw SAC
- SAC + MPObs
- SAC + CAPS
- raw 6D RL + reserve penalty
- structured RL + conflict-free allocator

## 8. 추천 controller 구조

raw 6D RL을 최종 구조로 고집하지 않는 편이 좋다.

권장 구조:

```text
observation
-> policy
-> low-dimensional pressure-rate / reserve command
-> conflict-free allocator
-> 6 valve commands
```

이 구조는 단순한 engineering trick이 아니다. 다음 실패 원인에서 직접 도출된다.

```text
raw 6D action
-> redundant valve combinations
-> simultaneous in/out opening
-> wasted flow, oscillation, reserve misuse
-> long-horizon degradation
```

따라서 allocator의 필요성을 ablation으로 설명할 수 있다.

## 9. 논문용 contribution 초안

아래 세 가지 정도로 압축하는 것이 좋다.

1. **Failure definition and evaluation**
   - soft pneumatic actuator의 closed-loop에서 static sim-real pressure matching이 충분하지
     않음을 보이고, frequency response, reserve violation, valve conflict를 포함한
     control-relevant fidelity 평가축을 정의한다.

2. **Multiphysics simulator**
   - 6-valve pressure source와 soft actuator를 대상으로 hysteretic dynamic valve model,
     variable-volume actuator geometry, displacement dynamics를 결합한 simulator를 구축한다.

3. **Reserve-aware structured control**
   - chamber reserve 붕괴와 redundant valve action 문제를 줄이기 위해 reserve-aware
     objective와 conflict-free allocation 구조를 사용하고, pressure 및 displacement
     tracking에서 검증한다.

## 10. 우선순위

### Priority 0: 논문 질문 고정

논문의 중심을 `RL 적용`이 아니라 `control-relevant fidelity와 구조적 제어`로 고정한다.

### Priority 1: simulator 버전 고정

- 실제 메인 simulator 경로를 하나로 통합한다.
- `lib3`, `lib9` 명칭 혼선을 제거한다.
- 튜닝 결과가 RL에서 실제로 로드되는지 자동 검증한다.

### Priority 2: 모델 ablation 완성

- M0-M3 비교를 먼저 닫는다.
- chirp, replay, long rollout metric을 자동 생성한다.
- 압력 RMSE가 비슷해도 policy 결과가 달라지는 사례를 찾는다.

### Priority 3: action structure 수정

- raw 6D SAC baseline을 유지한다.
- conflict-free allocator 버전을 추가한다.
- tracking, conflict, reserve violation을 동시에 비교한다.

### Priority 4: soft actuator displacement 모델 구현

- `V(L)`, `dV/dL`, force, rod dynamics를 simulator state에 추가한다.
- 실제 displacement sensor 데이터로 검증한다.
- 최종적으로 position tracking 실험을 수행한다.

## 11. 30초 설명

> 기존 연구는 양압과 음압 source의 pressure tracking을 다뤘지만, soft actuator를 연결하면
> 6개 밸브, 저장압, 밸브 히스테리시스, actuator volume 변화가 결합됩니다.
> 이때 static sim-real matching이 좋아도 closed-loop policy는 drift와 valve conflict로
> 실패할 수 있습니다. 저희는 이 control-relevant mismatch를 정량화하고,
> dynamic simulator와 reserve-aware structured controller로 줄입니다.

## 12. 마지막 판단

현재 연구는 선행 논문의 단순 반복은 아니다. 그러나 복잡성 자체가 novelty가 되지는 않는다.

좋은 논문으로 가기 위한 핵심은 다음과 같다.

> 더 많은 요소를 넣었다고 주장하지 말고, 기존 연구가 측정하지 못한 실패를 정의하고,
> 그 실패가 왜 발생하는지 보이며, 각 설계가 그 원인에서 필연적으로 도출되도록 만든다.

