# 밸브 동역학 확장과 MPObs/CAPS: 연구 흐름 및 증명 범위

작성: 2026-09-23. 현재 소스, 저장 cfg, 기존 분석 문서를 대조한 연구 메모다. 기존 학습 binary가 현재 소스와 일치한다는 보장은 없으며, 아래 수식은 명시한 조건에서의 유도이지 SAC 성능의 일반 정리가 아니다.

## 1. 연구 흐름을 어떻게 이해했는가

1. 연구실 선행 연구는 단일 펌프 양·음압 공급원을 모델링하고 SAC+MPObs+CAPS로 제어했다. Hyper-VAM 및 외부 부하 실험도 포함하지만, 중심 제어 목표는 압력이다.
2. 더 상세한 밸브 모델을 채택하고 개별 밸브 보정, 6-valve 공급원–actuator 전달계, 이후 가변부피·변위 모델로 확장했다. 동적 밸브 방정식 자체는 동료 연구의 기여로 구분돼 있다.
3. 확장된 plant에 기존 제어 구조를 적용했지만 tracking, 진동, chamber drift 문제가 남았다. History, action history, rate/reserve, action sequence를 도입했다.
4. 이후 검증은 단순 feature 추가에서 정보와 표현의 구분으로 진행됐다. Observability, supervised prediction, scale/placebo control, observation×reward 실험이 그 과정이다.
5. 현재 미완성 연결은 ‘모델 확장에서 어떤 변화가 기존 구성의 효용을 바꾸었는가’다. 차수, topology, 지연, 예측 초기화, loss 구현을 한 원인으로 묶으면 안 된다.

검토 문서:
- [research](research.md): 모델 식별과 controller 실패의 구분, 동료 연구와의 기여 경계.
- [밸브 정리](solvalve.md), [lib2 기록](ablation/lib2/README.md): 밸브 구조 및 reference confound.
- [lib3 실험 흐름](ablation/lib3_0807/README.md): 초기 제어기와 feature 도입 동기.
- [observability](ablation/lib3_0807/lib3_nonlinear_observability_analysis.md), [내부 상태 비교](ablation/lib3_0807/lib3_observation_set_observability_analysis.md).
- [예측 근거](ablation/lib3_0807/observation_predictive_evidence.md), [좌표 비교](ablation/lib3_0807/observation_coordinate_control_experiment.md), [reward 2×2](ablation/lib3_0807/history_rate_reward_2x2.md).
- [후속 structured-history 분석](ablation/lib3_0807/structured_history_screen_results.md), [lib4 결과](ablation/lib4/results.md).

문서의 초기 해석을 확정 사실로 취급하지 않는다. 예를 들어 K=2는 command hold가 아니라 서로 다른 두 action의 open-loop sequence이며, archive에는 interaction budget 차이도 있다. 기존 H20 분석도 full rank 증명이 아니라 특정 operating point에서 20–21/25 effective rank라는 결과다.

## 2. lib1 수식에 따른 정확한 비교

사용자 지정에 따라 기존 모델은 `pneu_env/src/pneu_env/lib/pneumatic_CT.cpp`의 수식으로 확인했다. 파라미터 값의 우열은 비교하지 않는다. `tune_lib1_valve_linear.py`도 이 모델을 lib1로 부른다.

기존 밸브는 기호적으로 다음과 같다.

\[
q=g(P_u,P_d,u)
 =c\,[a u+b\Delta P-d]_+\Phi(P_u,P_d).
\]

밸브 내부 미분방정식이 없는 **정적/준정적 유량 모델**이다. 선형인 부분은 opening의 입력–압력 관계이고, 전체 유량식은 clipping과 압축성 유동 때문에 비선형이다. 따라서 ‘1차 동적 밸브’와 구분해야 한다. 압력은 질량보존식의 1차 미분방정식으로 진화하지만 펌프까지 포함한 전체 시스템이 1차라는 뜻도 아니다.

현재 lib2는 이미 두 밸브에 z, q=x1, v=x2 및 command/branch memory가 있다. lib3는 같은 종류의 동적 밸브를 6개로 확장하고 고정부피 actuator pressure 두 개를 추가한다. lib4는 여기에 length, velocity, variable volume, leakage 등의 결합이 들어간다.

현재 `pred2.py`는 lib2 binary를 우선 로드하고 없으면 lib로 fallback한다. 따라서 파일명 pred2만으로 선행 정적 모델인지 판정하면 안 된다. 재현 실험에서는 실제 로드 경로와 source/binary hash를 기록해야 한다.

## 3. 증명 후보 A: 짧은 horizon에서 입력 효과가 나타나는 차수

비포화·고정 branch, 일정 부피, 한 유량 경로, 동일한 초기 상태를 가정한다. 펌프와 다른 경로는 이 국소 비교에서 고정한다. 단위 변환을 포함한 압력–유량 계수를 a, 정적 유량의 입력 미분을 b라 두고 선형화한다.

\[
\dot p=a q,\qquad
\dot q=v,\qquad
\dot v=-\omega_n^2q-2\zeta\omega_n v+\omega_n^2 b u.
\]

작은 일정 입력 변화 δu에 대한 초기 pressure 차이는:

| 밸브 모델 | 짧은 시간 h 뒤의 입력 유발 압력 차이 |
|---|---|
| 정적 q=bu | ab δu h + O(h²) |
| 1차 lag τ qdot+q=bu (진단용 중간 모델) | ab δu h²/(2τ) + O(h³) |
| 2차 lag | ab ω_n² δu h³/6 + O(h⁴) |

따라서 이 국소 모델에서는 static→second-order 변경으로 input-to-pressure relative degree가 1→3이 된다. 입력을 바꾸어도 짧은 예측 구간의 압력 차이는 작아질 수 있다. 이는 ‘MPObs가 안 된다’가 아니라 **고정된 짧은 horizon이 새 입력 효과를 충분히 드러내지 못할 조건**을 제공한다.

반면 초기 유량 오차 δq와 유량 변화율 오차 δv는

\[
\delta p(h)=a\delta q\,h+\tfrac12a\delta v\,h^2+O(h^3)
\]

로 나타난다. 여기서는 초기 pressure가 같고 위 단순 모델을 가정한다. 다른 pressure-dependent 경로가 있으면 h² 이상의 계수에 추가항이 생긴다.

핵심: **숨은 초기 유량의 영향은 O(h), 새 명령의 영향은 O(h³)**일 수 있다. 정확한 dynamics를 쓰더라도 잘못된 초기 유량을 사용하면 짧은 MPObs가 명령 효과보다 초기화 오차를 더 크게 반영할 수 있다. δq=0이면 해당 leading term은 사라지므로 보편적인 열화 주장은 아니다.

실제 구현은 hysteresis jump, clipping, RK stage 내 valve update를 포함한다. 위 연속시간 유도를 actual discrete map의 sensitivity로 검증해야 한다. 공급원 밸브→actuator 경로와 직접 actuator 밸브 경로도 분리한다.

## 4. 증명 후보 B: 초기 상태 불확실성의 출력 오차

\[
Y_H=G_H(x,U),\qquad \hat Y_H=G_H(\hat x,U).
\]

동일 dynamics와 입력열에서

\[
Y_H-\hat Y_H\approx J_{x,H}(x-\hat x).
\]

초기화 오차의 평균 μ와 covariance Σ가 있으면 국소 선형 근사에서

\[
E\|Y_H-\hat Y_H\|^2\approx
\|J_{x,H}\mu\|^2+\operatorname{tr}(J_{x,H}\Sigma J_{x,H}^{T}).
\]

이 식은 **state 수가 아니라, 잃어버린 state의 불확실성과 output sensitivity**가 중요함을 보여준다. 측정되지 않는 상태가 늘어도 output에 영향이 없으면 문제는 늘지 않는다.

또 동일 predictor 입력으로 구분되지 않는 두 허용 상태에서 출력이 Ya,Yb이면, 그 입력만 사용하는 결정론적 predictor에는 삼각부등식으로

\[
\max(\|Y^a-\hat Y\|,\|Y^b-\hat Y\|)\ge\tfrac12\|Y^a-Y^b\|
\]

가 성립한다. 이는 압력만 받는 predictor의 조건부 한계다. policy에 history가 있다는 사실과 predictor 초기화가 history를 실제로 활용하는지는 별개다. 근사적으로 같은 pressure pair에는 추가적인 연속성/오차 허용 분석이 필요하다.

### 현재 코드와 연결

`env3.predict_obs → pred3.set_init_press → set_init_env_act`는 매 prediction마다 네 압력을 주입하면서 valve states를 reset하고 pump pressure/phase도 재구성한다. 즉 같은 식을 쓴다는 사실만으로 exact prediction이 되지 않는다. lib4는 length/velocity도 초기화에 전달하지만 valve memory 문제가 자동으로 해결되지는 않는다.

작은 확인 실험을 현재 lib3 소스를 /tmp에 별도로 컴파일해 수행했다. 세 짧은 warmup 상태에서 동일한 25-step 미래 명령을 주고 raw pressure를 비교했다. 센서 지연·noise·RL 학습은 제외했다.

| 초기화 | 0.02 s | 0.10 s | 0.20 s | 0.50 s |
|---|---:|---:|---:|---:|
| 전체 상태 복제 | 0 | 0 | 0 | 0 |
| 압력만 유지한 기존 reset | 0.610 | 1.315 | 1.068 | 0.828 |
| valve runtime만 reset | 0.347 | 1.153 | 0.987 | 0.548 |
| pump pressure/phase만 reset | 0.460 | 0.700 | 0.705 | 0.687 |

단위 kPa. 각 시점 네 압력 endpoint RMS를 구한 뒤 세 초기 상태에서 평균했다. trajectory RMSE나 논문용 성능 비교가 아니다. 오차들은 비선형 상호작용하므로 행들을 합할 수 없다. 결과 파일: `/tmp/brl_research_audit/mpobs_reset_pilot.json`.

이 검사는 초기화만 바꿔도 오차가 발생함을 확인한 것이며, 실제 학습 실패의 크기·원인 비중 또는 lib1 대비 차이를 증명한 것은 아니다.

## 5. CAPS는 먼저 실제 loss를 분리해야 한다

CAPS 원문은 Euclidean distance를 사용한다: [저자 설명](https://ai.bu.edu/caps/). 붙여준 계획의 squared loss와 현재 구현을 동일시하지 않는다.

현재 `sac_with_loss.py` legacy 경로는 독립 stochastic action samples 사이의 batch 전체 L2 norm을 계산한다. mean_action 경로는 deterministic actor 출력의 per-sample L2 norm 평균이다. 둘 다 squared L2가 아니다.

저장된 260808 history_caps 및 history_mpobs1_caps cfg는 `caps_mode=legacy`, `noise_std=0`, `normalize_observation=false`다. 따라서 spatial 입력은 원래 입력과 같은데 별도 action sample을 비교한다. 현재 코드와 이 cfg의 조합에서는 spatial 항이 observation-neighborhood sensitivity보다 stochastic action dispersion을 벌점으로 준다.

같은 s에서 독립적으로 뽑은 a,a'에 대해 정확히

\[
E\|a-a'\|^2=2\operatorname{tr}\operatorname{Cov}[a\mid s].
\]

이 식은 **실제 unsquared loss의 제곱의 기대값**에 대한 항등식이지 실제 loss의 기대값과 같지 않다. 다만 같은 상태에서도 벌점이 생기고 dispersion에 의존한다는 원인은 명확하다.

배치 B, action dimension m에서 성분별 동일 평균 제곱 차이가 σ_d²이면 batch norm의 RMS는 √(Bm)σ_d다. 그러므로 batch/action dimension 변경 시 같은 λ가 같은 regularization 세기를 의미하지 않는다. SAC entropy/Q scale도 함께 변하므로 이 scaling만으로 최종 성능을 예측하지 않는다.

**따라서 기존 CAPS 결과를 곧바로 ‘2차 plant에 필요한 transient sensitivity를 억제했다’는 증거로 쓰지 않는다.** historical binary/code 재현, deterministic mean 기준, reduction/scale, temporal/spatial 분리부터 필요하다.

## 6. 증명 후보 C: command smoothing과 tracking의 조건부 trade-off

국소 LTI plant G, 일정 주파수 Ω의 command U와 reference R, squared temporal-difference penalty를 사용하는 이상화 문제를 두자. d(Ω)=exp(jΩΔt)-1이다.

\[
\min_U |GU-R|^2+\lambda |dU|^2
\quad\Rightarrow\quad
U^*=\frac{G^*R}{|G|^2+\lambda|d|^2}.
\]

따라서 reference 대비 출력 비율은 |G|²/(|G|²+λ|d|²)이다. λ가 고정일 때 해당 주파수에서 plant gain이 작으면 tracking attenuation이 커진다.

단순 2차 valve filter는

\[
H_v(s)=\frac{\omega_n^2}{s^2+2\zeta\omega_ns+\omega_n^2}
\]

이며 일정 구간의 gain 감소·phase lag를 계산할 수 있다. 그러나 underdamped resonance 부근에서는 gain이 증가할 수도 있다. ‘2차이므로 모든 주파수에서 smoothing이 더 나쁘다’는 결론은 성립하지 않는다.

이 유도는 frequency-domain의 squared smoothing surrogate이다. 현재 unsquared CAPS와 entropy-regularized SAC의 정리로 사용하면 안 된다. 실제 CAPS에서는 계수 sweep, gradient 크기/방향, tracking–smoothness curve로 연결을 확인한다. Spatial CAPS의 E||Jεπ||²=tr(JΣJᵀ)도 squared surrogate에만 직접 적용한다.

## 7. 추가 실험의 최소 순서

1. **같은 lib3 topology에서 valve dynamics만 교체.** 같은 정적 유량 map을 유지한 memoryless/1차 lag/2차 lag를 비교한다. 이것이 차수의 인과 효과 비교다. lib1의 다른 static map을 그대로 이식한 비교는 별도로 둔다. branch/hysteresis는 먼저 고정하고 이후 추가한다. gain/time-scale matching 기준도 명시한다.
2. **학습 없이 초기화×horizon 진단.** Oracle clone, pressure reset, valve-only reset, pump-only reset을 같은 reachable state bank와 미래 입력열에서 비교한다. raw output부터 확인한 후 동일한 sensor delay/time alignment를 적용한다. 기존 0.02 s MPOBS1뿐 아니라 긴 horizon의 이득과 누적 오차를 함께 본다.
3. **출력의 상태 민감도와 입력 민감도 비교.** Jx,H와 JU,H를 계산하되 raw norm의 단위 차이를 제거한다. 물리적으로 정한 state uncertainty와 admissible input perturbation을 사용해 두 효과를 pressure 단위로 비교한다. 한 operating point가 아니라 실제 운전 상태 bank에서 평가한다.
4. **CAPS 구현과 모델 효과 분리.** legacy 재현과 mean-action baseline을 분리하고 temporal/spatial를 각각 sweep한다. 동일 action dimension·batch·관측 scale로 valve-order 조건을 비교한다.
5. **핵심 RL 비교만 수행.** 위 진단에서 살아남은 가설로 history, history+rate, 초기화가 정렬된 MPObs를 같은 physical budget/reward/reference에서 비교한다. 2→6 valve topology 효과와 lib4 운동 결합은 이후 별도 축으로 추가한다.

기존 history×reward 2×2를 반복하거나 feature를 더 붙이는 것이 우선은 아니다. 기존 실험은 representation 효과의 근거로 보존하고, 새 실험은 ‘왜 모델 확장 이후 차이가 생겼는가’를 겨냥한다.

## 8. 현재 주장할 수 있는 문장

> 정적 유량 모델에서 동적 밸브 모델로 확장하면, 짧은 미래 압력에서 새 입력과 숨은 초기 유량 상태가 기여하는 방식이 달라진다. 본 연구는 이 차이가 압력 기반 초기화와 고정 horizon의 MPObs에 미치는 영향을 정량화하고, 실제 smoothing objective를 분리해 관측 표현 및 regularization의 유효 조건을 검증한다.

이는 검증할 연구 문장이다. ‘고차원 시스템에서는 MPObs/CAPS가 실패한다’, ‘rate가 숨은 상태를 모두 복원한다’, ‘oracle MPObs가 학습 성능 upper bound다’는 현재 증명되지 않는다. Oracle은 예측 초기화 기준점이며 유한 학습 성능의 보장된 상한은 아니다.
