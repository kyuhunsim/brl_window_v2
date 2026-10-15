# Lib3 구조화 history ablation: rate, reserve, K=2 action sequence

> Observability/POMDP를 우리 6-input, 4-pressure plant에 적용한 슬라이드용
> 정리는 [observability_pomdp_system_note.md](observability_pomdp_system_note.md)를
> 참고한다.

## 이번 결과에서 말할 수 있는 것

- 현재 archive에서 직접 비교 가능한 것은 **H+rate (K=1)**, **HRR
  (history+rate+reserve, K=1)**, **HRR+K=2 action sequence**의 세 조건이다.
  네 번째 조건인 **H+rate+K=2** checkpoint는 없다. 따라서 지금 표는 완결된
  2x2 factorial ablation이 아니며, reserve 효과와 K=2 효과를 완전히
  분리해서 주장할 수는 없다.
- 현재 fixed reference bank에서는 reserve를 넣은 HRR(K=1)이 H+rate(K=1)
  보다 tracking과 command variation을 함께 개선한다.
- archive의 K=2는 가장 낮은 pressure RMSE와 plateau pressure motion을
  보인다. 그러나 K=2는 K=1보다 episode당 physical simulation step을 두 배
  사용했고, 실제 valve command total variation은 HRR(K=1)의 약 24배다.
  따라서 이를 “더 smooth한 controller” 또는 “진동 억제 성공”이라고
  부르면 안 된다.
- 과거 RandomRef는 단순히 변화 주기가 더 긴 reference가 아니다. 일반적인
  slope 분포는 현재 bank와 비슷하지만, offset/parameter 변화가 큰 target
  jump를 만든다. 그래서 nominal tracking과 분리한 OOD jump-recovery
  test로 사용한다.

## 비교 조건과 공정성

질문은 다음과 같다.

> pressure/action history를 고정했을 때, explicit pressure rate, chamber
> reserve, 그리고 K=2 action-sequence 표현이 tracking 및 진동 관련 지표에
> 무엇을 더하는가?

| 조건 | Rate | Reserve | K | archive 상태 | 학습 episode당 physical step |
|---|:---:|:---:|:---:|---|---:|
| H+rate (K=1) | yes | no | 1 | 있음 | 512 |
| HRR (K=1) | yes | yes | 1 | 있음 | 512 |
| H+rate (K=2) | yes | no | 2 | **없음** | — |
| HRR (K=2) | yes | yes | 2 | 있음 | 1,024 |

세 archive run의 reward는 동일한 historical pressure-tracking objective이며
chamber_bound_rwd_coeff = 0.0이다. 다만 1,000 episode 기준 K=2는
1.024 M physical step, K=1은 0.512 M physical step을 사용했다. 즉 현재
K=1/K=2 차이는 **표현 방식뿐 아니라 interaction budget도 다르다.**

### K=2는 command-hold가 아니다

이 부분이 가장 중요하다. K=2에서 policy는 한 개의 6-valve command를
두 step 동안 유지하는 것이 아니라, 한 번의 observation에서 서로 다른 두
action을 출력한다.

\[
\pi(o_t) = [u_t^{(0)}, u_t^{(1)}], \qquad
u_t^{(0)} \rightarrow \text{20 ms plant step}, \quad
u_t^{(1)} \rightarrow \text{next 20 ms plant step}.
\]

두 번째 action \(u_t^{(1)}\)은 첫 번째 action 이후의 새로운 measurement를
보고 만든 것이 아니다. 즉 K=2는 25 Hz feedback controller가 아니라,
**50 Hz로 적용되는 2-step open-loop action sequence**다. 그래서 서로 다른
두 action이 번갈아 나오면 command plot이 실제로 빽빽해진다.

현재 evaluation은 K=2의 두 physical substep을 모두 기록한다. 따라서 아래
plot의 조밀함은 CSV down-sampling/plot artifact가 아니라 실제 applied
command variation이다.

## 현재 fixed reference bank 결과

Protocol은 seed 0 final checkpoint, fixed reference bank의 16 member,
member당 2,048 **physical** simulator step(약 40.86 s)이다. K=2도
2,048 physical step까지만 평가했으므로 rollout 시간 자체는 K=1과 같다.

![학습 곡선 및 fixed-bank 결과](../../../pneu_rl/exp/lib3_0907_rate_reserve_chunk_analysis/rate_reserve_chunk_summary.png)

| Controller | Combined RMSE (kPa) | POS / NEG RMSE (kPa) | POS / NEG plateau rate RMS (kPa/s) | Command TV/channel (1/s) | Mean tolerance |
|---|---:|---:|---:|---:|---:|
| H+rate (K=1) | 3.922 +/- 0.654 | 4.252 / 3.555 | 4.916 / 13.783 | 0.203 +/- 0.043 | 0.702 |
| HRR (K=1) | 2.063 +/- 0.691 | 1.151 / 2.640 | 10.017 / 2.606 | **0.166 +/- 0.039** | 0.950 |
| HRR (K=2) | **0.764 +/- 0.034** | **0.685 / 0.831** | **2.213 / 1.131** | 3.936 +/- 0.402 | **0.999** |

여기서 +/-는 16개 deterministic reference member 사이의 standard deviation이다.
training seed uncertainty가 아니다. Command TV는 각 valve channel에 대해
실제로 plant에 적용된 command의 total variation을 시간으로 나눈 값이다.
K=1/K=2 모두 50 Hz physical time base에서 계산했으므로 직접 비교 가능하며,
낮을수록 command switching이 적다.

![공통 약 40초 pressure tracking](../../../pneu_rl/exp/lib3_0907_rate_reserve_chunk_analysis/representative_40s_tracking.png)

![실제로 적용된 6-valve command](../../../pneu_rl/exp/lib3_0907_rate_reserve_chunk_analysis/representative_40s_commands.png)

### 해석

1. **K=1에서 reserve는 유효하다.** H+rate에서 HRR로 바꾸면 combined RMSE는
   3.922에서 2.063 kPa로, command TV는 0.203에서 0.166/s로 감소한다.
   POS tracking은 크게 좋아진다. 다만 plateau motion은 NEG
   13.783 -> 2.606 kPa/s로 줄고 POS 4.916 -> 10.017 kPa/s로 늘어난다.
   즉 reserve는 K=1 tracking과 전체 command variation을 개선하지만, 양쪽
   chamber의 진동을 균일하게 damping했다는 결론은 아니다.

2. **K=2의 pressure 결과는 좋지만, actuator smoothness 결과는 아니다.**
   HRR(K=2)의 pressure RMSE와 pressure-rate RMS가 가장 좋다. 그러나
   command TV는 3.936/s로 HRR(K=1)의 0.166/s보다 **23.8배** 크다.
   pressure가 덜 흔들리는 대신 high-bandwidth valve switching으로
   상쇄했을 가능성이 있다. 실물 valve bandwidth, delay, wear를 고려하면
   오히려 위험한 정책일 수 있다.

3. **학습 곡선에서 K=2를 sample efficient라고 말할 수 없다.** 그래프의
   x축은 physical simulator step이며, reward는 discounted physical reward
   count로 정규화했다. K=2는 약 1.024 M step까지, K=1은 약 0.512 M
   step까지만 학습했다. 같은 episode 수가 같은 data budget을 뜻하지 않는다.

원자료: [fixed-bank aggregate](../../../pneu_rl/exp/lib3_0907_rate_reserve_chunk_analysis/fullbank_metrics.csv),
[member별 지표](../../../pneu_rl/exp/lib3_0907_rate_reserve_chunk_analysis/per_member_metrics.csv),
[분석 script](../../../pneu_rl/scripts/analyze_lib3_rate_reserve_chunk_ablation.py).

## 과거 RandomRef: jump robustness 보조 결과

과거 RandomRef는 1–5 s 간격으로 period, amplitude, offset이 변한다. 보통
구간의 95-percentile reference slope는 현재 smooth capacity bank와 크게
다르지 않다. 그러나 parameter/offset이 바뀌는 순간 discontinuous target
jump가 생기며, 네 reference seed(61098–61101)에서 평균 peak absolute
reference rate는 521.4 kPa/s였다.

따라서 이것은 “과거 reference가 더 느리거나 주기가 더 길다”를 검증하는
test가 아니라, **갑작스러운 set-point jump 이후 tracking을 회복하는가**를
보는 OOD test다.

Protocol은 각 controller와 legacy reference seed마다 deterministic 약 40 s
rollout 하나이다. 아래 +/-는 reference seed 간 variation이며, training seed
confidence interval이 아니다.

![과거 RandomRef jump-robustness 결과](../../../pneu_rl/exp/lib3_0907_rate_reserve_chunk_legacy_random_analysis/legacy_random_summary.png)

| Controller | Combined RMSE (kPa) | Mean tolerance | POS / NEG plateau rate RMS (kPa/s) | Command TV/channel (1/s) |
|---|---:|---:|---:|---:|
| H+rate (K=1) | 3.953 +/- 0.339 | 0.862 | 14.614 / 12.092 | **0.219 +/- 0.063** |
| HRR (K=1) | 3.424 +/- 0.455 | 0.915 | 23.469 / 8.849 | 0.307 +/- 0.052 |
| HRR (K=2) | **2.831 +/- 0.392** | **0.969** | **9.858 / 5.591** | 2.407 +/- 0.388 |

과거 reference에서도 HRR과 K=2가 jump 후 pressure tracking을 개선한다.
하지만 HRR(K=2)의 command TV는 HRR(K=1)의 약 7.9배다. 이 결과는
jump-recovery 보조 결과로만 두고, smooth actuation 또는 real robustness의
근거로 사용하지 않는다.

원자료: [legacy aggregate](../../../pneu_rl/exp/lib3_0907_rate_reserve_chunk_legacy_random_analysis/legacy_random_metrics.csv),
[reference seed별 지표](../../../pneu_rl/exp/lib3_0907_rate_reserve_chunk_legacy_random_analysis/per_seed_metrics.csv),
[분석 script](../../../pneu_rl/scripts/analyze_lib3_rate_reserve_chunk_legacy_random.py).

## 다음 실험 우선순위

### 1. reward를 바꾸기 전에 빠진 H+rate (K=2)를 채운다

가장 먼저 필요한 것은 **H+rate (K=2)**다. 다만 기존 K=2처럼 episode당
physical step을 두 배로 두지 말고, 같은 interaction budget으로 학습한다.

- K=1: horizon=512, action_chunk_steps=1
- K=2: horizon=256, action_chunk_steps=2

둘 다 episode당 512 physical step이 되도록 하고, hard reset/carry도
physical time 기준으로 일치시킨다. policy update 횟수와 physical step 수를
둘 다 report한다.

| 확인하려는 효과 | 필요한 pairwise 비교 |
|---|---|
| Reserve가 도움이 되는가? | H+rate vs HRR at K=1, 그리고 H+rate+K=2 vs HRR+K=2 |
| K=2 action sequence가 도움이 되는가? | H+rate K=1 vs K=2, 그리고 HRR K=1 vs K=2 |

이 실험 전 reward를 다시 바꾸면 reward, reserve, action representation,
interaction budget이 동시에 바뀌어 causal ablation이 불가능해진다.

### 2. “낮은 control frequency”를 보려면 command-hold를 별도로 만든다

물리적인 질문이 “update frequency를 낮추면 진동이 줄어드는가?”라면,
현재 K=2는 맞는 baseline이 아니다. 한 action \(u_t\)를 두 20-ms step 동안
유지하는 다음 조건이 필요하다.

\[
\pi(o_t)=u_t, \qquad [u_t, u_t] \rightarrow \text{two physical steps}.
\]

이 **command-hold K=2**를 현재의 **planned-action K=2**와 같은 physical
budget에서 비교한다. 그때 pressure RMSE, plateau pressure-rate, command TV를
반드시 같이 보고 판단한다.

### 3. reward 실험은 representation을 고른 뒤 별도 축으로 진행한다

그 다음에만 normalized rate-mismatch reward를 적용한다.

\[
r_{\mathrm{smooth}} =
r_{\mathrm{base}}-\lambda\sum_c
\left|\frac{\dot P_{\mathrm{act},c}-\dot r_c}{s_c}\right|.
\]

raw \( |\dot P_{\mathrm{act}}| \)가 아니라 rate mismatch를 써야 reference
ramp에서 필요한 pressure motion까지 벌주지 않는다. smoothness를 주장하는
acceptance rule도 미리 정한다: combined RMSE와 steady offset이 사전 정의한
허용 범위 안에 있을 때만 plateau motion/command TV 감소를 개선으로 인정한다.

기존 H/HR reward 2x2는 이 reward 축의 context로 유지하되, 지금 당장 다시
reward sweep을 돌리는 것보다 빠진 representation control을 먼저 채우는 것이
논문 결론에 더 중요하다.

### 4. 과거 reference는 nominal 결과와 섞지 않는다

향후 legacy RandomRef seed 수를 늘리고, 초기 0-to-offset jump가 없는
continuous legacy version도 만든다. jump magnitude와 dwell time을 명시하면
OOD robustness 보조 결과로는 더 설득력 있어진다. 그러나 nominal periodic
tracking 표와 한 평균으로 합치지는 않는다.

### 5. 그 후 finalist만 lib4와 real로 보낸다

factorial ablation과 command-bandwidth 문제가 정리된 뒤, 가장 좋은
representation/objective pair를 lib4로 옮긴다. 이후 calibrated mismatch,
zero-shot real, 동일 budget의 real fine-tuning을 MPObs+CAPS와 비교한다.
그때 논문 주장은 단순히 “어려운 system에서도 SAC가 됐다”가 아니라,
**명시한 valve bandwidth 제약 아래 physics-structured coordinate/objective가
tracking을 개선한다**가 될 수 있다.

## 별도 결과: e_dot, PIDE, GPIDE-lite screen

새로 돌린 matched-reward screen은 위 archive table과 다른 질문을 다룬다.
고정된 history에서 e_dot, PIDE, GPIDE-lite 중 어떤 coordinate bias가
finite-budget learning에 유리한가를 보는 실험이다. 그 조건에서는 e_dot이
가장 좋은 tracker였고, PIDE는 tracking–motion trade-off, GPIDE-lite는
under-tracking에 따른 apparent smoothness를 보였다.

- feature 정의, NeurIPS 2023 PIDE/GPIDE와의 정확한 차이, 그리고 feature별
  분리 그림은 [pid_inspired_feature_screen.md](pid_inspired_feature_screen.md)에
  정리했다. 특히 `GPIDE-lite`는 논문의 full GPIDE가 아니라 fixed
  difference/action negative baseline이다.
- [공통 overview 그림](../../../pneu_rl/exp/lib3_0906_structured_history_analysis/structured_history_summary.png)
- [learning dynamics](../../../pneu_rl/exp/lib3_0906_structured_history_analysis/learning_dynamics.csv)
- [full-bank metrics](../../../pneu_rl/exp/lib3_0906_structured_history_analysis/fullbank_metrics.csv)

이 screen은 chamber_bound_rwd_coeff = 0.10을 사용했고 historical
H+rate/HRR archive는 0.0을 사용했다. 따라서 둘을 observation 효과만의
직접 비교로 섞지 않는다.
