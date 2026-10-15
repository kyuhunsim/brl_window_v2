# lib2 simulator-backed reference design

## 목적

기존 joint-point reference는 측정된 pressure point를 연결하므로 point coverage는 보장하지만, POS와 NEG의 변화 timing·cycle·phase를 독립적으로 설계하기 어렵다. 또한 pressure-space에서 두 점을 보간한 중간 상태가 실제 plant가 방문 가능한 상태인지 별도로 확인해야 한다.

이번 reference는 반대로 bounded continuous command를 먼저 만들고, 그 command를 전체 lib2 simulator에 입력한다.

\[
u_{pos}(t),u_{neg}(t)
\xrightarrow{\text{lib2 simulator}}
P_{pos}(t),P_{neg}(t)
\]

저장되는 pressure trajectory는 simulator가 실제로 방문한 상태이므로 valve flow, pump state, chamber coupling을 포함한다. pressure min/max와 rate limit은 추가적인 안전·난이도 필터로만 사용한다.

## 생성 모드

### 2048 reset reference — v2 transition/hold bank

[lib2_optimized_target_hold_bank_v2.yaml](../pneu_rl/config/reference/lib2_optimized_target_hold_bank_v2.yaml)은 40.96초 trajectory 10개를 포함한다. 이 bank는 단순 sine/phase bank가 아니다. 각 축마다 다음을 **독립적으로** sample한다.

- raw-command waypoint와 amplitude
- transition count: POS `2–4`, NEG `2–4`
- transition duration: `0.45–2.20 s`
- hold duration: `1–25 s`
- transition boundary time와 end command

각 transition은 quintic minimum-jerk 보간을 사용한다.

\[
s(\tau)=10\tau^3-15\tau^4+6\tau^5, \qquad
u(t)=u_0+s(\tau)(u_1-u_0).
\]

따라서 command는 연속이고 transition 시작·끝에서 변화율이 0이다. POS와 NEG가 같은 `s(\tau)` 또는 같은 boundary를 공유하지 않는다. 한 trajectory의 두 축은 각자 다른 transition/hold schedule을 갖고, simulator rollout 뒤의 **실제 pressure pair**만 저장한다.

학습 plant는 매 reset에서 대기압 `(101.325, 101.325) kPa`으로 시작하지만,
reference target은 `(150, 24) kPa` 부근에서 즉시 시작한다. 즉, 처음 몇 초의
추종 오차는 **의도된 operating-point 접근 구간**이며, reference가 대기압에서
천천히 operating point로 내려오는 이전 profile은 main 비교에 사용하지 않는다.
저장된 horizon 이후 trajectory를 다른 trajectory로 이어 붙이지 않도록
`repeat_after_end: false`로 설정했다. 512 carry에는 이 profile을 사용하지 않는다.

### 512 carry reference — v2 transition/hold smoke

`generate_lib2_stabilized_carry_reference.py --target-mode segmented`는 carry
중에도 POS/NEG target의 transition count, hold, transition time 및 waypoint를
각각 독립적으로 만든다. pressure reference 자체는 이 target을 그대로 쓰지 않고,
bounded reference-generation feedback을 거친 **하나의 연속 simulator rollout**이다.

- [10×512 profile](../pneu_rl/config/reference/lib2_stabilized_carry_target_hold_v2_10x512.yaml): 연결성/oracle smoke
- [100×512 profile](../pneu_rl/config/reference/lib2_stabilized_carry_target_hold_v2_100x512.yaml): long-carry drift smoke
- [100×512 validation](../pneu_rl/exp/_analysis/lib2_optimized_target_hold_reference_v2/carry100_validation.yaml)

100×512에서 POS/NEG span은 `71.88 / 6.02 kPa`, 최대 rate는 `20.06 / 8.66 kPa/s`이며,
band/rate/512-boundary continuity를 통과했다. oracle replay maximum absolute error는
`2.77e-4 / 7.35e-6 kPa`다. 반면 안전 guard 없이 같은 schedule을 실행하면 POS가
`224.8 kPa`까지 drift했다. 따라서 1000×512 최종 profile 생성에서도 이 guard를
반드시 유지하고, 생성 직후 `validate_carry_reference.py`와 oracle replay를 통과시켜야 한다.

environment carry semantics도 별도로 512 step에서 검사했다.
[carry_boundary_512.yaml](../pneu_rl/exp/_analysis/lib2_optimized_target_hold_reference_v2/carry_boundary_512.yaml)에서
simulator time, environment time, pressure history, reference buffer 및 직전 normalized
action이 reset 전후 동일했고, 다음 `step()`만 정확히 `0.02 s` 전진했다. 즉 carry reset은
`time_reset`, pressure reset, reference reset 또는 hidden reset action을 호출하지 않는다.
이 검사는 [check_lib2_carry_boundary.py](../pneu_rl/scripts/check_lib2_carry_boundary.py)로 재현한다.

1000×512 profile은 아직 생성하지 않았다. 가장 안전한 실행 방법은 아래 준비 script다.
SAC를 시작하지 않고 generation, carry validation, carry-boundary test, oracle replay,
fixed PID diagnostic를 순서대로 수행한다.

```bash
bash pneu_rl/scripts/prepare_lib2_carry_v2.sh
```

PID-only 전체 512,000-step diagnostic를 나중에 별도로 실행하려면 다음처럼 건너뛸 수 있다.

```bash
RUN_PID=0 bash pneu_rl/scripts/prepare_lib2_carry_v2.sh
```

아래는 위 script가 실행하는 final generation 명령이다.

```bash
PYTHONPATH= MPLCONFIGDIR=/tmp/mpl_lib2ref python3 pneu_rl/scripts/generate_lib2_stabilized_carry_reference.py \
  --seed 743 --duration 10240 --freq 50 --target-mode segmented \
  --target-pos-range 120,205 --target-neg-range 14,28 \
  --kp-pos 0.008 --kp-neg 0.013 --dither-amp-pos 0.04 --dither-amp-neg 0.03 \
  --pos-pressure-guard-kpa 205 --pos-pressure-guard-kp 0.03 --pos-pressure-guard-min-raw 0.38 \
  --progress-every-horizons 10 \
  --output pneu_rl/config/reference/lib2_stabilized_carry_target_hold_v2_1000x512.yaml \
  --npz-output pneu_rl/config/reference/lib2_stabilized_carry_target_hold_v2_1000x512.npz \
  --plot-output pneu_rl/img/reference/lib2_stabilized_carry_target_hold_v2_1000x512.png
```

생성 중에는 매 10개 512-step horizon마다 현재 pressure, target, raw action을 출력한다.
따라서 command/pressure가 장시간 drift하는 경우 결과 파일을 기다리지 않고 terminal에서
즉시 확인할 수 있다.

## 검증 결과

### v2 재생성 명령

다음 명령은 현재 v2 bank의 설계 범위로 candidate를 생성한다. 생성 전에 user가
`img/reference/lib2_optimized_target_hold_bank_v2.png`와 validation grid를 확인한 뒤에만
학습 launcher를 실행한다.

```bash
PYTHONPATH= MPLCONFIGDIR=/tmp/mpl_lib2ref python3 pneu_rl/scripts/optimize_lib2_reference.py \
  --seed 740 --candidates 50 --bank-size 10 \
  --segments-pos 2,4 --segments-neg 2,4 --hold-sec 1,25 \
  --output pneu_rl/config/reference/lib2_optimized_target_hold_bank_v2.yaml \
  --summary-csv pneu_rl/exp/_analysis/lib2_optimized_target_hold_reference_v2/trajectory_summary.csv \
  --plot-output pneu_rl/img/reference/lib2_optimized_target_hold_bank_v2.png
```

생성기는 pressure band, rate limit, 최소 span(POS `35 kPa`, NEG `4 kPa`)을 통과한
candidate만 저장한다. candidate의 command는 continuous이고, action projection이나
quantization은 생성·학습·평가 어느 단계에도 사용하지 않는다.

### Oracle replay

[validate_optimized_reference.py](../pneu_rl/scripts/validate_optimized_reference.py)로 저장된 command를 동일 simulator 초기조건에서 재생했다.

- target-start v2 reset bank 10개: POS/NEG replay RMSE = `0.0 kPa`
- target-start 1000x512 stress carry bank: POS RMSE 약 `3.6e-5 kPa`, NEG RMSE 약 `2.2e-5 kPa`

이는 reference가 simulator에서 도달 가능한지 검증하는 결과다. RL이나 고정 PID가 쉽게 추종한다는 뜻은 아니다.

### Bank-level diversity and constraint report

[validate_reference_bank.py](../pneu_rl/scripts/validate_reference_bank.py)는 reset
bank의 10개 trajectory를 POS/NEG 분리 grid와 CSV/YAML validation report로 만든다.
v2 기본 출력은
`pneu_rl/exp/_analysis/lib2_optimized_target_hold_reference_v2/reference_bank_validation/`이다.

- [pressure grid](../pneu_rl/exp/_analysis/lib2_optimized_target_hold_reference_v2/reference_bank_validation/pressure_grid.png): 10개 trajectory의 POS/NEG 개형을 각각 분리해 표시
- [coverage plot](../pneu_rl/exp/_analysis/lib2_optimized_target_hold_reference_v2/reference_bank_validation/coverage.png): pressure span, transition count/hold, maximum \(|dP/dt|\) 비교
- [validation table](../pneu_rl/exp/_analysis/lib2_optimized_target_hold_reference_v2/reference_bank_validation/validation.csv): trajectory별 pass/fail 및 수치

현재 target-start reset bank는 reference-rollout 내부 pump settling 1초를 제외한
설계 필터에서 다음을 모두 통과했다. 이는 training plant를 대기압에서 시작할 때의
초기 추종 오차와는 별도의 검증이다.

- pressure band: 10/10
- POS/NEG rate limit: 10/10
- 저장 command oracle replay joint-feasibility: 10/10
- POS/NEG independent transition/hold timing: 10/10

v2 coverage는 POS transition `2–4회`, NEG transition `2–4회`, POS span
`35.89–82.48 kPa`, NEG span `4.24–8.76 kPa`다. 최대 reference rate는 POS
`12.62 kPa/s`, NEG `9.46 kPa/s`로 각각 제한 `32.0/9.5 kPa/s` 아래다. 이 수치는
**bank 전체의 다양성** 증거이며, 한 trajectory 안에서 양·음압이 완전히 독립적인
plant로 움직인다는 뜻은 아니다. 실제 pressure coupling은 저장 rollout에 그대로 포함된다.

### Fixed PID

고정 feed-forward + I-term PID는 trajectory마다 필요한 feed-forward command를 바꾸지 못하므로 별도의 feedback 난이도 실험이다. PID RMSE가 크더라도 oracle replay가 통과하면 “reference 불가능”과 “feedback controller가 부족함”을 분리할 수 있다.

`evaluate_pid_reference_bank.py`로 target-start v2 reset bank 10개를 같은 ATM 초기조건,
base action `(0.5, 0.5)`, `Ki_pos=Ki_neg=0.01`, `Ka=10`으로 각각 평가했다.

- POS RMSE: 평균 `19.95 kPa`, 범위 `10.95–35.31 kPa`
- NEG RMSE: 평균 `7.72 kPa`, 범위 `6.71–9.31 kPa`
- 5초 이후 POS/NEG RMSE: 평균 `20.32 / 4.96 kPa`
- POS/NEG saturation fraction: 평균 `35.5 / 34.0%`

따라서 v2는 **oracle-feasible stress bank**이지, 이 고정 PID가 충분히 잘
추종하는 PID-gated training bank는 아니다. trajectory별 결과가 크게 달라지므로
이 결과는 bank 전체가 불가능하다는 뜻이 아니다. 고정 PID가 trajectory별
feed-forward와 양·음압 coupling을 조정하지 못한다는 뜻에 가깝다. 이 bank로
RL을 해석할 때는 “PID보다 RL이 좋은가”를 주장하지 않고, 반드시 oracle replay와
action perturbation을 함께 본다. PID-gated ablation을 주장하려면 legacy PID가
통과했던 난이도의 별도 bank를 같은 generator로 추가 생성·검증해야 한다.

target-start 512 carry profile을 10개 episode, 5120 step 동안 같은 PID로 평가한
결과는 POS RMSE `9.24 kPa`, NEG RMSE `6.69 kPa`였고 POS command saturation이
`88.5%`였다. carry profile에서도 fixed PID의 feedback/feed-forward 한계가
확인되지만, 이 결과만으로 reference 불가능을 결론내리지는 않는다.

### Local action perturbation

[local_action_perturbation.py](../pneu_rl/scripts/local_action_perturbation.py)는 실제 screen trajectory를 replay한 뒤 다음을 검사한다.

- POS command를 낮추거나 높였을 때 미래 cost가 감소하는가
- NEG command를 낮추거나 높였을 때 미래 cost가 감소하는가
- 양쪽을 같은 방향 또는 교차 방향(POS↓+NEG↑, POS↑+NEG↓)으로 바꾸는 조합이 더 좋은가
- action saturation 때문에 개선이 막혔는가

양의 improvement가 있으면 현재 policy가 사용하지 않은 local action 여지가 있다는 뜻이고, 모든 방향이 나쁘면 coupling 또는 local dynamic limitation 가능성이 커진다.
각 candidate는 기록된 action prefix를 처음부터 replay해 pressure만이 아니라 pump/valve
internal state도 같은 시점으로 재현한 뒤 평가한다. 다만 이후 horizon에는 “기록된 policy
action에 일정 delta를 더한” open-loop counterfactual이므로, 완전한 재계획 controller의
최적 성능을 뜻하지는 않는다.

`evaluate_screening.py`는 이제 reward total 대신 `reward_per_step`, error p95,
각 action의 low/high saturation 비율과 양쪽 경계까지의 평균 headroom도 저장한다.
5초 startup보다 짧은 smoke 평가에서는 rate window를 억지로 전체 구간으로
대체하지 않고 `rate_filter_evaluated: false`로 표시한다.

여러 screen 결과는
[`summarize_screening_batch.py`](../pneu_rl/scripts/summarize_screening_batch.py)로
RMSE, reward/step, saturation, pressure sign-change, local perturbation 결과를
한 표와 그래프로 묶을 수 있다.

## RL 실험 순서

1. 2048 reset stress: [lib2_optimized_target_hold_bank_v2.yaml](../pneu_rl/config/reference/lib2_optimized_target_hold_bank_v2.yaml)에서 SAC, SAC+CAPS, SAC+MPObs1+CAPS, SAC+MPObs1+CAPS+I-term
2. 각 결과를 동일 profile로 screen 및 quick viz
3. reset 결과는 representative screen/quick-viz 한 장만으로 결론내리지 않고, 10개 bank member 전체에 대해 RMSE median/p90/worst를 함께 비교
4. pressure 개형, RMSE, bias, reward/step, action saturation, (dp/dt), valve flow를 함께 비교
5. PID bank 결과와 비교해 feedback 난이도와 RL 실패를 분리
6. local action perturbation으로 “못 찾은 것”과 “물리적으로 불가능한 것”을 분리
7. 512 carry: v2 `lib2_stabilized_carry_target_hold_v2_1000x512.yaml`을 생성·band/rate/oracle/carry-boundary 검증한 뒤 동일 4개 ablation. 10×512 및 100×512 profile은 연결·drift smoke 전용이며, 1000-episode 학습에는 사용하지 않는다. 이전 `target_stress` profile은 legacy 참고용이며 main ablation에 사용하지 않는다.
8. lib2의 reference·초기조건·carry·CAPS·MPObs·I-term 영향을 분리한 뒤에만 lib3로 이동

`MPObs1`은 prediction observation을 사용하지만 `num_act=1`을 유지한다.
즉 SAC와 action dimension이 모두 2인 상태에서 observation/prediction 효과를
비교한다. 기존 `sac+mpobs+caps`는 `num_act=5`여서 10차원 action policy가 되므로,
main four-way ablation이 아니라 별도의 action-horizon 실험으로 분리한다.

[run_lib2_reference_ablation.sh](../pneu_rl/scripts/run_lib2_reference_ablation.sh)는
reset/carry와 네 main variant 중 정확히 하나만 실행한다. 기본값은 1000 episode,
seed 0이며, 먼저 `EPISODES=20`으로 smoke를 수행한 뒤 본 학습을 실행한다.

v2 1000×512 carry profile을 생성·검증한 뒤에는 launcher를 수정하지 않고 아래처럼
명시적으로 연결한다.

```bash
CARRY_REFERENCE_PROFILE=pneu_rl/config/reference/lib2_stabilized_carry_target_hold_v2_1000x512.yaml \
  bash pneu_rl/scripts/run_lib2_reference_ablation.sh carry sac 0
```

동일 environment variable을 `analyze_lib2_reference_ablation.sh carry 0`에도 주어야
screen, local perturbation, quick-viz가 training reference hash와 일치한다.

초기 `start_epi=10` episode는 raw normalized action `0.6667`을 사용하며,
lib2 physical scale `[0.85, 1.0]`에서 정확히 `(0.95, 0.95)`가 된다. 이후에는
SAC policy가 normalized `[0, 1]`, 즉 physical `[0.85, 1.0]` 전체를 출력한다.
`--action-low-physical 0.90`은 fixed warm-up을 끌 때만 uniform warm-up의 하한으로
쓰이며 learned policy 범위를 줄이지 않는다.

학습 완료 뒤에는 임의 profile을 고르는 legacy quick-viz 흐름을 사용하지 않는다.
[analyze_lib2_reference_ablation.sh](../pneu_rl/scripts/analyze_lib2_reference_ablation.sh)는
저장 model cfg와 같은 profile로 2048-step screen figure를 만들고, POS/NEG
분리 pressure·error·control·flow·slew를 한 번에 저장한다. 이어서 exact-profile
quick-viz도 저장하고 local action perturbation과 four-model summary를 생성한다.
quick-viz는 지정 YAML의 hash가 training reference와 다르면 즉시 실패한다.
의도적인 transfer test만 `QUICK_VIZ_ALLOW_REFERENCE_MISMATCH=1`을 명시한다.

분석 script는 `evaluate_screening.py --require-reference-match`를 사용한다.
저장 model config의 `reference`와 지정 YAML의 canonical SHA-256 hash가 다르면
평가를 시작하지 않는다. 각 `metrics.yaml`에도 training/evaluation reference hash와
`reference_content_match`를 저장하므로, train/evaluation reference mismatch를
추후 성능 저하로 잘못 해석하지 않는다. 다른 reference로의 transfer test만 이
flag 없이 별도 tag로 실행한다.

reset bank 전체 평가는 다음으로 수행한다. model 하나에 대해 10개 member의 screen을
저장하며, `reference_bank_aggregate.yaml/png`에는 mean/median/p90/worst RMSE가 남는다.

```bash
bash pneu_rl/scripts/evaluate_lib2_reset_reference_bank.sh MODEL_NAME
```

네 main variant가 모두 완료된 뒤에는 아래 runner가 variant별 10-member 평가와
aggregate comparison을 수행한다.

```bash
bash pneu_rl/scripts/analyze_lib2_reset_bank_ablation.sh 0
```

## 해석 게이트

- Oracle replay 실패: reference generator 문제
- Oracle 통과, PID 실패: feedback 난이도 또는 feed-forward 부족
- Oracle/PID 통과, SAC 실패: exploration·observation·reward·architecture 문제
- local perturbation 개선 가능: policy가 action 조합을 아직 충분히 탐색하지 못했을 가능성
- local perturbation 전 방향 실패: coupling 또는 reference rate/capacity 재검토
- carry 경계에서 jump: carry profile 생성 실패이며 RL 결과를 해석하지 않음
