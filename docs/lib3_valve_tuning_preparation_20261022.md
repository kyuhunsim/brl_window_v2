# Lib3 밸브 재튜닝 준비

## 지난번에 실제 확인된 설정

현재 lib3의 밸브 6개 × 13개 파라미터는 다른 PC의
`/home/rise/test/pneu_tune/tune_result/output_12.txt`와 일치한다.
근거는 [10월 18일 감사](lib3_sim2real_initialization_review.md#2026-10-18-과거-lib3-수집튜닝정확도밸브-feasibility-감사)와
[보존된 해시·일치 결과](img/lib3_calibration_archive_20261018/audit.json)다.
현재 PC에서는 해당 외부 폴더에 접근할 수 없다.

| 항목 | 확인된 과거 설정 |
|---|---|
| 입력 CSV | `260520_14_44_14_Flowrate_RND6_suite_joint_tuning.csv` |
| 데이터 | 36,100행, 약 722초, 약 50 Hz |
| 대상 | 밸브 1~6, 각 13개 모델 파라미터 |
| 사용 구간 | 파일 전체 |
| 입력 / 유량 smoothing | 각각 window=1, smoothing 없음 |
| flow delay | 0 sample |
| natural frequency `wn_up/down` | 11~25 rad/s (약 1.75~3.98 Hz) |
| damping ratio `zeta_up/down` | 0.4~0.6 |
| full-open penalty | weight=1, `ctrl=1`의 정적 면적비 목표 ≥0.95 |
| opening-shape penalty | v3·v4, weight=5 |
| shape 평가 명령 | 0.85 / 0.90 / 0.95 / 1.00, 최소 면적비 증가량 0.03 |

**형상 penalty의 정확한 허용 면적비 구간과 원본 구현은 이 PC에 없다.**
현재 optimizer에도 해당 옵션/목적함수가 없으므로 과거 설정을 완전히 재현했다고
말할 수 없다. 다른 PC의 원본 `output_12.txt`와 그때 `optimizer.py`가 있으면 복원할 수 있다.
과거 seed·초기 후보 수·초기 파라미터·다른 penalty 가중치도 보존된 감사만으로는 확정되지 않는다.

현재 저장소의 `pneu_env/tune_result/output_12.txt`는 **4월 30일 CSV로 밸브 1·2만
맞춘 별개 결과**다. 파일 이름이 같다는 이유로 lib3의 5월 결과로 사용하면 안 된다.

## 이번 준비물

- 설정: [`lib3_valve_baseline.json`](../pneu_env/config/tuning/lib3_valve_baseline.json)
- 실행: [`tune_lib3_valves.py`](../pneu_env/scripts/tune_lib3_valves.py)
- fit 본체: [`optimizer.py`](../pneu_env/src/pneu_env/optimizer.py)

확인된 과거 조건을 고정하고, 미확인 탐색 조건은 현재 코드 기준으로 명시했다.
이번 baseline은 13개 전체 파라미터를 최적화한다. 현재 탐색 기본값은
seed=0, 후보 50개를 평가해 상위 3개에서 Nelder–Mead 실행,
maxiter=5000 / maxfev=15000 / xatol=fatol=1e-4다.
초기 파라미터는 기존 `CONFIGS[].base`를 사용한다.
HF/R²-floor/peak penalty는 이번 baseline에서 0으로 지정했다.
이는 과거 실행에서 확인된 값이라는 뜻이 아니다.
**v3/v4 shape penalty는 아직 미적용**이며 실행·결과 기록에도 이 제한을 남긴다.

실측 `ctrl1~6` echo, `flow1~6`, 압력 4개와 `time`을 사용한다.
새 `command_ctrl`·`requested_ctrl` 열은 fit 입력으로 대체하지 않는다.
압력은 kPa 절대압, 유량은 L/min이고, 기존 모델처럼 명령을
`I=0.30*clip(2*(ctrl-0.5),0,1)`로 변환한다. 수집 범위를 0.8~1.0으로
확장했다고 이 전류 변환식을 바꾸지 않는다.

## Type 1·2 수집 후 실행

저장소 루트에서 SciPy, NumPy, pandas, Matplotlib, PyYAML이 있는 프로젝트
환경을 사용한다. 현재 PC에서는 `conda activate env_1`로 준비돼 있다.
아래 `실측_type1.csv` / `실측_type2.csv`를 실제 수집 파일 경로로 바꾼다.

```bash
conda activate env_1

# 먼저 CSV 열, 유한값, 시간 역행, 압력/명령 범위 및 실행 설정 확인.
# 파일 생성이나 최적화를 하지 않는다. preview의 commands.csv는 실측 데이터가 아니다.
python pneu_env/scripts/tune_lib3_valves.py pneu_env/exp/실측_type1.csv --dry-run
python pneu_env/scripts/tune_lib3_valves.py pneu_env/exp/실측_type2.csv --dry-run

# 필요 시 각각 별도로 fit. 기존 lib3 계수는 덮어쓰지 않는다.
python pneu_env/scripts/tune_lib3_valves.py pneu_env/exp/실측_type1.csv
python pneu_env/scripts/tune_lib3_valves.py pneu_env/exp/실측_type2.csv
```

결과는 `pneu_env/tune_result/lib3_valves/<CSV이름>/<UTC실행시각>/`에 분리된다.
`valves.txt`(C++ 파라미터), `valves.png`(6채널 유량 fit), 3D 진단 그림·HTML,
`run.json`(설정 사본·입력/코드/결과 해시·수집 metadata·실행 상태)을 저장한다.
기본적으로 GUI 없이 저장하며 `--show`를 주면 완료 후 그림을 연다.
현재 lib3의 소스·계수·시뮬레이터 바이너리는 자동 변경하지 않는다.

점검은 중복 timestamp와 큰 기록 간격의 개수를 보고한다. 5월 15일 CSV에도
중복 timestamp 2개가 있으므로 같은 시각의 행을 자동 삭제하지 않는다.
기존 optimizer는 전체 평균 dt로 적분한다. 일정 주기의 완벽한 기록이라고
가정하지 않고 `median_dt_sec`, `optimizer_dt_sec`, `max_dt_sec`를 함께 남긴다.

Type 1과 Type 2를 각각 fit한 수치는 각자의 학습 데이터 오차다.
독립 검증을 하려면 한 파일로만 맞춘 계수를 다른 파일에서도 고정 평가한다.
full-open penalty를 넣었다는 사실과 면적비 0.95를 충족한다는 사실도 구분한다.
과거 결과는 실제로 그 목표에 미달했다.

## 준비 코드 확인

5월 15일 기존 실측 CSV 전체로 `--dry-run`을 통과했다. 별도 임시 설정에서
같은 CSV의 첫 1초·밸브 1·후보 1개만 사용해 실행 파일→SciPy/C++ fit→
TXT/PNG/3D/HTML 저장→`run.json` 상태·결과 해시까지 확인했다.
이 실행 점검 결과는 임시 폴더에서만 생성했으며 새 calibration 결과가 아니다.
Type 1·2의 경로 분리와 수집 metadata 보존은 모의 fit으로도 확인했다.
실물 통신이나 기존 lib3 계수 변경은 수행하지 않았다.
