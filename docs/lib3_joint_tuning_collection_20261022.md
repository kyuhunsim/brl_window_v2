# Lib3 재수집: 0.8~1.0, 공급 밸브 저명령 시간 단축

`pneu_env/src/pneu_env/real/real_act_flowrate_6ctrl.py`의 기본 모드를
`CTRL_MODE="suite"`, `RUN_MODE="joint_tuning"`으로 변경했다. 타입을 생략하면 `type2`다.
6개 밸브의 물리 명령을 직접 생성하며 RL이나 압력 reference 추종을 사용하지 않는다.

이 구성은 대화에서 선택한 **5월 15일 CSV의 657초 구간 순서**를 바탕으로
재구성한 새 프로토콜이다. 현재 lib3 보정에 사용된 **5월 20일 722초 원본을
그대로 재현한 코드가 아니다.** 계단의 랜덤 수준과 동시 랜덤 입력은 새 고정 seed로 생성한다.

## 실행

저장소 루트에서 프로젝트 Python 환경으로 실행한다.

```bash
# 오프라인 미리보기: 통신 파일을 읽거나 제어 명령을 보내지 않는다.
python pneu_env/src/pneu_env/real/real_act_flowrate_6ctrl.py --type type1 --preview /tmp/lib3_joint_type1
python pneu_env/src/pneu_env/real/real_act_flowrate_6ctrl.py --type type2 --preview /tmp/lib3_joint_type2

# 펌프 ON, TCP 브리지 연결 상태에서 각각 순서대로 실물 수집.
python pneu_env/src/pneu_env/real/real_act_flowrate_6ctrl.py --type type1
python pneu_env/src/pneu_env/real/real_act_flowrate_6ctrl.py --type type2
```

미리보기에는 `commands.csv`, `profiles.csv`, `coverage.csv`, `config.json`, `commands.png`가 생긴다.
실물 실행은 `pneu_env/exp/`에 다음 이름과 같은 이름의 `.json`을 저장한다.

- Type 1: `*_Flowrate_RND6_suite_joint_tuning_type1_u0.80_supply0.20s.csv`
- Type 2: `*_Flowrate_RND6_suite_joint_tuning_type2_u0.80_supply0.20s_idle_ramps.csv`

각 설정을 별도 파일에 저장했으며 실행 때 해당 JSON을 읽는다. 설정 변경은 이 파일에서 한다.

| 타입 | 저장 설정 | 구성 |
|---|---|---|
| Type 1 | [`lib3_joint_tuning_type1.json`](../pneu_env/config/collection/lib3_joint_tuning_type1.json) | 단독 시험 중심, 비대상 채널 ramp 없음 |
| Type 2 | [`lib3_joint_tuning_type2.json`](../pneu_env/config/collection/lib3_joint_tuning_type2.json) | 단독 시험 일부 보존 + 비대상 채널 ramp |

두 파일은 `idle_ramps`만 다르다. 각각 657초이며 주파수·명령 범위·seed·공급측
시간 제한은 같다. 결과 JSON에는 `collection_type`, 실제 사용한 설정 사본,
설정 파일 경로·해시와 명령 해시가 함께 남는다. 펌프 ON/OFF는 이 프로그램이 제어하지 않는다.

## 설정과 변경 동작

| 설정 | 기본값 |
|---|---:|
| 제어 주기 | 50 Hz / 20 ms |
| 전체 구간 | 657초 / 예정 명령 32,850개 |
| `command_min`, `command_max` | 0.8, 1.0 |
| `supply_low_threshold` | 0.85 |
| `supply_low_max_sec` | 0.2초 |
| `seed` | 20261022 |
| `observation_timeout_sec` | 0.5초 |
| `idle_ramps` | Type 1: false / Type 2: true |
| `idle_ramp_period_sec` | 20초 |
| `solo_prefix_sec` | 10초 |

**밸브 1·2의 `u < 0.85` 연속 예정 명령을 최대 0.2초로 제한**한다.
각 낮은 계단마다 시간을 다시 허용하지 않는다. 랜덤 값이 다른 낮은 값으로
변하거나 profile 경계를 넘어가도 저명령이 이어지면 같은 연속 구간으로 센다.
해당 예산을 넘으면 요청값이 다시 0.85 이상이 될 때까지 그 밸브를 1.0으로 명령한다.

- 공급 삼각파는 주기 10초·2초를 유지하고 하단만 빠르게 통과한다.
  `0.85→0.8→0.85`에 최대 0.2초를 배정한 구간별 선형 파형이다.
- 공급 계단의 연속 저명령 구간은 전체를 압축한다. 남는 시간은 1.0 개방 유지로 채운다.
- 공급 ramp의 저명령 통과를 단축하고, 기존 마지막 최솟값 유지는 개방 유지로 바꾼다.
- 동시 저명령 5초 구간은 0.0 대신 0.8을 사용한다. 밸브 1·2는 0.2초 후 1.0으로
  복귀하고 밸브 3~6은 5초 동안 0.8을 유지한다.
- 밸브 3~6은 삼각파와 단계별 2초 유지·0.15초 전환을 유지한다.
  이 채널에는 공급측 시간 제한을 적용하지 않는다.
- 시작 대기와 종료 시 6개 밸브에 1.0을 명령한다.

### 비어 있는 채널의 추가 ramp

기존 구성에서 밸브별 1.0 비율은 약 80~82%였다. 이는 주로 한 밸브를
시험할 때 나머지 밸브를 열어 두는 구조 때문이다. 기존 구성의 6개 밸브가
동시에 전부 1.0인 시간은 97.56초(약 14.8%)였다.

단독 시험 일부를 남기면서 다음과 같이 원래 1.0이던 비대상 채널에만 ramp를 추가했다.

- 각 삼각파 profile의 **첫 1주기**(느린 파형 10초, 빠른 파형 2초)는 기존 단독 시험이다.
- 각 계단/ramp profile의 **초반 10초**도 기존 단독 시험이다.
- 이후 나머지 5개 채널에 주기 20초인 왕복 ramp를 넣는다. 각 밸브의 위상은
  주기의 1/6씩 다르며, profile 시작·끝의 추가 입력은 0.5초 동안 연결한다.
- 원래 시험 대상 밸브의 전체 명령은 그대로 유지한다. 전체 개방 회복,
  동시 저명령, 동시 랜덤 profile에는 추가 ramp를 넣지 않는다.
- 추가 ramp에도 공급 밸브 1·2의 연속 저명령 0.2초 제한을 적용한다.
- `--type type1`로 실행하면 이전 단독 시험 구성을 다시 생성한다.

기본 설정의 예정 명령에서 1.0 비율은 밸브 1이 81.4→29.7%, 밸브 2가
82.4→30.7%, 밸브 3~6이 80.2→약 26.3%로 줄었다. 전체 수집 시간은 657초로 같다.
이 비율은 명령 분포이며 모델 정확도가 개선됐다는 측정 결과는 아니다.
추가 ramp는 그림의 주황색으로 표시하며 `coverage.csv`에서 채널별 비율을 확인할 수 있다.

| 구간(초) | 패턴 |
|---|---|
| 0~5 | 전체 개방 |
| 5~47 | 밸브 1: 0.1 Hz 30초 → 0.5 Hz 12초 |
| 47~52 | 전체 개방 |
| 52~94 | 밸브 2: 0.1 Hz 30초 → 0.5 Hz 12초 |
| 94~99 | 전체 개방 |
| 99~186 / 186~273 / 273~360 / 360~447 | 밸브 3 / 4 / 5 / 6: 삼각파 42초 + 계단 45초 |
| 447~492 / 492~497 / 497~542 | 밸브 1 계단 / 전체 개방 / 밸브 2 계단 |
| 542~547 / 547~552 | 동시 저명령(공급측 0.2초 제한) / 전체 개방 |
| 552~577 / 577~582 | 밸브 1 ramp / 전체 개방 |
| 582~607 / 607~612 | 밸브 2 ramp / 전체 개방 |
| 612~652 / 652~657 | 6채널 랜덤 / 전체 개방 |

위 표는 두 타입에 공통인 주 시험의 시간표다. Type 2는 삼각파·계단·ramp 행의
단독 시험 접두 구간 이후에 비대상 채널의 추가 ramp가 동시에 실행된다.

## 시간과 데이터

새 suite는 센서 timestamp가 멈추더라도 저명령 구간이 늘어나지 않도록
monotonic 경과 시간으로 명령을 선택한다. 0.5초 동안 새 관측이 없으면 예외로
수집을 종료하고 전체 개방 명령을 게시한 뒤 수집분을 저장한다.

기존 압력 4개, `ctrl1~ctrl6`, `flow1~flow6`, 구간 정보 열을 유지했다.
추가 열 `command_ctrl1~6`은 Python이 게시한 명령, `requested_ctrl1~6`은
최종 공급측 제한 전 계획값, `supply_low_limited1/2`는 개방 복귀 개입 여부다.
`idle_ramp1~6`은 각 채널에 추가 ramp가 배정된 표본을 표시한다.
`ctrl1~6`은 기존과 같이 관측 JSON의 명령 echo이며 실제 기계적 개방 측정값은 아니다.
`command_time`은 실행 시작부터의 monotonic 초, `suite_time`은 시작 감지 이후의
계획 경과 시간이다. JSON에는 설정, 구간표, 계획 명령 해시, 수집 코드 해시가 저장된다.

0.2초 제한은 **명령 시퀀스에 대한 제한**이다. 통신 지연·중단이나 실제 밸브 동작을
독립적으로 막는 RT watchdog이나 과압 차단 기능을 추가한 것은 아니다.
릴리프 모델과 압력 상한 제어도 추가하지 않았다.

## 검증

```bash
python -m unittest discover -s pneu_env/tests -p test_joint_tuning_profiles.py -v
```

657초 전체 범위, 6채널 입력, 공급측 연속 시간 제한, actuator 유지 시간,
프로파일·랜덤 경계에서의 제한 유지, seed 재현성, 추가 ramp의 채널·단독 시험 보존,
명령 분포 변화, 두 저장 설정과 CLI 선택, bridge 접근 없는 미리보기,
가짜 시계·통신을 통한 두 타입의 전체 수집 루프와 타입별 저장을 확인한다.
실물 수집은 실행하지 않았다.
