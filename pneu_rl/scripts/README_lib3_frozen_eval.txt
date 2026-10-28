HRR 고정 정책 평가: Sim-only 2000 epi / Real fine-tuning 191 epi

실행 전
- 이 저장소 코드와 두 모델 폴더를 실물 PC에 준비한다.
- 기존 PyTorch 환경을 활성화한다. 이 PC는 conda env_1을 사용했다.
- 저장소 루트에서 실행한다. 스크립트가 자체적으로 PYTHONPATH를 구성한다.
- Real은 기존 방식대로 펌프와 bridge를 먼저 준비한다.
- 모델 폴더명은 아래 예시와 다르면 --model만 수정한다. 절대경로도 지원한다.
- 각 명령은 하나씩 실행한다. 실물 제어 프로세스를 동시에 실행하지 않는다.

기본 protocol
전체 개방(1.0 x 6) 준비 30초 -> 정책 constant hold 20초 -> bank0 40초.
hold 목표는 모델에 저장된 bank0의 시작 목표이며 실측 시작압으로 rebasing하지 않는다.
Type1은 약 123.24/73.87kPa. 준비 후 실제 시작압은 manifest에 별도 기록한다.
hold 동안 기존 reference preview가 채워진다. 생산/학습 reference buffer 코드는 변경하지 않는다.
마지막 bank 시점 뒤의 preview는 40초 endpoint를 유지한다. 기존 REF6과 마지막 preview가 다르므로
이 protocol끼리 비교하고 기존 40초 Quick Viz와 완전히 동일한 실험이라고 해석하지 않는다.
Sim은 모델의 delay/noise/library를 사용하며 Real은 실제 backend를 사용한다.
명령은 normalized 0~1 -> physical 0.8~1.0. PID, optimizer, replay, 정책 업데이트 없음.
두 sampling 모드 모두 actor는 고정한다. stochastic은 학습을 다시 켠다는 뜻이 아니다.

실물 연결 없이 두 모델 파일 검사 (backend를 열지 않음)
python pneu_rl/scripts/evaluate_lib3_frozen.py --model 261026_lib3_pressurecal_type1_HRR_smooth_seed0 --obs real --dry-run
python pneu_rl/scripts/evaluate_lib3_frozen.py --model 1027_HRR_retrain_1 --obs real --dry-run

A. Sim-only HRR, deterministic
python pneu_rl/scripts/evaluate_lib3_frozen.py --model 261026_lib3_pressurecal_type1_HRR_smooth_seed0 --obs real --mode deterministic --seed 101

B. 동일 Sim-only HRR, frozen stochastic
python pneu_rl/scripts/evaluate_lib3_frozen.py --model 261026_lib3_pressurecal_type1_HRR_smooth_seed0 --obs real --mode stochastic --seed 101

C. Real191 후 actor, deterministic 재평가
python pneu_rl/scripts/evaluate_lib3_frozen.py --model 1027_HRR_retrain_1 --obs real --mode deterministic --seed 101

D. 동일 Real191 후 actor, frozen stochastic
python pneu_rl/scripts/evaluate_lib3_frozen.py --model 1027_HRR_retrain_1 --obs real --mode stochastic --seed 101

모델 이름의 숫자/기존 cfg.deployment.checkpoint_episode는 학습 종료점 증거로 사용하지 않는다.
명령은 지정 폴더의 model.pth를 읽으며 실제 checkpoint hash와 actor tensor hash를 기록한다.
다른 checkpoint는 --checkpoint model_epi_2000.pth 또는 절대 파일 경로로 지정한다.
반복 stochastic 평가는 --seed 102,103 등으로 별도 실행. 같은 seed여도 Real의 상태·잡음은 같지 않다.
Sim에서는 위 명령의 --obs real을 --obs sim으로 변경한다. 현재 PC의 Real191 폴더는
pneu_rl/exp/lib3_hrr_retrain191_20261027/weights/pneu_rl/models/1027_HRR_retrain_1 이다.

출력
pneu_rl/exp/<model_name>/<시간>_frozen_<sim/real>_<mode>_seed<seed>/
--output 새폴더 로 지정 가능. 기존 결과가 있는 폴더는 덮어쓰지 않는다.
- manifest.json: 실제 checkpoint/actor/source/library/ref hash, sampling 모드·seed,
  초기 압력, 실행 성공/중단 상태, attempted/completed transition 수, 가중치 변경 여부.
- source_config.yaml / resolved_config.yaml: 원본 모델 설정 / 실제 적용한 설정.
- reference.npz / source_snapshot/: ref 원본과 평가에 사용한 주요 Python 코드 snapshot.
- preparation.csv: 초기화 및 전체 개방 준비 교환 기록. 학습 reward·본 평가 지표에는 미포함.
- rollout.csv: step 후 압력, 적용 ref, 요청 명령, backend echo, 유량(제공 시), 시각·소요 시간.
- policy_trace.npz: step 전 raw state / normalized policy input, pre-tanh mu/log_std,
  epsilon, 선택 action, tanh(mu) deterministic action, physical command, next_state, 완료 flag.
  실패한 마지막 명령 시도도 남으며 next_state는 NaN, completed는 false다.
- metrics.json: whole / hold_all / hold_last10 / bank_all / bank_after5의 RMSE, bias,
  error std, 압력 peak-to-peak, 2–20Hz pressure RMS, 명령 변화량, 양끝 왕복 횟수.
- quick_viz_main.png / quick_viz_ctrl.png: Quick Viz 형식 압력 3단 / 명령 6개.
  기본은 파일로 저장. --show를 붙이면 종료 후 GUI를 표시한다.

해석
- hold 마지막10초가 정착했다는 보장은 없다. 그래프의 drift/진동과 함께 판단한다.
- 움직이는 ref 구간의 pressure peak-to-peak는 ref 움직임도 포함하므로 진동 지표가 아니다.
- pressure 2–20Hz 성분도 센서잡음·aliasing·물리진동을 구분하지 못한다.
- exp(log_std)는 squashing 전 Gaussian 표준편차. 실제 명령 표준편차와 다르다.
- tanh(mu)는 deterministic action이며 일반적으로 stochastic action의 조건부 평균이 아니다.
- sensor_time은 backend 시각으로, 물리 센서 취득 시각이나 end-to-end latency 인증이 아니다.
- 명령 echo는 실제 밸브 개구율 측정값이 아니다. 물리적 동특성 식별에는 이 점을 고려한다.
- stochastic/deterministic 비교와 Sim/Real 비교는 같은 ref hash 및 설정을 확인한다.
- 중단·timeout 결과는 manifest의 status와 completed_transitions를 함께 보고 완주 결과와 구분한다.

종료
Ctrl+C 또는 sensor timeout 시 env.close()를 거쳐 Real 전체 개방 명령을 게시하고 부분 데이터를 저장한다.
펌프는 이 스크립트가 끄지 않는다. close_error가 있으면 manifest에 별도로 남는다.
SIGKILL/전원 중단은 예외 처리 불가능: 주기적으로 flush한 rollout.csv만 남을 수 있다.
이번 준비 과정에서는 Real 연결 없이 Sim 4조건과 mock timeout/dry-run 테스트를 수행했다.
