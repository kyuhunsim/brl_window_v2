Real 비교 후보 3개 — 2026-10-26

포함 모델 (모두 seed0, epi2000 하나만 포함; 파일명은 loader에 맞춰 model.pth)
1) 261024_lib3_type2_HR_meanCAPS_t006_seed0: Type2, temporal CAPS 0.06
2) 261024_lib3_type2_HR_meanCAPS_t020_seed0: Type2, temporal CAPS 0.20
3) 261026_lib3_pressurecal_type1_HR_smooth_seed0: 재보정 Type1 HR, CAPS 0

기존 Type1/Type2 plain HR은 2026-10-23 별도 ZIP에 이미 업로드되어 있다.
CAPS 0.20은 CAPS 후보 중 우선 비교 대상, 0.06은 smoothness/bias 비교 대상이다.
새 Type1 HR은 별도 재보정 plant에서 학습했다. Type1/Type2 차이와 CAPS 효과를
혼동하지 말고 각각 해당 기존 plain HR과 같은 조건에서 비교한다.

설치 (Real PC의 최신 brl_window_v2 저장소 루트에서 실행)
  unzip -n /다운로드/경로/lib3_real_candidates_CAPS006_CAPS020_pressurecalHR_epi2000_20261026.zip
  python configure_bundle.py

압축은 pneu_rl/models/<모델명>/ 구조를 그대로 포함한다.
같은 모델명 폴더가 이미 있으면 먼저 확인한다. unzip -n은 기존 파일을 덮어쓰지 않는다.
configure_bundle.py는 포함 파일의 checksum을 검증하고, 추출된 cfg.yaml의
obs.library_path만 이 PC의 simulator.so 절대경로로 연결한다. 원본 cfg.training.yaml은
수정하지 않는다. metadata의 학습 PC 경로는 provenance이며 runtime 입력이 아니다.

Zero-shot
  python pneu_rl/scripts/quick_viz3.py
  모델명 선택 -> REF 6 (model bank trajectory0, 40초) -> OBS 2 (Real)
  -> PID 2 (OFF) -> Logging 1 (저장)
  현재 코드의 model.predict는 deterministic mean action을 사용한다.
  비교할 때 초기화/ref/제어주기를 맞춘다. Type1/Type2 ref bank는 서로 다르다.

Fine-tuning 예시 (원본을 보존하는 새 이름 사용)
  python pneu_rl/scripts/retrain3.py --model 261026_lib3_pressurecal_type1_HR_smooth_seed0 --name real_pressurecal_type1_HR_ft1_200 --obs real --pid off --episodes 200
  python pneu_rl/scripts/retrain3.py --model real_pressurecal_type1_HR_ft1_200 --name real_pressurecal_type1_HR_ft2_200 --obs real --pid off --episodes 200 --load-buffer
  첫 실행은 fresh replay이므로 --load-buffer를 붙이지 않는다.
  두 번째 --load-buffer는 첫 Real 학습 결과에 buffer.pkl이 저장된 경우에만 사용한다.
  CAPS 모델은 --model과 --name을 해당 이름으로 바꾸며, 저장된 CAPS 설정을 유지한다.

공통 학습/배포 설정
- HR observation, 6 actions, 50Hz, physical command 0.8..1.0.
- 시작 시 pump ON·전체 개방으로 30초 준비한 뒤 measured history를 유지한다.
- Fine-tuning은 session 시작 시 한 번만 준비하고 ref를 연속 연결한다.
- Sim observation delay는100ms. Real에100ms를 자동 추가한다는 뜻은 아니다.
- 새 Type1 reference profile의 partial_pid_witness 상태를 그대로 보존했다.
- Real zero-shot 성공/진동 제거가 확인된 모델이라는 뜻은 아니다.

포함: checkpoint의 actor/critic/target/optimizer/log_alpha, cfg.yaml, 원 학습 cfg,
infos.pkl, reference.npz, reference_profile.yaml, 정확한 학습 simulator.so.
제외: 중간 checkpoint, replay/backup buffer, 학습 그래프/대용량 로그.
SAC loader는 alpha optimizer를 재생성하므로 bitwise 학습 재개 snapshot은 아니다.
저장소 코드는 ZIP에 포함하지 않으므로 Real PC 코드도 최신 상태여야 한다.
