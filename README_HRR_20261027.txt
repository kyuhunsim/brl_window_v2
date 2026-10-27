HRR baseline / HRR + Dynamics Randomization — Type1, episode2000

포함 모델
1) 261026_lib3_pressurecal_type1_HRR_smooth_seed0
2) 261026_lib3_pressurecal_type1_HRR_DR_smooth_seed0

기존 brl_window_v2 저장소의 루트에 압축을 풀고 실행:
  python configure_hrr_bundle_20261027.py

각 model.pth는 해당 모델의 model_epi_2000.pth 원본이다.
관측162차원: pressure history44 + command history60 + rate4 + reserve2 + reference52.
명령6개, 제어50Hz, 물리 명령0.8~1.0, HRR feature는 cfg에서 자동 적용된다.
두 모델은 같은 Type1 reference bank를 포함한다. REF6은 bank0 40초, 반복 없음.

Zero-shot:
  python pneu_rl/scripts/quick_viz3.py
  위 두 모델 중 하나 선택 → REF6(기본값) → Real → PID OFF.
  저장된 설정을 사용하며 deterministic 평가다. 시작 시 전체개방1.0×6,30초 준비 후 제어.
  펌프·통신은 기존 실물 PC 절차로 준비한다.

Fine-tuning (시작 시 준비1회, reference연속연결, fresh Real replay):
  python pneu_rl/scripts/retrain3.py --model 261026_lib3_pressurecal_type1_HRR_smooth_seed0 --name HRR_Type1_Real200_20261027 --obs real --pid off --episodes 200
  python pneu_rl/scripts/retrain3.py --model 261026_lib3_pressurecal_type1_HRR_DR_smooth_seed0 --name HRR_DR_Type1_Real200_20261027 --obs real --pid off --episodes 200
  위 명령은 실물에서 각각 별도로 실행한다. 준비운전은 reward/replay에 포함되지 않는다.
  Sim replay는 이 compact package에 포함되지 않는다. 기본 fresh replay 경로로 진행한다.

DR은 학습 시 사용한 조건이다. Real 배포에서는 추정 파라미터 입력이나 randomization이 실행되지 않는다.
내장 simulator.so는 두 모델 모두 nominal Type1이다. Real backend는 simulator.so를 로드하지 않는다.

검증: 다른cwd에서도 model전용ref탐색, 162차원 checkpoint load, mockReal준비/제어,
deterministic action, fresh replay학습로드, 200개boundary의초기준비1회 통과.
실물 통신/운전은 이 검증에서 실행하지 않았다.

주의할 평가 결과:
HRR nominal POS/NEG RMSE0.751/0.447kPa; DR0.899/1.107kPa.
DR은120/130kPa hold에서조용하지만140kPa hold에는진동이남는다.
Real zero-shot 성능은 직접 비교해야 한다.
