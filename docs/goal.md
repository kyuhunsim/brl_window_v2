목표: 내 simulator (cpp, python based runge kutta simulator)가 sim real mismatch 가 없어야 하며, 이를 기반으로 reinforcement learning을 수행해서 soft actuator의 양/음압 제어 및 변위 제어를 하는 것

현재 내 문제점: rl flow 혹은 sim (cpp) 쪽에 문제가 있어서 rl 학습이 안되고, sim도 real과 matching이 잘 되는지 의문임.

예상 원인: tuning 하는 데이터의 window size가 real 및 강화학습에 문제를 일으킨다. 혹은 rl, sim, pred, env, 쪽의 연결이나 코드 쪽에 문제가 있어서 강화학습이 안된다.

참고 논문: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11027315, **Modeling and Reinforcement Learning-Based Control of Simultaneous Positive and Negative Pressure Generation in Pneumatic Systems**

실험 순서

1. main pc의 fpga.vi를 켠다.
2. rt.vi를 켠다.
3. 펌프를 켠다.
4.  통신을 시작한다. (contoller pc의 tcpip_.py 실행)
    1. 통신을 시작하면 rt.vi에 특정 루프가 돌기 시작
5. Real_act.py를 실행
    1. real에서 tcpip로 명령 전달 → rt에서 fpga로 제어명령 전달, 밸브 열고 닫음
    2. actuator 변위 생김
    3. 이를 수행하면서 csv에 각종 값 저장이 됨
6. 이 csv를 기반으로 파라미터 튜닝을 하거나 simulator 유사도 검증함. → 다양한 코드 있음
7. 만약에 simulator가 진동하지 않고 sim real matching이 잘 된다 → cpp 기반으로 바인딩 된 python sim을 reinforcement learning 학습을 한다.

---

## 일반화된 운영 플로우 (권장)

아래는 특정 파일명에 덜 의존하는 형태의 end-to-end 흐름이다.

1. **통신 브리지 시작 (RT 연결)**
   - 제어 명령 파일(`ctrl_*.json`)을 읽어서 RT로 송신
   - RT 응답을 관측 파일(`obs_*.json`)과 히스토리 CSV로 저장
   - 핵심 체크: 루프 주기 안정성, time reset/start 감지, 패킷 포맷 일치

2. **Real 데이터 취득**
   - Real 환경 루프가 제어 입력을 생성/전송하고 센서 관측을 수집
   - 실험 CSV(`env/exp/*.csv`)를 생성
   - 핵심 체크: 제어 채널 매핑 일관성, 압력/유량/시간축 누락 여부

3. **튜닝/식별**
   - 취득한 real CSV로 파라미터 튜닝 수행
   - 결과물은 계수 파일(yaml/pkl 등) 또는 C++ 파라미터 블록 형태로 저장
   - 핵심 체크: 튜닝 구간(window), 클리핑 구간, objective 정의
   - 튜닝은 밸브 파라미터 튜닝과 이를 반영한 sim을 이용한 펌프 파라미터 튜닝으로 나뉨.

4. **튜닝 결과를 시뮬레이터에 반영**
   - 방법 A: C++ 기본 파라미터를 갱신 후 라이브러리 재빌드
   - 방법 B: 런타임 setter/API로 파라미터 주입
   - 반영 후 replay 비교(sim vs real)로 검증
   - 핵심 체크: 반영 경로가 RL 실행 시점까지 실제로 이어지는지

5. **RL 학습 (Simulation 우선)**
   - 검증된 simulator를 observation backend로 학습 실행
   - 모델/설정은 `rl/models`와 실험 로그 폴더에 저장
   - 핵심 체크: state/action 차원, reward 항, reference 생성기, 학습 안정성

6. **Real 빠른 검증 (Quick Viz)**
   - 동일 정책을 Real observation 모드로 실행해 추종 성능 확인
   - 로그/플롯 생성 후 RMSE 등 지표 확인
   - 핵심 체크: sim에서 쓰던 전처리/스케일링/제어범위와 Real 일치 여부

## 운영 원칙

- 현재 메인 트랙은 **lib9 기반**으로 유지하는 것이 일관성 측면에서 유리하다.
- 튜닝 결과가 RL 학습에 자동 반영되지 않는 구조라면, 반드시
  **반영 단계(주입/빌드)를 명시적 파이프라인으로 고정**해야 한다.
- 실험 반복 시에는 항상
  **(입력 CSV) -> (튜닝 산출물) -> (sim 반영 버전) -> (RL 모델 버전)**
  의 대응관계를 함께 기록한다.
