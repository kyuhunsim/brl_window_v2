# new_solvalve 정리본

`new_solvalve/tune.py`는 다음 파일의 핵심 수식을 순수 Python으로 정리한 버전입니다.

- `pneu_env/src/pneu_env/tuner/sim_core.cpp`
- `pneu_env/src/pneu_env/optimizer.py`

## 모델 수식(새 솔레노이드 밸브)

1. 상태 전이(열림/닫힘)  
`State[k] = 1` (전류 증가), `0` (전류 감소), 그 외 이전 상태 유지

2. Bouc-Wen 히스테리시스  
`dI = |I[k]-I[k-1]| * (2*State[k]-1)`  
`z <- z + A_bw*dI - beta_bw*|dI|*z - gamma_bw*dI*|z|`

3. 유효 면적 S-curve  
`Force_net = I + C_z*z + C_p*P_in_abs - C_k`  
`Area_eff = A_max / (1 + exp(-k_shape*Force_net))^alpha`

4. 압축성 유동 계수  
임계압력비 `P_cr = (2/(kappa+1))^(kappa/(kappa-1))`  
`Phi`는 초크/비초크 식으로 계산

5. 정적 유량  
`Q_static = Area_eff * P_in_abs * Phi`

6. 동특성(2차 시스템, 열림/닫힘 비대칭)  
`x1_dot = x2`  
`x2_dot = wn^2*(Q_static - x1) - 2*zeta*wn*x2`  
`Q_pred = x1`

7. 오차함수  
`SSE = sum((Q_actual - Q_pred)^2)`  
초기 비정상 튐에 대해 추가 penalty 적용

## 기존 lib 모델과 차이

기존 `lib`(`pneumatic_CT.cpp`)의 `solenoid_valve()`는 전류-압력 선형식 기반 정적 모델(`Cdkx`)이고,  
새 모델은 히스테리시스(Bouc-Wen) + 비선형 면적 + 2차 동특성(열림/닫힘 분리)까지 포함합니다.

## lib2로 함수만 교체하는 절차

1. `pneu_env/src/pneu_env/lib`를 `lib2`로 복사
2. `lib2/pneumatic_CT.cpp`의 `PneumaticCT::solenoid_valve()`만 새 모델 식으로 교체
3. `PneumaticCT` 내부 상태 추가
`z`, `x1`, `x2`, `I_prev`, `State_prev`를 밸브별(최소 pos/neg 2세트)로 멤버에 추가
4. 기존 호출부 유지
`model()`의 `mdot_pos_valve`, `mdot_neg_valve` 계산 위치/인터페이스는 유지
5. 파라미터 주입 방식 결정
고정 상수(하드코딩) 또는 setter/구조체 방식으로 13개 파라미터를 주입
6. 빌드 산출물 분리
기존 `lib`는 그대로 두고 `lib2` 전용 `.so`를 별도 이름으로 생성
7. Python 로더 분기
`sim.py`/`pred.py`에서 기존 `.so`와 `lib2` `.so`를 선택 로드 가능하게 분기
8. 회귀 검증
같은 입력 CSV로 기존/신규 밸브 유량(`flow1~flow6`)을 비교해서 안정성 확인

핵심은 `solenoid_valve()`만 교체하더라도 새 식은 내부 상태를 가지므로, 함수 외에 상태 멤버 초기화/유지 코드가 함께 필요하다는 점입니다.

## 사용 예시

```bash
python3 new_solvalve/tune.py your_data.csv --valves all --samples 50 --start 200 --output new_solvalve_result.json
```
