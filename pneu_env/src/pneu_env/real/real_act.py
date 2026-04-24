#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional
import numpy as np
import time
import threading
import json
import os


from utils.utils import get_pkg_path, color
from env.pid import PID


class PneuRealAct:
    """
    실험용 Soft Actuator + Main Chamber Real 환경 래퍼.

    - ctrl:
      - 길이 6: main 2 + actuator 4, 모두 [-1,1]
    - goal: 길이 2, [pos_ref, neg_ref] 목표 압력
    - scale=True일 때 최종 출력은 main/actuator 모두 [0.8,1.0]로 스케일링.

    파일 인터페이스:
      - write:  ctrl_act.json
      - read:   obs_act.json

    LabVIEW / TCP 브리지 쪽에서 이 두 파일을 기반으로 실제 밸브 제어 & 센서 읽기를 수행.
    """

    def __init__(
        self,
        freq: float = 100.0,
        scale: bool = False,
    ):
        # 제어 주파수
        self.freq = float(freq)
        self.sen_period = 1.0 / self.freq
        default_obs_timeout = max(0.03, 3.0 * self.sen_period)
        self.obs_wait_timeout = float(
            os.getenv("PNEU_REAL_ACT_OBS_TIMEOUT_SEC", f"{default_obs_timeout:.6f}")
        )

        # 메인 챔버 측 상태
        self.time = 0.0
        self.pos_press = 101.325
        self.neg_press = 101.325
        self.pos_ref = 101.325
        self.neg_ref = 101.325
        self.pos_ctrl = 1.0       # [0,1]
        self.neg_ctrl = 1.0       # [0,1]

        # 액추에이터 측 상태
        self.act_pos_press = 101.325
        self.act_neg_press = 101.325
        self.act_pos_ref = 0.0
        self.act_neg_ref = 0.0
        self.act_pos_ctrl1 = 0.0  # [0,1]
        self.act_pos_ctrl2 = 0.0  # [0,1]
        self.act_neg_ctrl1 = 0.0  # [0,1]
        self.act_neg_ctrl2 = 0.0  # [0,1]

        # 엔코더
        self.angle = 0.0          # [rad] or [deg] (LabVIEW 쪽 정의에 맞춰 사용)
        self.angle_reference = 0.0
        self.angular_vel = 0.0    # [rad/s] or [deg/s]
        self.len1 = float("nan")  # [m] Optional future displacement sensor
        self.vel1 = float("nan")  # [m/s] Optional future displacement sensor
        self.flowrate1 = 0.0
        self.flowrate2 = 0.0
        self.flowrate3 = 0.0
        self.flowrate4 = 0.0
        self.flowrate5 = 0.0
        self.flowrate6 = 0.0

        # 타이밍
        self.flag_time = time.time()
        self.start_time = time.time()

        # 메인/액추에이터 밸브를 [0,1] 또는 [0.7,1.0]로 스케일할지 여부
        self.scale = scale

        # LabVIEW / TCP 브리지와 공유할 디렉터리
        # 실제 프로젝트 구조에 맞춰 경로 확인 필수
        self.labview_path = f"{get_pkg_path('pneu_env')}/tcpip"

        self.stop_flag = threading.Event()

        # PID 관련
        self.is_pid = False
        self.is_anti_windup = False
        self.pid: Optional[PID] = None

        # PID 내부 오차 계산은 actuator 압력쌍 기준으로 수행.
        self.obs = np.array([self.act_pos_press, self.act_neg_press], dtype=np.float32)

    # ------------------------------------------------------------------
    # 파일 I/O
    # ------------------------------------------------------------------
    def _ctrl_filepath(self) -> str:
        return os.path.join(self.labview_path, "ctrl_act.json")

    def _obs_filepath(self) -> str:
        return os.path.join(self.labview_path, "obs_act.json")

    def write_ctrl_file(self) -> None:
        """
        현재 제어/참조/상태 항목을
        ctrl_act.json 에 '원자적으로' 기록.
        """
        data = {
            "time": float(time.time() - self.start_time),

            "pos_press": float(self.pos_press),
            "neg_press": float(self.neg_press),
            "pos_ref": float(self.pos_ref),
            "neg_ref": float(self.neg_ref),
            "pos_ctrl": float(self.pos_ctrl),
            "neg_ctrl": float(self.neg_ctrl),

            "act_pos_press": float(self.act_pos_press),
            "act_neg_press": float(self.act_neg_press),
            "act_pos_ref": float(self.act_pos_ref),
            "act_neg_ref": float(self.act_neg_ref),
            "act_pos_ctrl1": float(self.act_pos_ctrl1),
            "act_pos_ctrl2": float(self.act_pos_ctrl2),
            "act_neg_ctrl1": float(self.act_neg_ctrl1),
            "act_neg_ctrl2": float(self.act_neg_ctrl2),

            "angle": float(self.angle),
            "angle_reference": float(self.angle_reference),
            "angular_vel": float(self.angular_vel),
            "flowrate1": float(self.flowrate1),
            "flowrate2": float(self.flowrate2),
            "flowrate3": float(self.flowrate3),
            "flowrate4": float(self.flowrate4),
            "flowrate5": float(self.flowrate5),
            "flowrate6": float(self.flowrate6),
        }

        tmp = self._ctrl_filepath() + ".tmp"
        dst = self._ctrl_filepath()

        os.makedirs(self.labview_path, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, dst)  # 원자적 교체

    def read_obs_file(self) -> None:
        """
        obs_act.json 에서 상태 항목을 읽어 self.* 상태 업데이트.
        LabVIEW / TCP 브리지에서 이 파일을 써줘야 한다.
        """
        path = self._obs_filepath()
        prev_time = float(self.time)
        obs_wait_timeout = float(
            getattr(self, "obs_wait_timeout", max(0.03, 3.0 * self.sen_period))
        )
        deadline = time.time() + max(0.0, obs_wait_timeout)

        def _get_float(data: Dict, key: str, default: float) -> float:
            """
            obs_act.json에서 키가 없거나 타입이 잘못돼도 이전 값(default)을 유지.
            """
            value = data.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _get_any_float(data: Dict, keys: tuple[str, ...], default: float) -> float:
            for key in keys:
                if key in data:
                    return _get_float(data, key, default)
            return default

        while not self.stop_flag.is_set():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obs = json.load(f)
                obs_time = _get_float(obs, "time", self.time)
                # Prefer a fresh sample. If bridge update is delayed, keep trying
                # until timeout and then fall back to the latest readable sample.
                if obs_time <= prev_time + 1e-9 and time.time() < deadline:
                    time.sleep(min(0.002, 0.2 * self.sen_period))
                    continue
                # 필드 매핑
                self.time = obs_time

                self.pos_press = _get_float(obs, "pos_press", self.pos_press)
                self.neg_press = _get_float(obs, "neg_press", self.neg_press)
                self.pos_ref = _get_float(obs, "pos_ref", self.pos_ref)
                self.neg_ref = _get_float(obs, "neg_ref", self.neg_ref)
                self.pos_ctrl = _get_float(obs, "pos_ctrl", self.pos_ctrl)
                self.neg_ctrl = _get_float(obs, "neg_ctrl", self.neg_ctrl)

                self.act_pos_press = _get_float(obs, "act_pos_press", self.act_pos_press)
                self.act_neg_press = _get_float(obs, "act_neg_press", self.act_neg_press)
                self.act_pos_ref = _get_float(obs, "act_pos_ref", self.act_pos_ref)
                self.act_neg_ref = _get_float(obs, "act_neg_ref", self.act_neg_ref)
                self.act_pos_ctrl1 = _get_float(obs, "act_pos_ctrl1", self.act_pos_ctrl1)
                self.act_pos_ctrl2 = _get_float(obs, "act_pos_ctrl2", self.act_pos_ctrl2)
                self.act_neg_ctrl1 = _get_float(obs, "act_neg_ctrl1", self.act_neg_ctrl1)
                self.act_neg_ctrl2 = _get_float(obs, "act_neg_ctrl2", self.act_neg_ctrl2)

                self.angle = _get_float(obs, "angle", self.angle)
                self.angle_reference = _get_float(obs, "angle_reference", self.angle_reference)
                self.angular_vel = _get_float(obs, "angular_vel", self.angular_vel)
                self.len1 = _get_any_float(
                    obs,
                    ("len1", "length", "length_m", "disp", "displacement"),
                    self.len1,
                )
                self.vel1 = _get_any_float(
                    obs,
                    ("vel1", "velocity", "vel_ms", "length_velocity"),
                    self.vel1,
                )
                self.flowrate1 = _get_float(obs, "flowrate1", self.flowrate1)
                self.flowrate2 = _get_float(obs, "flowrate2", self.flowrate2)
                self.flowrate3 = _get_float(obs, "flowrate3", self.flowrate3)
                self.flowrate4 = _get_float(obs, "flowrate4", self.flowrate4)
                self.flowrate5 = _get_float(obs, "flowrate5", self.flowrate5)
                self.flowrate6 = _get_float(obs, "flowrate6", self.flowrate6)
                break

            except FileNotFoundError:
                # 아직 obs_act.json 이 안 생겼으면 계속 기다림
                if time.time() >= deadline:
                    break
                continue
            except json.JSONDecodeError:
                # 쓰는 중에 읽어서 깨진 경우: 다시 시도
                if time.time() >= deadline:
                    break
                continue
            except Exception:
                # 다른 에러는 일단 다시 시도 (필요하면 로깅 추가)
                if time.time() >= deadline:
                    break
                continue

    # ------------------------------------------------------------------
    # 타이밍 제어
    # ------------------------------------------------------------------
    def wait(self, margin: float = 0.004) -> None:
        """
        self.freq 에 맞춰 루프 주기를 맞추기 위한 대기 함수.
        margin 만큼 여유를 두고 깨어나도록 조정.
        """
        curr_flag_time = time.time()
        sleep_time = self.sen_period - (curr_flag_time - self.flag_time) - margin
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.flag_time = time.time()

    # ------------------------------------------------------------------
    # 메인 스텝
    # ------------------------------------------------------------------
    def observe(
        self,
        ctrl: np.ndarray,
        goal: np.ndarray,
    ) -> tuple[np.ndarray, Dict]:
        """
        한 스텝 진행:
          1) 6채널 입력 해석
          2) (옵션) PID로 actuator 4밸브 명령 생성
          3) main/actuator 모두 최종 밸브 명령으로 스케일링
          4) ctrl_act.json 기록
          5) 주기 맞춰 대기
          6) obs_act.json 읽기
          7) [time, pos_press, neg_press, angle, angular_vel] 반환
        """
        ctrl_arr = np.asarray(ctrl, dtype=np.float64).reshape(-1)
        goal = np.asarray(goal, dtype=np.float64).reshape(-1)
        if goal.shape != (2,):
            raise ValueError(f"goal must be shape (2,), got {goal.shape}")
        if ctrl_arr.size != 6:
            raise ValueError(f"ctrl must be length 6, got {ctrl_arr.size}")
        if not np.all(np.isfinite(ctrl_arr)):
            ctrl_arr = np.nan_to_num(ctrl_arr, nan=0.0, posinf=1.0, neginf=-1.0)
        ctrl_arr = np.clip(ctrl_arr, -1.0, 1.0)
        main_ctrl_bipolar = ctrl_arr[:2]
        act_ctrls_unit = 0.5 * ctrl_arr[2:] + 0.5

        # PID ON이면 actuator 밸브를 PID/입력 혼합으로 생성한다.
        if self.is_pid:
            pid_out = self.pid.get_action(self.obs, goal)  # [2]
            # PID class는 legacy sign을 사용하므로 pos축만 부호를 뒤집어 actuator 방향으로 맞춤.
            u_pos = float(-pid_out[0])
            u_neg = float(pid_out[1])

            # print("PID_OUT (pos, neg): ", u_pos, u_neg)

            act_unsat = np.array(
                [
                    max(u_pos, 0.0),
                    max(-u_pos, 0.0),
                    max(u_neg, 0.0),
                    max(-u_neg, 0.0),
                ],
                dtype=np.float64,
            )
            act_sat = np.clip(act_unsat, 0.0, 1.0)
            rl_pos_in = float(np.clip(act_ctrls_unit[0], 0.0, 1.0))
            rl_pos_out = float(np.clip(act_ctrls_unit[1], 0.0, 1.0))
            rl_neg_in = float(np.clip(act_ctrls_unit[2], 0.0, 1.0))
            rl_neg_out = float(np.clip(act_ctrls_unit[3], 0.0, 1.0))

            pos_in_mix = float(np.clip(rl_pos_in + act_sat[0], 0.0, 1.0))
            pos_out_mix = float(np.clip(rl_pos_out + act_sat[1], 0.0, 1.0))
            neg_in_mix = float(np.clip(rl_neg_in + act_sat[2], 0.0, 1.0))
            neg_out_mix = float(np.clip(rl_neg_out + act_sat[3], 0.0, 1.0))
            act_ctrls_unit = np.array(
                [
                    pos_in_mix,
                    pos_out_mix,
                    neg_in_mix,
                    neg_out_mix,
                ],
                dtype=np.float64,
            )
            # 합성 이후 실제 PID 기여분(포화 반영)만 anti-windup에 전달.
            pid_pos_in_sat = float(np.clip(pos_in_mix - rl_pos_in, 0.0, 1.0))
            pid_pos_out_sat = float(np.clip(pos_out_mix - rl_pos_out, 0.0, 1.0))
            pid_neg_in_sat = float(np.clip(neg_in_mix - rl_neg_in, 0.0, 1.0))
            pid_neg_out_sat = float(np.clip(neg_out_mix - rl_neg_out, 0.0, 1.0))
            u_pos_sat = float(pid_pos_in_sat - pid_pos_out_sat)
            u_neg_sat = float(pid_neg_in_sat - pid_neg_out_sat)

            if self.is_anti_windup:
                # 유효 signed command를 PID 출력 도메인으로 복원.
                pid_out_sat = np.array([-u_pos_sat, u_neg_sat], dtype=np.float64)
                self.pid.anti_windup(ctrl=pid_out, sat_ctrl=pid_out_sat)
        else:
            # PID OFF일 때는 입력 actuator 명령(RL 6ch)을 그대로 사용.
            act_ctrls_unit = np.clip(act_ctrls_unit, 0.0, 1.0)

        # main: [-1,1] -> [0,1]
        ctrl_unit = 0.5 * main_ctrl_bipolar + 0.5

        # optional final scale to [0.8,1.0]
        if self.scale:
            ctrl_unit = 0.2 * ctrl_unit + 0.8

        # actuator command is in [0,1] by construction.
        if self.scale:
            act_ctrls_unit = 0.2 * act_ctrls_unit + 0.8

        # 내부 상태 업데이트
        self.pos_ref = float(goal[0])
        self.neg_ref = float(goal[1])
        # lib5 기본 목표는 actuator pair 기준으로도 동일 ref를 사용한다.
        self.act_pos_ref = float(goal[0])
        self.act_neg_ref = float(goal[1])
        self.pos_ctrl = float(ctrl_unit[0])
        self.neg_ctrl = float(ctrl_unit[1])

        self.act_pos_ctrl1 = float(act_ctrls_unit[0])
        self.act_pos_ctrl2 = float(act_ctrls_unit[1])
        self.act_neg_ctrl1 = float(act_ctrls_unit[2])
        self.act_neg_ctrl2 = float(act_ctrls_unit[3])



        # 2) ctrl_act.json 쓰기 → LabVIEW/TCP 브리지 쪽에서 읽어서 실제 밸브 제어
        self.write_ctrl_file()

        # 3) 주기 맞춰 대기
        self.wait()

        # 4) obs_act.json 읽기 → 센서/상태 업데이트
        self.read_obs_file()

        # 5) 관측값 구성
        next_obs = np.array(
            [self.time, self.pos_press, self.neg_press, self.angle, self.angular_vel],
            dtype=np.float64,
        )
        # PID용 내부 obs (actuator pos/neg 압력)
        self.obs = np.array([self.act_pos_press, self.act_neg_press], dtype=np.float32)

        obs_info = dict(
            time=self.time,
            pos_press=self.pos_press,
            neg_press=self.neg_press,
            pos_ref=self.pos_ref,
            neg_ref=self.neg_ref,
            pos_ctrl=self.pos_ctrl,
            neg_ctrl=self.neg_ctrl,
            act_pos_press=self.act_pos_press,
            act_neg_press=self.act_neg_press,
            act_pos_ref=self.act_pos_ref,
            act_neg_ref=self.act_neg_ref,
            act_pos_ctrl1=self.act_pos_ctrl1,
            act_pos_ctrl2=self.act_pos_ctrl2,
            act_neg_ctrl1=self.act_neg_ctrl1,
            act_neg_ctrl2=self.act_neg_ctrl2,
            angle=self.angle,
            angle_reference=self.angle_reference,
            angular_vel=self.angular_vel,
            len1=self.len1,
            vel1=self.vel1,
            flowrate1=self.flowrate1,
            flowrate2=self.flowrate2,
            flowrate3=self.flowrate3,
            flowrate4=self.flowrate4,
            flowrate5=self.flowrate5,
            flowrate6=self.flowrate6,
        )
        info: Dict = {"Observation": obs_info}
        return next_obs, info

    # ------------------------------------------------------------------
    # PID 관련
    # ------------------------------------------------------------------
    def set_pid(
        self,
        Kp_pos: float, Ki_pos: float, Kd_pos: float,
        Kp_neg: float, Ki_neg: float, Kd_neg: float,
    ) -> None:
        self.is_pid = True
        self.pid = PID(
            Kp_pos, Ki_pos, Kd_pos,
            Kp_neg, Ki_neg, Kd_neg,
            freq=self.freq,
        )

    def set_anti_windup(self, Ka) -> None:
        assert self.is_pid, color("PID controller is not turned on.", "red")
        self.is_anti_windup = True
        self.pid.set_anti_windup(Ka)

    def reset_pid(self) -> None:
        if self.pid is not None:
            self.pid.reset()


if __name__ == "__main__":
    env = PneuRealAct(freq=200.0, scale=False)

    for i in range(2000):
        ctrl = np.array([1.0, 1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64)
        goal = np.array([110.0, 110.0], dtype=np.float64)     # 목표 압력

        obs, info = env.observe(ctrl=ctrl, goal=goal)
        print(f"[Step {i+1}] obs = {obs}")
