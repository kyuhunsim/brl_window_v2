#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import struct
import time
import traceback
import os

from tcpip_bridge_common import (
    FIELDS_22,
    PACK_COUNT_22,
    read_ctrl_values,
    recv_packet,
    save_csv,
    write_json_atomic,
)

SERVER_IP = "192.168.1.1"    # RT.vi가 돌고 있는 cRIO/RT의 IP
SERVER_PORT = 5555           # RT.vi TCP 서버 포트

RECV_TIMEOUT_SEC = float(os.getenv("PNEU_TCP_RECV_TIMEOUT", "1.0"))
# 0 이하이면 자동 종료 비활성화. (권장: 0)
AUTO_EXIT_STALE_SEC = float(os.getenv("PNEU_TCP_AUTO_EXIT_STALE_SEC", "0"))
OBS_TIME_SOURCE = os.getenv("PNEU_TCP_OBS_TIME_SOURCE", "ctrl").strip().lower()
if OBS_TIME_SOURCE not in ("ctrl", "rt"):
    raise ValueError(
        "PNEU_TCP_OBS_TIME_SOURCE must be 'ctrl' or 'rt', "
        f"got {OBS_TIME_SOURCE}"
    )

TCP_FLOAT_ENDIAN = os.getenv("PNEU_TCP_FLOAT_ENDIAN", "little").strip().lower()
if TCP_FLOAT_ENDIAN not in ("little", "big"):
    raise ValueError(f"PNEU_TCP_FLOAT_ENDIAN must be 'little' or 'big', got {TCP_FLOAT_ENDIAN}")
FLOAT_ENDIAN_CHAR = "<" if TCP_FLOAT_ENDIAN == "little" else ">"

USE_LENGTH_PREFIX = os.getenv("PNEU_TCP_USE_LENGTH_PREFIX", "0").strip() in ("1", "true", "TRUE", "yes", "YES")
LENGTH_ENDIAN = os.getenv("PNEU_TCP_LENGTH_ENDIAN", "little").strip().lower()
if LENGTH_ENDIAN not in ("little", "big"):
    raise ValueError(f"PNEU_TCP_LENGTH_ENDIAN must be 'little' or 'big', got {LENGTH_ENDIAN}")
LENGTH_FMT = "<I" if LENGTH_ENDIAN == "little" else ">I"

FIELDS = FIELDS_22
PACK_COUNT = PACK_COUNT_22

CTRL_HEADER = ["seq"] + FIELDS
OBS_HEADER  = ["seq"] + FIELDS + ["rtt_ms", "loop_dt_ms", "loop_hz"]

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CTRL_JSON_FN  = os.path.join(BASE_DIR, "ctrl_act.json")
OBS_JSON_FN   = os.path.join(BASE_DIR, "obs_act.json")
CTRL_CSV_FN   = os.path.join(BASE_DIR, "ctrl_history.csv")
OBS_CSV_FN    = os.path.join(BASE_DIR, "obs_history.csv")


def read_ctrl_file():
    return read_ctrl_values(CTRL_JSON_FN, FIELDS)



def main():
    seq = 0
    last_recv_ns = None

    ctrl_rows = []  # CTRL CSV 누적 (보낸 값 로그)
    obs_rows  = []  # OBS  CSV 누적 (받은 값 로그)

    # time 기반 실험 구간 감지용 상태 변수
    prev_ctrl_time = None   # 직전 루프에서 읽은 ctrl time
    started = False         # real_act.py가 시작해서 time이 리셋됐는지 여부
    in_tail = False         # real_act 종료 후 time이 더 이상 변하지 않는 구간인지 여부
    stable_count = 0        # time이 변하지 않은 연속 step 수
    stale_start_wall = None # ctrl_time 정체 시작 wall-clock 시각

    os.makedirs(BASE_DIR, exist_ok=True)

    try:
        print(f"[INFO] Protocol: {PACK_COUNT} floats")
        print(f"[INFO] Float endian: {TCP_FLOAT_ENDIAN}")
        print(f"[INFO] Length prefix: {USE_LENGTH_PREFIX} (len endian: {LENGTH_ENDIAN})")
        print("[INFO] Payload layout: with_ref + flowrate(6)")
        print(f"[INFO] Obs time source: {OBS_TIME_SOURCE}")
        print(f"서버 연결 중... ({SERVER_IP}:{SERVER_PORT})")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            # 지연 최소화를 위해 Nagle 끔
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.connect((SERVER_IP, SERVER_PORT))
            client.settimeout(RECV_TIMEOUT_SEC)
            print("연결 성공. 서버 준비 대기 0.5초...")
            time.sleep(0.5)

            while True:
                # -------------------------------
                # 1) 제어 메시지 읽기 및 time 기반 상태 업데이트
                # -------------------------------
                ctrl_msg = read_ctrl_file()
                ctrl_time = float(ctrl_msg[0])

                if prev_ctrl_time is None:
                    # 첫 루프: 비교 기준만 저장
                    prev_ctrl_time = ctrl_time
                    in_tail = False
                else:
                    if not started:
                        # 아직 real_act.py가 안 돌아갈 때:
                        # 예전 run의 time(예: 6.x...)이 그대로 유지되다가,
                        # real_act.py가 시작되면 time이 6.x -> 0.x 로 리셋됨.
                        if ctrl_time < prev_ctrl_time - 1e-3:
                            started = True
                            in_tail = False
                            stable_count = 0
                            seq = 0
                            ctrl_rows.clear()
                            obs_rows.clear()
                            print(
                                f"[INFO] time 리셋 감지: {prev_ctrl_time:.3f} → "
                                f"{ctrl_time:.3f} (이후부터 로그 저장)"
                            )
                    else:
                        # 이미 실험이 시작된 이후:
                        # time이 더 이상 변하지 않는 구간 감지
                        if abs(ctrl_time - prev_ctrl_time) < 1e-6:
                            stable_count += 1
                            in_tail = True
                            if stale_start_wall is None:
                                stale_start_wall = time.time()
                        else:
                            stable_count = 0
                            in_tail = False
                            stale_start_wall = None

                prev_ctrl_time = ctrl_time

                # 자동 종료는 기본 비활성화(0). 필요 시 초 단위로만 동작.
                if (
                    started
                    and AUTO_EXIT_STALE_SEC > 0
                    and stale_start_wall is not None
                    and (time.time() - stale_start_wall) >= AUTO_EXIT_STALE_SEC
                ):
                    print(
                        f"[INFO] time가 {time.time() - stale_start_wall:.2f}s 동안 "
                        f"{ctrl_time:.3f}로 고정 → auto-exit"
                    )
                    break

                # -------------------------------
                # 2) RT로 송신 (리틀엔디안 float PACK_COUNT개)
                # -------------------------------
                t_send = time.monotonic_ns()
                payload = struct.pack(FLOAT_ENDIAN_CHAR + "f" * PACK_COUNT, *ctrl_msg)
                if USE_LENGTH_PREFIX:
                    prefix = struct.pack(LENGTH_FMT, len(payload))
                    client.sendall(prefix + payload)
                else:
                    client.sendall(payload)

                # -------------------------------
                # 3) RT에서 수신 (float PACK_COUNT개)
                # -------------------------------
                encoded_obs = recv_packet(client)
                t_recv = time.monotonic_ns()
                if encoded_obs is None:
                    print("서버 연결 끊김 (recv_all 반환 None)")
                    break

                obs_msg = list(struct.unpack(FLOAT_ENDIAN_CHAR + "f" * PACK_COUNT, encoded_obs))
                rt_time_raw = float(obs_msg[0])
                ctrl_time_sent = float(ctrl_msg[0])
                # 기본은 ctrl time 축 유지. 필요 시 RT time으로 검증 가능.
                if OBS_TIME_SOURCE == "ctrl":
                    obs_msg[0] = ctrl_time_sent
                else:
                    obs_msg[0] = rt_time_raw

                # -------------------------------
                # 4) 타이밍 메트릭 계산
                # -------------------------------
                rtt_ms = (t_recv - t_send) / 1e6
                if last_recv_ns is not None:
                    loop_dt_ms = (t_recv - last_recv_ns) / 1e6
                    loop_hz = 1000.0 / loop_dt_ms if loop_dt_ms > 0 else 0.0
                else:
                    loop_dt_ms, loop_hz = 0.0, 0.0
                last_recv_ns = t_recv

                # -------------------------------
                # 5) obs_act.json 업데이트 (실시간 관측)
                # -------------------------------
                try:
                    obs_dict = {k: float(v) for k, v in zip(FIELDS, obs_msg)}
                    # Timestamp diagnostics for sync checks.
                    obs_dict["time_ctrl_sent"] = ctrl_time_sent
                    obs_dict["time_rt_raw"] = rt_time_raw
                    obs_dict["time_delta_ms"] = (rt_time_raw - ctrl_time_sent) * 1000.0
                    # Backward compatibility for readers that still expect chamber ref keys.
                    if "act_pos_ref" in obs_dict and "pos_ref" not in obs_dict:
                        obs_dict["pos_ref"] = float(obs_dict["act_pos_ref"])
                    if "act_neg_ref" in obs_dict and "neg_ref" not in obs_dict:
                        obs_dict["neg_ref"] = float(obs_dict["act_neg_ref"])
                    write_json_atomic(OBS_JSON_FN, obs_dict)
                except Exception as e:
                    print(f"[WARN] obs_act.json 쓰기 실패: {e}")

                # -------------------------------
                # 6) CSV 로그 누적 (실험 구간만)
                # -------------------------------
                if started and not in_tail:
                    seq += 1
                    ctrl_rows.append([seq] + ctrl_msg)  # 보낸 값
                    obs_rows.append([seq] + obs_msg + [rtt_ms, loop_dt_ms, loop_hz])  # 받은 값

                    # -------------------------------
                    # 7) 상태 출력 (50 step마다)
                    # -------------------------------
                    if seq % 50 == 0:
                        print(
                            f"[{seq:06d}] "
                            f"RTT={rtt_ms:.2f}ms | "
                            f"dt={loop_dt_ms:.2f}ms | "
                            f"{loop_hz:.1f}Hz"
                        )

    except KeyboardInterrupt:
        print("\n[INFO] 사용자에 의해 종료 요청 (KeyboardInterrupt)")
    except TimeoutError as e:
        print("\n========== 수신 타임아웃 ==========")
        print(f"{type(e).__name__}: {e}")
        print(
            "[HINT] 22-float 모드는 고정입니다. "
            "RT.vi 응답 송신 경로 또는 프레이밍/엔디안 설정을 확인하세요. "
            "(float_endian="
            f"{TCP_FLOAT_ENDIAN}, length_prefix={USE_LENGTH_PREFIX}, layout=with_ref+flowrate6)"
        )
        print("===================================\n")
    except Exception as e:
        print("\n========== 오류 ==========")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        print("==========================\n")
    finally:
        # 프로그램 종료 시 CSV 저장
        save_csv(CTRL_CSV_FN, CTRL_HEADER, ctrl_rows)
        save_csv(OBS_CSV_FN,  OBS_HEADER,  obs_rows)


if __name__ == "__main__":
    main()
