import csv
import json
import os
import socket
import struct
from typing import Sequence


FIELDS_22 = [
    "time",
    "pos_press",
    "neg_press",
    "act_pos_ref",
    "act_neg_ref",
    "pos_ctrl",
    "neg_ctrl",
    "act_pos_press",
    "act_neg_press",
    "act_pos_ctrl1",
    "act_pos_ctrl2",
    "act_neg_ctrl1",
    "act_neg_ctrl2",
    "angle",
    "angle_reference",
    "angular_vel",
    "flowrate1",
    "flowrate2",
    "flowrate3",
    "flowrate4",
    "flowrate5",
    "flowrate6",
]
PACK_COUNT_22 = len(FIELDS_22)
BYTES_NEEDED_22 = 4 * PACK_COUNT_22


def read_ctrl_values(
    ctrl_json_path: str,
    field_names: Sequence[str] = FIELDS_22,
) -> list[float]:
    """
    Read ctrl_act.json in the requested field order.
    Missing fields fall back to 0.0.
    """
    try:
        with open(ctrl_json_path, "r", encoding="utf-8") as f:
            ctrl = json.load(f)

        values: list[float] = []
        for field_name in field_names:
            if field_name == "act_pos_ref" and field_name not in ctrl:
                values.append(float(ctrl.get("pos_ref", 0.0)))
                continue
            if field_name == "act_neg_ref" and field_name not in ctrl:
                values.append(float(ctrl.get("neg_ref", 0.0)))
                continue
            values.append(float(ctrl.get(field_name, 0.0)))
        return values
    except Exception as exc:
        print(f"[WARN] ctrl json read failed: {exc}")
        return [0.0] * len(field_names)


def recv_all(sock: socket.socket, n_bytes: int) -> bytes | None:
    """Receive exactly n_bytes unless the socket closes or times out."""
    data = bytearray()
    while len(data) < n_bytes:
        try:
            packet = sock.recv(n_bytes - len(data))
        except socket.timeout as exc:
            raise TimeoutError(
                f"recv timeout while waiting for {n_bytes} bytes "
                f"(got {len(data)} bytes)"
            ) from exc
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def recv_packet(
    sock: socket.socket,
    *,
    payload_bytes: int,
    use_length_prefix: bool,
    length_fmt: str,
) -> bytes | None:
    if not use_length_prefix:
        return recv_all(sock, payload_bytes)

    hdr = recv_all(sock, 4)
    if hdr is None:
        return None

    (frame_len,) = struct.unpack(length_fmt, hdr)
    if frame_len <= 0:
        raise ValueError(f"invalid frame_len from RT: {frame_len}")

    data = recv_all(sock, frame_len)
    if data is None:
        return None
    if len(data) < payload_bytes:
        raise ValueError(f"frame too short: {len(data)} bytes (need >= {payload_bytes})")
    if len(data) > payload_bytes:
        return data[:payload_bytes]
    return data


def write_json_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def save_csv(fname: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    if not rows:
        print(f"{fname}: no data to save")
        return

    try:
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"{fname} saved ({len(rows)} rows)")
    except Exception as exc:
        print(f"[WARN] failed to save {fname}: {exc}")
