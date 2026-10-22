"""Run the saved lib3 valve baseline against one measured collection CSV.

This is an offline fit. No hardware interfaces or live control files are used.
The historical v3/v4 opening-shape penalty has not been recovered.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "pneu_env"
OPTIMIZER = PACKAGE / "src/pneu_env/optimizer.py"
DEFAULT_PRESET = PACKAGE / "config/tuning/lib3_valve_baseline.json"
PRESSURES = ("press_pos", "press_neg", "act_pos_press", "act_neg_press")
REQUIRED = ("time",) + PRESSURES + tuple(
    f"{prefix}{i}" for prefix in ("ctrl", "flow") for i in range(1, 7)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_data(path: Path) -> dict:
    """Check the actual echo/pressure/flow columns used by the existing fit."""
    times = []
    limits = {name: [math.inf, -math.inf] for name in REQUIRED[1:]}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"실측 CSV 필수 열 누락: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            for name in REQUIRED:
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"CSV {line}행 {name}: 숫자가 필요합니다.") from exc
                if not math.isfinite(value):
                    raise ValueError(f"CSV {line}행 {name}: 유한한 값이 필요합니다.")
                if name == "time":
                    if times and value < times[-1]:
                        raise ValueError(f"CSV {line}행: time이 역행합니다.")
                    times.append(value)
                else:
                    limits[name][0] = min(limits[name][0], value)
                    limits[name][1] = max(limits[name][1], value)
                    if name in PRESSURES and value <= 0:
                        raise ValueError(f"CSV {line}행 {name}: kPa 절대압(>0)이 필요합니다.")
                    if name.startswith("ctrl") and not 0 <= value <= 1:
                        raise ValueError(f"CSV {line}행 {name}: 명령은 0~1이어야 합니다.")
    if len(times) < 2:
        raise ValueError("튜닝에는 실측 CSV가 최소 2행 필요합니다.")
    differences = [b - a for a, b in zip(times, times[1:])]
    positive_differences = [dt for dt in differences if dt > 0]
    if not positive_differences:
        raise ValueError("CSV time이 모두 같습니다.")
    median_dt = statistics.median(positive_differences)
    return {
        "rows": len(times),
        "time_start_sec": times[0],
        "time_end_sec": times[-1],
        "duration_sec": times[-1] - times[0],
        "median_dt_sec": median_dt,
        "optimizer_dt_sec": (times[-1] - times[0]) / (len(times) - 1),
        "max_dt_sec": max(differences),
        "duplicate_timestamp_count": sum(dt == 0 for dt in differences),
        "dt_gap_count_over_1_5_median": sum(dt > 1.5 * median_dt for dt in differences),
        "column_ranges": limits,
        "command_columns": [f"ctrl{i}" for i in range(1, 7)],
    }


def optimizer_command(data: Path, preset: dict, result_dir: Path, show: bool) -> list:
    command = [sys.executable, str(OPTIMIZER), str(data)]
    for name, value in preset["optimizer_options"].items():
        option = "--" + name.replace("_", "-")
        if value is True:
            command.append(option)
        elif value is not False and value is not None:
            command.extend([option, str(value)])
    command.extend(["--result-dir", str(result_dir), "--output", "valves.txt"])
    if not show:
        command.append("--no-show")
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Type 1 또는 Type 2 실측 CSV")
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--result-dir", type=Path, default=PACKAGE / "tune_result/lib3_valves")
    parser.add_argument("--dry-run", action="store_true", help="CSV·설정 확인과 명령 출력만 수행; 파일 생성·최적화 없음")
    parser.add_argument("--show", action="store_true", help="최적화 종료 후 그림 창 표시")
    args = parser.parse_args(argv)
    data = args.data.expanduser().resolve()
    preset_path = args.preset.expanduser().resolve()
    try:
        preset = json.loads(preset_path.read_text(encoding="utf-8"))
        summary = inspect_data(data)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    result_dir = args.result_dir.expanduser().resolve() / data.stem / stamp
    command = optimizer_command(data, preset, result_dir, args.show)
    print("[설정] 확인된 과거 조건 + 현재 탐색 기본값을 사용한 baseline입니다.")
    print("[미복원] 과거 v3/v4 opening-shape penalty(weight=5)의 면적비 구간·구현이 빠져 있습니다.")
    print(f"[데이터] {data}\n{json.dumps(summary, ensure_ascii=False, indent=2)}")
    print(f"[실행] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return 0

    source_paths = [
        Path(__file__).resolve(), preset_path, OPTIMIZER,
        PACKAGE / "src/pneu_env/tuner/sim_core.cpp",
        PACKAGE / "src/pneu_env/tuner/sim_core.so",
    ]
    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(data), "data_sha256": sha256(data),
        "summary": summary, "preset": preset,
        "sources_sha256": {str(path): sha256(path) for path in source_paths},
        "optimizer_command": command,
        "python": sys.version,
    }
    collection_metadata = data.with_suffix(".json")
    if collection_metadata.is_file():
        manifest["collection_metadata"] = json.loads(collection_metadata.read_text(encoding="utf-8"))
        manifest["sources_sha256"][str(collection_metadata)] = sha256(collection_metadata)

    result_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = result_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    if not args.show:
        environment["MPLBACKEND"] = "Agg"
    try:
        completed = subprocess.run(command, cwd=str(ROOT), env=environment, check=False)
        manifest["returncode"] = completed.returncode
        manifest["status"] = "completed" if completed.returncode == 0 else "failed"
        return completed.returncode
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["outputs_sha256"] = {
            path.name: sha256(path) for path in result_dir.iterdir()
            if path.is_file() and path != manifest_path
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[저장] {result_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
