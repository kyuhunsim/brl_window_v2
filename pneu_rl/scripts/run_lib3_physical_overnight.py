#!/usr/bin/env python3
"""Supervise fresh type1/type2 H/HR runs, with persistent logs and completion checks.

Use --dry-run to inspect commands, --detach to keep the supervisor alive after the
terminal closes, and --status RUN_DIR to inspect its recorded state. This script
does not change training settings beyond the explicit train entrypoint arguments.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = "/home/rise/miniconda3/envs/env_1/bin/python"


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completion_error(job):
    model = Path(job["model_dir"])
    wall_path = model / "wall_time.json"
    if not wall_path.is_file():
        return "Training exited without wall_time.json"
    timing = read_json(wall_path)
    if not timing.get("completed"):
        return "Training process exited without completed=true"
    if timing.get("episodes_requested") != job["episodes"]:
        return "Requested episode count differs from the training record"
    checkpoint = model / f'model_epi_{job["episodes"]:04d}.pth'
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return f"Missing or empty final checkpoint: {checkpoint.name}"
    return None


def worker(plan_path):
    plan = read_json(plan_path)
    run_dir = Path(plan_path).parent
    status_path = run_dir / "status.json"
    state = dict(supervisor_pid=os.getpid(), started=now(), state="running",
                 max_parallel=plan["max_parallel"], jobs=plan["jobs"])
    running = {}
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS=str(plan["threads"]),
                       MKL_NUM_THREADS=str(plan["threads"]),
                       OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1",
                       MPLCONFIGDIR=str(run_dir / "matplotlib"))
    local_sources = [str(ROOT / package / "src") for package in
                     ("pneu_env", "pneu_ref", "pneu_rl", "pneu_utils")]
    environment["PYTHONPATH"] = os.pathsep.join(local_sources +
                                              [environment.get("PYTHONPATH", "")])
    Path(environment["MPLCONFIGDIR"]).mkdir(exist_ok=True)
    while True:
        failed = any(job["state"] == "failed" for job in state["jobs"])
        for index, (process, output) in list(running.items()):
            result = process.poll()
            if result is None:
                if interrupted:
                    process.terminate()
                continue
            output.close()
            job = state["jobs"][index]
            error = None
            if result == 0:
                try:
                    error = completion_error(job)
                except (OSError, ValueError) as exc:
                    error = str(exc)
            else:
                error = f"Training process exited with code {result}"
            job.update(state="failed" if error else "completed", returncode=result,
                       finished=now())
            if error:
                job["error"] = error
                failed = True
            print(f'{now()} {job["name"]}: {job["state"]}', flush=True)
            del running[index]
        for index, job in enumerate(state["jobs"]):
            if job["state"] != "queued":
                continue
            if interrupted or failed:
                job.update(state="cancelled", finished=now(),
                           error="Supervisor interrupted" if interrupted else
                           "Earlier job failed; remaining jobs were not started")
                continue
            if len(running) >= plan["max_parallel"]:
                break
            error = None
            if Path(job["model_dir"]).exists():
                error = "Model directory appeared after planning; refusing overwrite"
            for artifact, expected in job["input_sha256"].items():
                if not Path(artifact).is_file() or digest(artifact) != expected:
                    error = f"Input changed after planning: {artifact}"
                    break
            if error:
                job.update(state="failed", error=error, finished=now())
                failed = True
                continue
            output = open(job["log"], "xb", buffering=0)
            try:
                process = subprocess.Popen(job["command"], cwd=ROOT,
                                           env=environment, stdout=output,
                                           stderr=subprocess.STDOUT)
            except OSError as exc:
                output.close()
                job.update(state="failed", error=str(exc), finished=now())
                failed = True
                continue
            job.update(state="running", pid=process.pid, started=now())
            running[index] = process, output
            print(f'{now()} started {job["name"]} PID={process.pid}', flush=True)
        state["updated"] = now()
        if not running and not any(job["state"] == "queued" for job in state["jobs"]):
            state.update(state="completed" if all(job["state"] == "completed"
                                                  for job in state["jobs"]) else
                         "failed", finished=now())
            write_json(status_path, state)
            return 0 if state["state"] == "completed" else 1
        write_json(status_path, state)
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type1-library")
    parser.add_argument("--type2-library")
    parser.add_argument("--type1-reference")
    parser.add_argument("--type2-reference")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--prefix", default="261023_lib3_opening")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--precondition-seconds", type=float, default=30)
    parser.add_argument("--run-dir", default="pneu_rl/exp/lib3_physical_overnight_20261023")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", metavar="RUN_DIR")
    parser.add_argument("--worker-plan", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan:
        return worker(Path(args.worker_plan).resolve())
    if args.status:
        print(json.dumps(read_json(Path(args.status) / "status.json"), indent=2,
                         ensure_ascii=False))
        return 0
    if args.episodes < 1 or args.episodes % 500:
        parser.error("episodes must be a positive multiple of 500 for checkpoint checks")
    if args.threads < 1 or args.precondition_seconds < 0:
        parser.error("threads must be positive and precondition seconds nonnegative")
    if not args.prefix or Path(args.prefix).name != args.prefix:
        parser.error("prefix must be a plain model folder prefix")
    python = Path(args.python).resolve()
    if not python.is_file():
        parser.error(f"Python interpreter does not exist: {python}")
    run_dir = (ROOT / args.run_dir).resolve()
    if run_dir.exists():
        parser.error(f"Run directory already exists: {run_dir}; use --status to inspect it")
    jobs = []
    train_script = ROOT / "pneu_rl/scripts/train_lib3_all_open.py"
    for kind in ("type1", "type2"):
        files = {}
        for field in ("library", "reference"):
            value = getattr(args, f"{kind}_{field}")
            if not value:
                parser.error(f"--{kind}-{field} is required")
            path = (ROOT / value).resolve()
            if not path.is_file():
                parser.error(f"Input does not exist: {path}")
            files[field] = path
        try:
            profile = yaml.safe_load(files["reference"].read_text())
            bank_path = (ROOT / profile["kwargs"]["trajectory_npz"]).resolve()
        except (KeyError, TypeError, OSError, yaml.YAMLError) as exc:
            parser.error(f"Invalid {kind} reference bank path: {exc}")
        if not bank_path.is_file():
            parser.error(f"Reference bank does not exist: {bank_path}")
        files["bank"] = bank_path
        for policy in ("H", "HR"):
            name = f"{args.prefix}_{kind}_{policy}_smooth_seed0"
            model_dir = ROOT / "pneu_rl/models" / name
            if model_dir.exists():
                parser.error(f"Refusing to overwrite existing model directory: {model_dir}")
            source = ROOT / f"pneu_rl/models/260831_lib3_histreward2x2_r025_{policy}_smooth_seed0/cfg.yaml"
            hashes = {str(path): digest(path) for path in
                      (train_script, source, files["library"], files["reference"],
                       files["bank"])}
            command = [str(python), "-u", str(train_script), "--policy", policy,
                       "--episodes", str(args.episodes), "--threads", str(args.threads),
                       "--precondition-seconds", str(args.precondition_seconds),
                       "--name", name, "--library-path", str(files["library"]),
                       "--reference-profile", str(files["reference"])]
            jobs.append(dict(name=name, kind=kind, policy=policy, state="queued",
                             episodes=args.episodes, command=command,
                             model_dir=str(model_dir), log=str(run_dir / f"{kind}_{policy}.log"),
                             input_sha256=hashes))
    provenance_paths = [ROOT / "pneu_env/src/pneu_env" / relative for relative in
                        ("lib3/pneumatic_CT.cpp", "lib3/pneumatic_simulator.cpp",
                         "sim3.py", "env3.py", "env3_all_open.py")]
    plan = dict(created=now(), max_parallel=args.max_parallel, threads=args.threads,
                run_dir=str(run_dir), jobs=jobs,
                source_provenance_sha256={str(path): digest(path)
                                          for path in provenance_paths if path.is_file()})
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    run_dir.mkdir(parents=True)
    plan_path = run_dir / "plan.json"
    write_json(plan_path, plan)
    if args.detach:
        with (run_dir / "supervisor.log").open("xb") as output:
            process = subprocess.Popen([str(python), "-u", str(Path(__file__).resolve()),
                                        "--worker-plan", str(plan_path)], cwd=ROOT,
                                       stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        (run_dir / "supervisor.pid").write_text(str(process.pid) + "\n")
        print(f"Supervisor PID={process.pid}; verify running state in {run_dir / 'status.json'}")
        return 0
    return worker(plan_path)


if __name__ == "__main__":
    sys.exit(main())
