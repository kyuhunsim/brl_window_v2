#!/usr/bin/env python3
"""Render or execute the gated lib4 follow-up experiment manifest."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "pneu_rl/config/experiments/lib4_followup_plan.yaml"


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def option_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def append_options(command: list[str], options: dict[str, Any]) -> None:
    for key, value in options.items():
        if value is None or value is False:
            continue
        command.append(option_name(key))
        if value is not True:
            command.append(str(value))


def load_plan(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        plan = yaml.safe_load(handle)
    if plan.get("schema") != "lib4_followup_plan_v1":
        raise ValueError(f"unsupported follow-up plan: {path}")
    return plan


def selected_jobs(
    plan: dict[str, Any],
    phases: set[str],
    job_names: set[str],
    include_gated: bool,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for phase_name, phase in plan["phases"].items():
        if phases and phase_name not in phases:
            continue
        jobs = [
            job
            for job in phase["jobs"]
            if (bool(job.get("enabled", False)) or include_gated)
            and (not job_names or str(job["name"]) in job_names)
        ]
        if jobs:
            selected[phase_name] = jobs
    return selected


def train_command(plan: dict[str, Any], job: dict[str, Any], python: str) -> list[str]:
    command = [
        python,
        str(resolve(ROOT, plan["train_script"])),
        "--variant",
        str(job["variant"]),
        "--reference-profile",
        str(resolve(ROOT, plan["train_reference"])),
        "--seed",
        str(job["seed"]),
        "--episodes",
        str(job["episodes"]),
        "--name",
        str(job["name"]),
    ]
    append_options(command, job.get("args", {}))
    return command


def evaluation_command(
    plan: dict[str, Any],
    phase_name: str,
    jobs: list[dict[str, Any]],
    python: str,
) -> list[str]:
    return [
        python,
        str(resolve(ROOT, plan["evaluate_script"])),
        "--models",
        *[str(job["name"]) for job in jobs],
        "--reference-profile",
        str(resolve(ROOT, plan["evaluation_reference"])),
        "--protocol",
        str(resolve(ROOT, plan["evaluation_protocol"])),
        "--output-dir",
        str(resolve(ROOT, plan["evaluation_root"]) / phase_name),
    ]


def run_or_print(command: list[str], execute: bool) -> None:
    print(shlex.join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--job", action="append", default=[])
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--python", default=None)
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan = load_plan(plan_path)
    python = args.python or str(plan["python"])
    phases = selected_jobs(plan, set(args.phase), set(args.job), args.include_gated)
    if not phases:
        raise SystemExit("no enabled jobs selected")

    print(f"plan={plan_path}")
    for phase_name, jobs in phases.items():
        phase_cfg = plan["phases"][phase_name]
        print(f"phase={phase_name} gate={phase_cfg['gate']}")
        runnable: list[dict[str, Any]] = []
        if args.stage in {"train", "all"}:
            for job in jobs:
                model_dir = ROOT / "pneu_rl/models" / str(job["name"])
                if model_dir.exists():
                    if args.skip_existing:
                        print(f"skip-existing: {job['name']}")
                        runnable.append(job)
                        continue
                    if args.execute:
                        raise FileExistsError(
                            f"refusing to overwrite existing model directory: {model_dir}"
                        )
                run_or_print(train_command(plan, job, python), args.execute)
                runnable.append(job)
        else:
            runnable = jobs

        if args.stage in {"evaluate", "all"}:
            missing = [
                str(job["name"])
                for job in runnable
                if not (ROOT / "pneu_rl/models" / str(job["name"]) / "model.pth").is_file()
            ]
            if args.execute and missing:
                raise FileNotFoundError(
                    "cannot evaluate missing checkpoints: " + ", ".join(missing)
                )
            run_or_print(evaluation_command(plan, phase_name, runnable, python), args.execute)


if __name__ == "__main__":
    main()
