from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .scheduler import ExperimentScheduler, SchedulerConfig
from .spec import ExperimentSpec, expand_grid


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def scheduler_config_from_dict(data: Dict[str, Any]) -> SchedulerConfig:
    allowed = {f.name for f in fields(SchedulerConfig)}
    kwargs = {k: v for k, v in (data or {}).items() if k in allowed}
    return SchedulerConfig(**kwargs)


def experiments_from_config(data: Dict[str, Any]) -> List[ExperimentSpec]:
    experiments: List[ExperimentSpec] = []
    for item in data.get("experiments", []):
        common = {
            "name": item["name"],
            "command": item["command"],
            "params": item.get("params", {}),
            "probe_params": item.get("probe_params", {"epochs": 1}),
            "run_params": item.get("run_params", {}),
            "env": item.get("env", {}),
            "cwd": item.get("cwd"),
            "shell": item.get("shell", False),
            "num_gpus": item.get("num_gpus", 1),
            "priority": item.get("priority", 0),
            "tags": item.get("tags", []),
            "metadata": item.get("metadata", {}),
            "timeout_s": item.get("timeout_s"),
        }
        if "grid" in item:
            experiments.extend(expand_grid(grid=item.get("grid") or {}, **common))
        else:
            experiments.append(ExperimentSpec(**common))
    return experiments


def print_event(event: str, payload: Dict[str, Any]) -> None:
    if event == "prepare_start":
        print(f"[prepare] {payload['num_experiments']} experiments")
    elif event == "probe_start":
        print(f"[probe {payload['index']}/{payload['total']}] {payload['experiment']} gpus={payload['gpus']}")
    elif event == "probe_skip":
        print(f"[probe skip {payload['index']}/{payload['total']}] {payload['experiment']}")
    elif event == "probe_end":
        print(
            f"[probe done] {payload['experiment_name']} status={payload['status']} "
            f"sec={payload['duration_s']:.1f} gpu_mem={payload['max_gpu_mem_mb']}MB"
        )
    elif event == "gpu_schedule_start":
        print(f"[schedule] gpu budgets={payload['budgets_mb']}")
    elif event == "cpu_schedule_start":
        print(f"[schedule] cpu-only max_parallel_cpu={payload['max_parallel_cpu']}")
    elif event == "run_start":
        print(
            f"[run] {payload['experiment']} gpus={payload['gpus']} "
            f"est_mem={payload['estimated_mem_per_gpu_mb']}MB"
        )
    elif event == "run_start_oversized":
        print(
            f"[run oversized] {payload['experiment']} gpus={payload['gpus']} "
            f"est_mem={payload['estimated_mem_per_gpu_mb']}MB"
        )
    elif event == "run_skip":
        print(f"[run skip] {payload['experiment']} reason={payload['reason']}")
    elif event == "run_end":
        print(
            f"[run done] {payload['experiment_name']} status={payload['status']} "
            f"sec={payload['duration_s']:.1f} gpu_mem={payload['max_gpu_mem_mb']}MB"
        )


def cmd_prepare(args: argparse.Namespace) -> int:
    data = load_config(args.config)
    cfg = scheduler_config_from_dict(data.get("scheduler", {}))
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.dry_run:
        cfg.dry_run = True
    scheduler = ExperimentScheduler(cfg, on_event=print_event)
    experiments = experiments_from_config(data)
    metrics = scheduler.prepare(experiments, resume=not args.no_resume)
    print(json.dumps([m.to_dict() for m in metrics], ensure_ascii=False, indent=2))
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    data = load_config(args.config)
    cfg = scheduler_config_from_dict(data.get("scheduler", {}))
    if args.work_dir:
        cfg.work_dir = args.work_dir
    scheduler = ExperimentScheduler(cfg, on_event=print_event)
    experiments = experiments_from_config(data)
    estimates = scheduler.estimate(experiments)
    print(json.dumps({k: v.to_dict() for k, v in estimates.items()}, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    data = load_config(args.config)
    cfg = scheduler_config_from_dict(data.get("scheduler", {}))
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.dry_run:
        cfg.dry_run = True
    scheduler = ExperimentScheduler(cfg, on_event=print_event)
    experiments = experiments_from_config(data)
    summary = scheduler.run(experiments, prepare_first=not args.no_prepare, resume=not args.no_resume)
    summary_path = Path(cfg.work_dir) / "last_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"[summary] {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exp-scheduler", description="Profile and schedule ML experiment commands.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("config", help="YAML/JSON experiment config path")
        p.add_argument("--work-dir", default=None, help="Override scheduler.work_dir")
        p.add_argument("--no-resume", action="store_true", help="Do not reuse successful previous probes/runs")
        p.add_argument("--dry-run", action="store_true", help="Render commands and scheduling state without executing commands")

    p_prepare = sub.add_parser("prepare", help="Run probe stage only")
    add_common(p_prepare)
    p_prepare.set_defaults(func=cmd_prepare)

    p_estimate = sub.add_parser("estimate", help="Estimate resources from existing probe metrics")
    p_estimate.add_argument("config", help="YAML/JSON experiment config path")
    p_estimate.add_argument("--work-dir", default=None, help="Override scheduler.work_dir")
    p_estimate.set_defaults(func=cmd_estimate)

    p_run = sub.add_parser("run", help="Run probes, then full scheduled experiments")
    add_common(p_run)
    p_run.add_argument("--no-prepare", action="store_true", help="Skip probe stage and use existing state.json")
    p_run.set_defaults(func=cmd_run)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
