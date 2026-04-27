from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .monitor import GpuInfo, gpu_memory_for_pids, process_tree_pids, rss_mb_for_pids, visible_devices_env
from .spec import ExperimentSpec


@dataclass
class RunMetrics:
    experiment_id: str
    experiment_name: str
    mode: str
    status: str
    command: Union[str, List[str]]
    assigned_gpus: List[int]
    return_code: Optional[int]
    start_time: float
    end_time: float
    duration_s: float
    max_gpu_mem_mb: int = 0
    max_cpu_rss_mb: float = 0.0
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "success" and self.return_code == 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d


class CommandRunner:
    def __init__(self, log_root: Union[str, Path], poll_interval_s: float = 2.0):
        self.log_root = Path(log_root)
        self.poll_interval_s = poll_interval_s
        self.log_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        spec: ExperimentSpec,
        *,
        mode: str,
        assigned_gpus: Optional[Sequence[GpuInfo]] = None,
        timeout_s: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
        dry_run: bool = False,
    ) -> RunMetrics:
        assigned_gpus = list(assigned_gpus or [])
        gpu_indices = [g.index for g in assigned_gpus]
        gpu_uuids = {g.uuid for g in assigned_gpus}
        command = spec.rendered_command(mode)
        rendered_name = spec.rendered_name(mode)

        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{mode}_{spec.id}_{_safe_name(rendered_name)}"
        run_dir = self.log_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        metrics_path = run_dir / "metrics.json"

        start = time.time()
        if dry_run:
            metrics = RunMetrics(
                experiment_id=spec.id,
                experiment_name=rendered_name,
                mode=mode,
                status="dry_run",
                command=command,
                assigned_gpus=gpu_indices,
                return_code=None,
                start_time=start,
                end_time=start,
                duration_s=0.0,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
            _write_json(metrics_path, metrics.to_dict())
            return metrics

        env = os.environ.copy()
        env.update(visible_devices_env(gpu_indices))
        env.update(spec.env)
        env.update(extra_env or {})
        env["EXP_SCHEDULER_MODE"] = mode
        env["EXP_SCHEDULER_NAME"] = rendered_name
        env["EXP_SCHEDULER_EXPERIMENT_ID"] = spec.id

        proc: Optional[subprocess.Popen] = None
        max_gpu_mem_mb = 0
        max_cpu_rss_mb = 0.0
        return_code: Optional[int] = None
        status = "failed"
        error: Optional[str] = None

        try:
            with open(stdout_path, "w", encoding="utf-8") as stdout_f, open(stderr_path, "w", encoding="utf-8") as stderr_f:
                proc = subprocess.Popen(
                    command,
                    cwd=spec.cwd,
                    env=env,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    shell=spec.shell,
                    start_new_session=True,
                    text=True,
                )
                deadline = None if timeout_s is None else start + timeout_s
                while True:
                    return_code = proc.poll()
                    pids = process_tree_pids(proc.pid)
                    max_cpu_rss_mb = max(max_cpu_rss_mb, rss_mb_for_pids(pids))
                    if gpu_uuids:
                        max_gpu_mem_mb = max(max_gpu_mem_mb, gpu_memory_for_pids(pids, gpu_uuids))
                    if return_code is not None:
                        break
                    if deadline is not None and time.time() > deadline:
                        status = "timeout"
                        error = f"Process exceeded timeout_s={timeout_s}"
                        _terminate_process_group(proc)
                        return_code = proc.wait(timeout=10)
                        break
                    time.sleep(self.poll_interval_s)

            if status != "timeout":
                status = "success" if return_code == 0 else "failed"
        except Exception as exc:  # noqa: BLE001 - runner should convert exceptions to metrics
            error = repr(exc)
            status = "failed"
            if proc is not None and proc.poll() is None:
                _terminate_process_group(proc)
                try:
                    return_code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    return_code = None

        end = time.time()
        metrics = RunMetrics(
            experiment_id=spec.id,
            experiment_name=rendered_name,
            mode=mode,
            status=status,
            command=command,
            assigned_gpus=gpu_indices,
            return_code=return_code,
            start_time=start,
            end_time=end,
            duration_s=end - start,
            max_gpu_mem_mb=max_gpu_mem_mb,
            max_cpu_rss_mb=max_cpu_rss_mb,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error=error,
            metadata={
                "run_dir": str(run_dir),
                "cwd": spec.cwd,
                "timeout_s": timeout_s,
            },
        )
        _write_json(metrics_path, metrics.to_dict())
        return metrics


def _safe_name(name: str, max_len: int = 80) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return safe[:max_len].strip("_") or "experiment"


def _terminate_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, 15)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def _write_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
