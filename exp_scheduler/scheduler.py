from __future__ import annotations

import concurrent.futures as cf
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .monitor import GpuInfo, current_environment_summary, detect_gpus
from .runner import CommandRunner, RunMetrics
from .spec import ExperimentSpec

EventCallback = Callable[[str, Dict[str, Any]], None]


@dataclass
class SchedulerConfig:
    work_dir: str = "runs"
    probe_timeout_s: Optional[float] = None
    run_timeout_s: Optional[float] = None
    poll_interval_s: float = 2.0
    max_gpu_memory_fraction: float = 0.90
    reserve_gpu_memory_mb: int = 512
    safety_margin_mb: int = 768
    unknown_gpu_mem_mb: int = 4096
    min_gpu_mem_mb: int = 256
    max_parallel_cpu: int = 1
    skip_failed_probes: bool = True
    allow_oversized_single_job: bool = True
    retry_failed_runs: int = 0
    dry_run: bool = False
    sort_by: str = "priority_then_slowest"  # or "input"


@dataclass
class ResourceEstimate:
    experiment_id: str
    experiment_name: str
    memory_per_gpu_mb: int
    probe_duration_s: float
    expected_num_gpus: int
    probe_status: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScheduleSummary:
    prepared: List[RunMetrics] = field(default_factory=list)
    skipped_probe: List[str] = field(default_factory=list)
    estimates: Dict[str, ResourceEstimate] = field(default_factory=dict)
    runs: List[RunMetrics] = field(default_factory=list)
    skipped_run: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prepared": [m.to_dict() for m in self.prepared],
            "skipped_probe": self.skipped_probe,
            "estimates": {k: v.to_dict() for k, v in self.estimates.items()},
            "runs": [m.to_dict() for m in self.runs],
            "skipped_run": self.skipped_run,
        }


class ExperimentScheduler:
    """Profile experiments with 1-epoch probes, then schedule full runs by GPU memory estimates."""

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        *,
        on_event: Optional[EventCallback] = None,
    ):
        self.config = config or SchedulerConfig()
        self.work_dir = Path(self.config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.runner = CommandRunner(self.work_dir / "logs", poll_interval_s=self.config.poll_interval_s)
        self.state_path = self.work_dir / "state.json"
        self.on_event = on_event
        self.state = self._load_state()

    def prepare(self, experiments: Sequence[ExperimentSpec], *, resume: bool = True) -> List[RunMetrics]:
        """Run every experiment in probe mode, usually with epochs=1."""
        self._emit("prepare_start", {"num_experiments": len(experiments)})
        gpus = detect_gpus()
        self.state.setdefault("environment", current_environment_summary())
        self.state.setdefault("probe", {})
        prepared: List[RunMetrics] = []

        for i, spec in enumerate(experiments, start=1):
            previous = self.state["probe"].get(spec.id)
            if resume and previous and previous.get("status") == "success":
                self._emit("probe_skip", {"index": i, "total": len(experiments), "experiment": spec.rendered_name("probe")})
                continue

            assigned = self._select_probe_gpus(gpus, spec.num_gpus)
            self._emit(
                "probe_start",
                {
                    "index": i,
                    "total": len(experiments),
                    "experiment": spec.rendered_name("probe"),
                    "gpus": [g.index for g in assigned],
                },
            )
            metrics = self.runner.run(
                spec,
                mode="probe",
                assigned_gpus=assigned,
                timeout_s=self.config.probe_timeout_s or spec.timeout_s,
                dry_run=self.config.dry_run,
            )
            prepared.append(metrics)
            self.state["probe"][spec.id] = metrics.to_dict()
            self._save_state()
            self._emit("probe_end", metrics.to_dict())
        return prepared

    def estimate(self, experiments: Sequence[ExperimentSpec]) -> Dict[str, ResourceEstimate]:
        """Create resource estimates from probe metrics."""
        probe_state = self.state.get("probe", {})
        estimates: Dict[str, ResourceEstimate] = {}
        gpus_available = bool(detect_gpus())

        for spec in experiments:
            probe = probe_state.get(spec.id)
            if not probe:
                estimates[spec.id] = ResourceEstimate(
                    experiment_id=spec.id,
                    experiment_name=spec.rendered_name("run"),
                    memory_per_gpu_mb=self.config.unknown_gpu_mem_mb,
                    probe_duration_s=math.inf,
                    expected_num_gpus=max(0, spec.num_gpus),
                    probe_status="missing",
                    reason="No probe metric found.",
                )
                continue

            status = str(probe.get("status", "unknown"))
            duration = float(probe.get("duration_s") or 0.0)
            num_gpus = max(0, int(spec.num_gpus))
            raw_gpu_mem = int(probe.get("max_gpu_mem_mb") or 0)

            if num_gpus <= 0 or not gpus_available:
                mem_per_gpu = 0
            elif raw_gpu_mem <= 0:
                mem_per_gpu = self.config.unknown_gpu_mem_mb
            else:
                mem_per_gpu = int(math.ceil(raw_gpu_mem / max(1, num_gpus)))
                mem_per_gpu = max(mem_per_gpu, self.config.min_gpu_mem_mb)
                mem_per_gpu += self.config.safety_margin_mb

            estimates[spec.id] = ResourceEstimate(
                experiment_id=spec.id,
                experiment_name=spec.rendered_name("run"),
                memory_per_gpu_mb=mem_per_gpu,
                probe_duration_s=duration,
                expected_num_gpus=num_gpus,
                probe_status=status,
                reason="ok" if status == "success" else "probe did not finish successfully",
            )
        self.state["estimates"] = {k: v.to_dict() for k, v in estimates.items()}
        self._save_state()
        return estimates

    def run(
        self,
        experiments: Sequence[ExperimentSpec],
        *,
        prepare_first: bool = True,
        resume: bool = True,
    ) -> ScheduleSummary:
        """Run full experiments. By default, missing probes are executed first."""
        summary = ScheduleSummary()
        if prepare_first:
            summary.prepared = self.prepare(experiments, resume=resume)

        estimates = self.estimate(experiments)
        summary.estimates = estimates
        gpus = detect_gpus()
        self.state.setdefault("runs", {})

        runnable: List[ExperimentSpec] = []
        for spec in experiments:
            previous_runs = self.state["runs"].get(spec.id, [])
            if resume and any(r.get("status") == "success" for r in previous_runs):
                summary.skipped_run.append(spec.id)
                self._emit("run_skip", {"experiment": spec.rendered_name("run"), "reason": "already succeeded"})
                continue
            est = estimates[spec.id]
            if self.config.skip_failed_probes and est.probe_status not in {"success", "dry_run"}:
                summary.skipped_run.append(spec.id)
                self._emit("run_skip", {"experiment": spec.rendered_name("run"), "reason": est.reason})
                continue
            runnable.append(spec)

        if self.config.sort_by == "priority_then_slowest":
            runnable.sort(key=lambda s: (-s.priority, -safe_duration(estimates[s.id].probe_duration_s), -estimates[s.id].memory_per_gpu_mb))

        if not gpus:
            summary.runs.extend(self._run_cpu_only(runnable))
        else:
            summary.runs.extend(self._run_gpu_scheduled(runnable, gpus, estimates))

        self._emit("schedule_end", summary.to_dict())
        return summary

    def _run_cpu_only(self, experiments: Sequence[ExperimentSpec]) -> List[RunMetrics]:
        results: List[RunMetrics] = []
        max_workers = max(1, self.config.max_parallel_cpu)
        self._emit("cpu_schedule_start", {"num_experiments": len(experiments), "max_parallel_cpu": max_workers})
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_spec = {
                executor.submit(
                    self.runner.run,
                    spec,
                    mode="run",
                    assigned_gpus=[],
                    timeout_s=self.config.run_timeout_s or spec.timeout_s,
                    dry_run=self.config.dry_run,
                ): spec
                for spec in experiments
            }
            for future in cf.as_completed(future_to_spec):
                spec = future_to_spec[future]
                metrics = future.result()
                results.append(metrics)
                self._append_run_state(spec.id, metrics)
                self._emit("run_end", metrics.to_dict())
        return results

    def _run_gpu_scheduled(
        self,
        experiments: Sequence[ExperimentSpec],
        gpus: Sequence[GpuInfo],
        estimates: Dict[str, ResourceEstimate],
    ) -> List[RunMetrics]:
        budgets = self._gpu_budgets(gpus)
        reserved = {g.index: 0 for g in gpus}
        gpu_by_index = {g.index: g for g in gpus}
        pending = list(experiments)
        active: Dict[cf.Future, Tuple[ExperimentSpec, List[GpuInfo], int, int]] = {}
        results: List[RunMetrics] = []
        max_workers = max(1, len(experiments))

        self._emit(
            "gpu_schedule_start",
            {
                "num_experiments": len(experiments),
                "gpus": [g.__dict__ for g in gpus],
                "budgets_mb": budgets,
            },
        )

        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pending or active:
                launched_any = False
                for spec in list(pending):
                    allocation = self._try_allocate(spec, estimates[spec.id], gpus, budgets, reserved)
                    if allocation is None:
                        continue
                    assigned, mem_per_gpu, attempt = allocation
                    for g in assigned:
                        reserved[g.index] += mem_per_gpu
                    future = executor.submit(
                        self._run_with_retries,
                        spec,
                        assigned,
                        attempt,
                    )
                    active[future] = (spec, assigned, mem_per_gpu, attempt)
                    pending.remove(spec)
                    launched_any = True
                    self._emit(
                        "run_start",
                        {
                            "experiment": spec.rendered_name("run"),
                            "gpus": [g.index for g in assigned],
                            "estimated_mem_per_gpu_mb": mem_per_gpu,
                            "reserved_mb": dict(reserved),
                        },
                    )

                if not active:
                    # No job fits by budget. Run the largest remaining job alone on the GPU(s)
                    # with the most budget if oversized jobs are allowed.
                    if pending and self.config.allow_oversized_single_job:
                        spec = pending.pop(0)
                        needed = max(1, int(spec.num_gpus))
                        assigned = sorted(gpus, key=lambda g: budgets.get(g.index, 0), reverse=True)[:needed]
                        mem_per_gpu = estimates[spec.id].memory_per_gpu_mb
                        for g in assigned:
                            reserved[g.index] += mem_per_gpu
                        future = executor.submit(self._run_with_retries, spec, assigned, 0)
                        active[future] = (spec, assigned, mem_per_gpu, 0)
                        self._emit(
                            "run_start_oversized",
                            {
                                "experiment": spec.rendered_name("run"),
                                "gpus": [g.index for g in assigned],
                                "estimated_mem_per_gpu_mb": mem_per_gpu,
                            },
                        )
                    else:
                        break

                done, _ = cf.wait(active.keys(), timeout=0.5 if launched_any else 5.0, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    spec, assigned, mem_per_gpu, attempt = active.pop(future)
                    for g in assigned:
                        reserved[g.index] = max(0, reserved[g.index] - mem_per_gpu)
                    try:
                        metrics = future.result()
                    except Exception as exc:  # noqa: BLE001
                        now = time.time()
                        metrics = RunMetrics(
                            experiment_id=spec.id,
                            experiment_name=spec.rendered_name("run"),
                            mode="run",
                            status="failed",
                            command=spec.rendered_command("run"),
                            assigned_gpus=[g.index for g in assigned],
                            return_code=None,
                            start_time=now,
                            end_time=now,
                            duration_s=0.0,
                            error=repr(exc),
                        )
                    results.append(metrics)
                    self._append_run_state(spec.id, metrics)
                    self._emit(
                        "run_end",
                        {
                            **metrics.to_dict(),
                            "reserved_mb": dict(reserved),
                            "gpu_names": {g.index: gpu_by_index[g.index].name for g in assigned},
                        },
                    )
        return results

    def _run_with_retries(self, spec: ExperimentSpec, assigned: Sequence[GpuInfo], first_attempt: int) -> RunMetrics:
        max_attempts = max(1, self.config.retry_failed_runs + 1)
        last_metrics: Optional[RunMetrics] = None
        for attempt in range(first_attempt, max_attempts):
            metrics = self.runner.run(
                spec,
                mode="run",
                assigned_gpus=assigned,
                timeout_s=self.config.run_timeout_s or spec.timeout_s,
                extra_env={"EXP_SCHEDULER_ATTEMPT": str(attempt)},
                dry_run=self.config.dry_run,
            )
            last_metrics = metrics
            if metrics.ok or metrics.status == "dry_run":
                return metrics
        assert last_metrics is not None
        return last_metrics

    def _try_allocate(
        self,
        spec: ExperimentSpec,
        est: ResourceEstimate,
        gpus: Sequence[GpuInfo],
        budgets: Dict[int, int],
        reserved: Dict[int, int],
    ) -> Optional[Tuple[List[GpuInfo], int, int]]:
        needed = max(0, int(spec.num_gpus))
        if needed <= 0:
            return ([], 0, 0)
        if len(gpus) < needed:
            return None
        mem = max(0, est.memory_per_gpu_mb)
        candidates = sorted(gpus, key=lambda g: budgets[g.index] - reserved[g.index], reverse=True)
        chosen: List[GpuInfo] = []
        for gpu in candidates:
            available = budgets[gpu.index] - reserved[gpu.index]
            if available >= mem:
                chosen.append(gpu)
                if len(chosen) == needed:
                    return (chosen, mem, 0)
        return None

    def _gpu_budgets(self, gpus: Sequence[GpuInfo]) -> Dict[int, int]:
        budgets: Dict[int, int] = {}
        for gpu in gpus:
            target_total = int(gpu.total_mb * self.config.max_gpu_memory_fraction) - self.config.reserve_gpu_memory_mb
            external_used = max(0, gpu.used_mb)
            budgets[gpu.index] = max(0, target_total - external_used)
        return budgets

    def _select_probe_gpus(self, gpus: Sequence[GpuInfo], num_gpus: int) -> List[GpuInfo]:
        needed = max(0, int(num_gpus))
        if needed <= 0 or not gpus:
            return []
        return sorted(gpus, key=lambda g: g.free_mb, reverse=True)[:needed]

    def _append_run_state(self, spec_id: str, metrics: RunMetrics) -> None:
        self.state.setdefault("runs", {})
        self.state["runs"].setdefault(spec_id, []).append(metrics.to_dict())
        self._save_state()

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"created_at": time.time()}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"created_at": time.time(), "state_load_error": True}

    def _save_state(self) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event, payload)


def safe_duration(value: float) -> float:
    if math.isinf(value) or math.isnan(value):
        return -1.0
    return value
