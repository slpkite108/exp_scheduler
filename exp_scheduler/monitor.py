from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set


def _run_nvidia_smi(args: Sequence[str]) -> Optional[str]:
    try:
        cp = subprocess.run(
            ["nvidia-smi", *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if cp.returncode != 0:
        return None
    return cp.stdout.strip()


@dataclass
class GpuInfo:
    index: int
    uuid: str
    name: str
    total_mb: int
    used_mb: int
    free_mb: int

    @property
    def display(self) -> str:
        return f"GPU {self.index} {self.name} total={self.total_mb}MB free={self.free_mb}MB"


def detect_gpus() -> List[GpuInfo]:
    """Return NVIDIA GPU inventory. Empty list when nvidia-smi is unavailable."""
    out = _run_nvidia_smi(
        [
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []

    gpus: List[GpuInfo] = []
    reader = csv.reader(out.splitlines())
    for row in reader:
        if len(row) < 6:
            continue
        idx, uuid, name, total, used, free = [x.strip() for x in row[:6]]
        try:
            gpus.append(
                GpuInfo(
                    index=int(idx),
                    uuid=uuid,
                    name=name,
                    total_mb=int(float(total)),
                    used_mb=int(float(used)),
                    free_mb=int(float(free)),
                )
            )
        except ValueError:
            continue
    return gpus


def _read_children_from_proc(pid: int) -> List[int]:
    path = f"/proc/{pid}/task/{pid}/children"
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except OSError:
        return []
    if not raw:
        return []
    children: List[int] = []
    for token in raw.split():
        try:
            children.append(int(token))
        except ValueError:
            pass
    return children


def process_tree_pids(root_pid: int) -> Set[int]:
    """Return root pid and descendants on Linux. Falls back to root only."""
    seen: Set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(_read_children_from_proc(pid))
    return seen


def rss_mb_for_pids(pids: Iterable[int]) -> float:
    total_kb = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            total_kb += int(parts[1])
                        break
        except OSError:
            continue
    return total_kb / 1024.0


@dataclass
class GpuProcessSample:
    pid: int
    gpu_uuid: str
    used_memory_mb: int


def gpu_process_samples() -> List[GpuProcessSample]:
    """Per-process GPU memory samples from nvidia-smi."""
    out = _run_nvidia_smi(
        [
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if out is None:
        return []
    if not out:
        return []
    samples: List[GpuProcessSample] = []
    reader = csv.reader(out.splitlines())
    for row in reader:
        if len(row) < 3:
            continue
        pid_s, uuid, mem_s = [x.strip() for x in row[:3]]
        try:
            samples.append(GpuProcessSample(pid=int(pid_s), gpu_uuid=uuid, used_memory_mb=int(float(mem_s))))
        except ValueError:
            continue
    return samples


def gpu_memory_for_pids(pids: Iterable[int], gpu_uuids: Optional[Set[str]] = None) -> int:
    pid_set = set(pids)
    total = 0
    for sample in gpu_process_samples():
        if sample.pid in pid_set and (gpu_uuids is None or sample.gpu_uuid in gpu_uuids):
            total += sample.used_memory_mb
    return total


def visible_devices_env(gpu_indices: Sequence[int]) -> Dict[str, str]:
    if not gpu_indices:
        return {}
    return {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpu_indices)}


def current_environment_summary() -> Dict[str, object]:
    return {
        "pid": os.getpid(),
        "gpus": [gpu.__dict__ for gpu in detect_gpus()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
