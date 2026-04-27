from __future__ import annotations

import hashlib
import itertools
import json
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

Command = Union[str, Sequence[str]]


class MissingTemplateKey(KeyError):
    """Raised when a command/name template references a missing parameter."""


class _StrictFormatDict(dict):
    def __missing__(self, key: str) -> str:
        raise MissingTemplateKey(key)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def stable_id(*parts: Any, prefix: str = "exp") -> str:
    raw = "\n".join(_stable_json(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def format_template(template: str, params: Mapping[str, Any]) -> str:
    return template.format_map(_StrictFormatDict(params))


@dataclass(frozen=True)
class ExperimentSpec:
    """One executable experiment.

    command may be a shell-like string template or an argv list template.
    Example: "python train.py --model {model} --lr {lr} --epochs {epochs}"
    """

    name: str
    command: Command
    params: Dict[str, Any] = field(default_factory=dict)
    probe_params: Dict[str, Any] = field(default_factory=lambda: {"epochs": 1})
    run_params: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    shell: bool = False
    num_gpus: int = 1
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_s: Optional[float] = None

    @property
    def id(self) -> str:
        return stable_id(
            self.name,
            self.command,
            self.params,
            self.probe_params,
            self.run_params,
            self.cwd,
            self.env,
            self.shell,
            self.num_gpus,
            prefix="exp",
        )

    def merged_params(self, mode: str) -> Dict[str, Any]:
        merged = dict(self.params)
        if mode == "probe":
            merged.update(self.probe_params)
        elif mode == "run":
            merged.update(self.run_params)
        else:
            raise ValueError("mode must be 'probe' or 'run'")
        return merged

    def rendered_name(self, mode: str = "run") -> str:
        params = self.merged_params(mode)
        try:
            return format_template(self.name, params)
        except MissingTemplateKey:
            # Names do not have to be templates. If formatting fails, return raw name.
            return self.name

    def rendered_command(self, mode: str = "run") -> Union[str, List[str]]:
        params = self.merged_params(mode)
        if isinstance(self.command, str):
            rendered = format_template(self.command, params)
            return rendered if self.shell else shlex.split(rendered)
        return [format_template(str(part), params) for part in self.command]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "command": self.command,
            "params": self.params,
            "probe_params": self.probe_params,
            "run_params": self.run_params,
            "env": self.env,
            "cwd": self.cwd,
            "shell": self.shell,
            "num_gpus": self.num_gpus,
            "priority": self.priority,
            "tags": self.tags,
            "metadata": self.metadata,
            "timeout_s": self.timeout_s,
        }


def expand_grid(
    *,
    name: str,
    command: Command,
    grid: Mapping[str, Iterable[Any]],
    params: Optional[Mapping[str, Any]] = None,
    probe_params: Optional[Mapping[str, Any]] = None,
    run_params: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[str] = None,
    shell: bool = False,
    num_gpus: int = 1,
    priority: int = 0,
    tags: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    timeout_s: Optional[float] = None,
) -> List[ExperimentSpec]:
    """Expand a parameter grid into ExperimentSpec objects."""
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    base_params = dict(params or {})
    specs: List[ExperimentSpec] = []
    for combo in itertools.product(*values):
        combo_params = dict(zip(keys, combo))
        merged_params = dict(base_params)
        merged_params.update(combo_params)
        specs.append(
            ExperimentSpec(
                name=name,
                command=command,
                params=merged_params,
                probe_params=dict(probe_params or {"epochs": 1}),
                run_params=dict(run_params or {}),
                env=dict(env or {}),
                cwd=cwd,
                shell=shell,
                num_gpus=num_gpus,
                priority=priority,
                tags=list(tags or []),
                metadata=dict(metadata or {}),
                timeout_s=timeout_s,
            )
        )
    return specs
