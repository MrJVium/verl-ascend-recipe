"""Core data models for KernelGym refactor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Artifact:
    name: str
    uri: Optional[str] = None
    data: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    unit: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    task_id: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    metrics: list[Metric] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class TaskSpec:
    """Minimal task spec for scheduler submission."""

    kind: str
    payload: dict[str, Any]
    resources: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskGroup:
    """Container for multiple tasks with optional dependencies."""

    tasks: list[TaskSpec]
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
