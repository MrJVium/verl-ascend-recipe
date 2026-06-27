"""Schema for kernel simple workflow tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class KernelSimpleTask:
    task_id: str
    kernel_code: str
    toolkit: str = "kernel_simple"
    backend_adapter: str = "kernelbench"
    backend: str = "triton"
    entry_point: str = "ModelNew"
    num_perf_trials: int = 100
    num_warmup: int = 3
    timeout: int = 300
    device: str = "npu:0"
    priority: str = "normal"
    run_correctness: Optional[bool] = None
    run_performance: Optional[bool] = None
    enable_profiling: Optional[bool] = None
    cases_code: Optional[str] = None
    cases: Optional[list[Any]] = None
    resources: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KernelSimpleTask:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
