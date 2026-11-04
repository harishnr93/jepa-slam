"""Lightweight runtime and memory profiling."""

from __future__ import annotations

import resource
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np


def _cuda():
    try:
        import torch  # noqa: PLC0415

        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Profiler:
    """Wall-clock time per named stage, plus peak GPU (PyTorch allocator) and CPU memory."""

    def __init__(self) -> None:
        self.durations: dict[str, list[float]] = defaultdict(list)
        self.items: dict[str, int] = defaultdict(int)
        torch = _cuda()
        if torch is not None:
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def stage(self, name: str, items: int = 1, sync_cuda: bool = True) -> Iterator[None]:
        torch = _cuda() if sync_cuda else None
        if torch is not None:
            torch.cuda.synchronize()  # time GPU work, not just the kernel launch
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch is not None:
                torch.cuda.synchronize()
            self.durations[name].append(time.perf_counter() - start)
            self.items[name] += items

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, values in self.durations.items():
            arr = np.asarray(values)
            out[name] = {
                "calls": int(arr.size),
                "items": int(self.items[name]),
                "total_s": round(float(arr.sum()), 3),
                "ms_per_item": round(1000 * float(arr.sum()) / max(self.items[name], 1), 2),
                "median_call_ms": round(1000 * float(np.median(arr)), 2),
                "p95_call_ms": round(1000 * float(np.percentile(arr, 95)), 2),
            }
        torch = _cuda()
        if torch is not None:
            out["peak_gpu_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            out["peak_gpu_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        out["peak_cpu_rss_mb"] = round(peak_rss_mb(), 1)
        return out
