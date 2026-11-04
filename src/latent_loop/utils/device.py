"""Hardware inspection. PyTorch is optional until Phase 2, so every probe degrades gracefully."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def _read_meminfo() -> dict[str, float]:
    info: dict[str, float] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                info[key] = int(parts[0]) / 1024**2  # kB -> GiB
    return info


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def query_nvidia_smi() -> list[dict[str, Any]]:
    """Return one dict per GPU reported by ``nvidia-smi`` (empty if unavailable)."""
    if shutil.which("nvidia-smi") is None:
        return []
    fields = "name,memory.total,memory.used,driver_version,compute_cap"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, total, used, driver, cc = (p.strip() for p in line.split(","))
        gpus.append(
            {
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "driver_version": driver,
                "compute_capability": cc,
            }
        )
    return gpus


def torch_info() -> dict[str, Any]:
    """Report PyTorch/CUDA status without requiring PyTorch to be installed."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return {"installed": False}
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_environment(disk_path: str | Path = ".") -> dict[str, Any]:
    mem = _read_meminfo()
    disk = shutil.disk_usage(disk_path)
    release = platform.release()
    return {
        "platform": platform.platform(),
        "kernel": release,
        "wsl": "microsoft" in release.lower(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_total_gib": round(mem.get("MemTotal", float("nan")), 2),
        "ram_available_gib": round(mem.get("MemAvailable", float("nan")), 2),
        "disk_path": str(Path(disk_path).resolve()),
        "disk_free_gib": round(disk.free / 1024**3, 1),
        "gpus": query_nvidia_smi(),
        "torch": torch_info(),
        "packages": package_versions(
            [
                "numpy",
                "scipy",
                "opencv-python-headless",
                "matplotlib",
                "Pillow",
                "imageio",
                "PyYAML",
                "pandas",
                "pytest",
                "ruff",
                "torch",
                "faiss-cpu",
            ]
        ),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }


def resolve_device(preference: str = "auto") -> str:
    """Map ``auto|cuda|cpu`` to an available device; CUDA requests fall back to CPU."""
    if preference not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device must be auto|cuda|cpu, got {preference!r}")
    try:
        import torch  # noqa: PLC0415

        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if preference == "cpu" or not cuda:
        if preference == "cuda":
            from latent_loop.utils.logging import get_logger  # noqa: PLC0415

            get_logger(__name__).warning("CUDA requested but unavailable; using CPU")
        return "cpu"
    return "cuda"


def resolve_precision(preference: str, device: str) -> str:
    """Map ``auto|fp32|fp16|bf16`` to a precision the device supports.

    ``auto`` selects BF16 on GPUs with native BF16 (Ampere and newer). Otherwise it
    uses FP16 only if a micro-benchmark shows FP16 matrix multiplication is faster than
    FP32 on this GPU, else FP32. GPUs without Tensor Cores can be much slower in FP16:
    the GTX 1650 Ti (TU117) measured 0.36 TFLOPS FP16 vs 1.62 TFLOPS FP32. CPU always
    uses FP32. Unsupported explicit requests fall back with a warning.
    """
    if preference not in ("auto", "fp32", "fp16", "bf16"):
        raise ValueError(f"precision must be auto|fp32|fp16|bf16, got {preference!r}")
    from latent_loop.utils.logging import get_logger  # noqa: PLC0415

    log = get_logger(__name__)
    if device == "cpu":
        if preference not in ("auto", "fp32"):
            log.warning(
                "Reduced precision is not used on CPU; using fp32", extra={"requested": preference}
            )
        return "fp32"
    import torch  # noqa: PLC0415

    try:
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older PyTorch without the keyword
        bf16 = torch.cuda.is_bf16_supported()
    if preference == "auto":
        if bf16:
            return "bf16"
        return "fp16" if fp16_faster_than_fp32() else "fp32"
    if preference == "bf16" and not bf16:
        log.warning("BF16 is not natively supported on this GPU; using fp16")
        return "fp16"
    return preference


_FP16_FASTER: dict[int, bool] = {}


def fp16_faster_than_fp32(size: int = 2048, repeats: int = 3) -> bool:
    """Time FP16 vs FP32 matrix multiplication on the current GPU (cached per process)."""
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    device = torch.cuda.current_device()
    if device not in _FP16_FASTER:
        timings = {}
        for dtype in (torch.float16, torch.float32):
            a = torch.randn(size, size, device="cuda", dtype=dtype)
            a @ a  # warm-up (kernel selection)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                a @ a
            torch.cuda.synchronize()
            timings[dtype] = time.perf_counter() - start
            del a
        _FP16_FASTER[device] = timings[torch.float16] < timings[torch.float32]
        from latent_loop.utils.logging import get_logger  # noqa: PLC0415

        get_logger(__name__).info(
            "Precision micro-benchmark",
            extra={
                "fp16_ms": round(1000 * timings[torch.float16] / repeats, 1),
                "fp32_ms": round(1000 * timings[torch.float32] / repeats, 1),
                "auto_choice": "fp16" if _FP16_FASTER[device] else "fp32",
            },
        )
    return _FP16_FASTER[device]


def torch_dtype(precision: str):
    import torch  # noqa: PLC0415

    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
