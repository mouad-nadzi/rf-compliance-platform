"""
core/utils/vram.py — GPU VRAM guardrail utilities.

Prevents uncatchable CUDA OOM kernel crashes on memory-constrained GPUs
(e.g. the 16 GB Tesla T4 used in Colab). Provides:

  - free_vram_mb(): current free VRAM in MiB.
  - used_vram_mb(): current used VRAM in MiB.
  - flush_gpu_cache(): releases cached PyTorch/transformer memory back to the driver.
  - ensure_headroom(min_free_mb, op): raises a graceful MemoryError when free VRAM
    falls below a safety threshold BEFORE a heavy stage runs.

Usage pattern (between heavy pipeline stages):

    flush_gpu_cache()
    ensure_headroom(config.MIN_FREE_VRAM_MB, "OCR generation")
    ...run heavy stage...
"""

import gc
import logging
import subprocess

logger = logging.getLogger(__name__)

#: nvidia-smi query string returning bare MiB numbers (one per line).
_QUERY = "--query-gpu=memory.used,memory.free,memory.total"


def _smi_numbers() -> list[int]:
    """Returns [used_mib, free_mib, total_mib] from nvidia-smi (empty list on failure)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", _QUERY, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [int(x.strip()) for x in out.stdout.strip().split(",")]
    except Exception as e:  # noqa: BLE001 - never crash the caller on a probe failure
        logger.warning(f"  VRAM probe failed: {e}")
        return []


def used_vram_mb() -> int:
    nums = _smi_numbers()
    return nums[0] if len(nums) == 3 else 0


def free_vram_mb() -> int:
    nums = _smi_numbers()
    return nums[1] if len(nums) == 3 else -1


def total_vram_mb() -> int:
    nums = _smi_numbers()
    return nums[2] if len(nums) == 3 else 0


def flush_gpu_cache() -> None:
    """Return cached PyTorch/CUDA memory to the driver before a heavy stage."""
    try:
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f" GPU cache flushed (free VRAM now ~{free_vram_mb()} MiB).")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  GPU cache flush skipped: {e}")


def ensure_headroom(min_free_mb: int, op: str = "pipeline stage") -> None:
    """
    Raises MemoryError if free VRAM is below `min_free_mb` before running `op`.

    This converts a would-be uncatchable kernel OOM into a catchable, graceful
    error so the API can respond 4xx/5xx instead of crashing the whole process.
    """
    free = free_vram_mb()
    if free == -1:
        logger.warning("  Cannot probe VRAM; proceeding without headroom guard.")
        return
    if free < min_free_mb:
        raise MemoryError(
            f"Insufficient free VRAM ({free} MiB) for {op}; "
            f"need >= {min_free_mb} MiB headroom. Close other heavy processes."
        )
    logger.info(f"  VRAM headroom OK: {free} MiB free (threshold {min_free_mb} MiB) for {op}.")


def monitor_peak(interval_seconds: float = 0.1) -> "PeakMonitor":
    """Returns a context-manager-style sampler that tracks peak used VRAM.

    Usage:
        with monitor_peak() as peak:
            ...heavy work...
        print(peak.value())
    """
    return PeakMonitor(interval_seconds)


class PeakMonitor:
    """Background thread sampling peak GPU memory usage (for diagnostics only)."""

    def __init__(self, interval_seconds: float = 0.1):
        import threading

        self._interval = interval_seconds
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = used_vram_mb()
            if used > self._peak:
                self._peak = used
            self._stop.wait(self._interval)

    def __enter__(self) -> "PeakMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self._stop.set()
        self._thread.join(timeout=5)

    def value(self) -> int:
        return self._peak
