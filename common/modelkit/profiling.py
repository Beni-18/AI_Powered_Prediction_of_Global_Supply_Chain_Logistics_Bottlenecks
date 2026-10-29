"""ROLE: ML ENGINEER  |  COMMON MODELKIT
Inference latency and memory benchmarking tools for Day 0 foundations through Day 4 SLM.

Owner: Role 1 - ML Engineer
"""

import time
from typing import Any, Callable
import numpy as np


def latency_probe(fn: Callable[[], Any], n: int = 100, warmup: int = 5) -> dict[str, float]:
    """Benchmark the execution latency and estimate memory usage of a callable function.

    Performs warmup iterations to eliminate JIT/caching bias, then runs n repetitions.
    Returns dictionary with p50_ms, p95_ms, mean_ms, and peak_rss_mb (if psutil available).
    """
    # Discard warm-up runs to prime caches, thread pools, and memory allocators.
    for _ in range(warmup):
        fn()

    durations_ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)

    results = {
        "p50_ms": float(np.percentile(durations_ms, 50)),
        "p95_ms": float(np.percentile(durations_ms, 95)),
        "mean_ms": float(np.mean(durations_ms)),
        "min_ms": float(np.min(durations_ms)),
        "max_ms": float(np.max(durations_ms)),
        "iterations": n,
    }

    # If psutil is present, record resident set memory usage.
    try:
        import psutil
        process = psutil.Process()
        results["peak_rss_mb"] = float(process.memory_info().rss / (1024 * 1024))
    except ImportError:
        results["peak_rss_mb"] = 0.0

    return results
