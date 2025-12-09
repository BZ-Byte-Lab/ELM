"""
Performance profiling for synthesis pipeline.
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from collections import defaultdict

from .utils import get_logger

logger = get_logger("profiler")


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    samples: List[float] = field(default_factory=list)

    def add(self, duration: float) -> None:
        """Add a timing sample."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        # Only keep last 1000 samples to prevent memory growth
        if len(self.samples) < 1000:
            self.samples.append(duration)

    @property
    def avg_time(self) -> float:
        """Average time."""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def median_time(self) -> float:
        """Median time."""
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def p95_time(self) -> float:
        """95th percentile time."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "median_time": self.median_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0.0,
            "max_time": self.max_time,
            "p95_time": self.p95_time,
        }


class PerformanceProfiler:
    """Tracks performance metrics for the synthesis pipeline."""

    def __init__(self, enabled: bool = True):
        """Initialize profiler.

        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.timings: Dict[str, TimingStats] = defaultdict(TimingStats)
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def measure(self, operation: str):
        """Context manager to measure operation time.

        Args:
            operation: Name of the operation being measured
        """
        if not self.enabled:
            yield
            return

        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            async with self.lock:
                self.timings[operation].add(duration)

    async def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all timing statistics.

        Returns:
            Dictionary mapping operation name to statistics
        """
        async with self.lock:
            return {
                operation: stats.to_dict()
                for operation, stats in self.timings.items()
            }

    async def print_summary(self) -> None:
        """Print performance summary."""
        stats = await self.get_stats()

        if not stats:
            logger.info("No performance data collected")
            return

        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 80)

        # Sort by total time (most expensive first)
        sorted_ops = sorted(
            stats.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )

        for operation, timing in sorted_ops:
            logger.info(f"\n{operation}:")
            logger.info(f"  Count:      {timing['count']}")
            logger.info(f"  Total:      {timing['total_time']:.2f}s")
            logger.info(f"  Avg:        {timing['avg_time']*1000:.2f}ms")
            logger.info(f"  Median:     {timing['median_time']*1000:.2f}ms")
            logger.info(f"  P95:        {timing['p95_time']*1000:.2f}ms")
            logger.info(f"  Min:        {timing['min_time']*1000:.2f}ms")
            logger.info(f"  Max:        {timing['max_time']*1000:.2f}ms")

        logger.info("\n" + "=" * 80)

    def reset(self) -> None:
        """Reset all timing statistics."""
        self.timings.clear()
