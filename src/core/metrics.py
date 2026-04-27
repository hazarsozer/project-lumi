"""
Lightweight stdlib histogram and counter metrics for Project Lumi.

All modules should import MetricsCollector directly:
    from src.core.metrics import MetricsCollector

Usage::

    mc = MetricsCollector()
    mc.record("latency_ms", 42.3)
    mc.increment("requests_total")
    mc.start_periodic_logging(interval_seconds=60)
    # … at shutdown …
    mc.stop()

Snapshot JSON format::

    {
        "latency_ms": {
            "count": 1,
            "mean": 42.3,
            "p50": 42.3,
            "p95": 42.3,
            "p99": 42.3
        },
        "requests_total": 3
    }

Histogram entries are dicts; counter entries are plain integers.
Both live in the same snapshot dict, keyed by name.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

# Type alias for snapshot values: either a histogram dict or an integer counter.
_SnapshotValue = dict | int


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of a pre-sorted list using nearest-rank.

    Args:
        sorted_values: A non-empty list sorted in ascending order.
        pct: Percentile in [0, 100].

    Returns:
        The nearest-rank percentile value.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    # Nearest-rank method: index = ceil(pct/100 * n) - 1, clamped to [0, n-1]
    idx = int((pct / 100.0) * n + 0.5) - 1
    idx = max(0, min(idx, n - 1))
    return sorted_values[idx]


class MetricsCollector:
    """Thread-safe histogram and counter collector with periodic JSON logging.

    Histograms are tracked as raw value lists (suitable for low-to-moderate
    cardinality; swap for a streaming estimator if needed at very high volume).
    Counters are plain integer accumulators.

    All public methods are protected by a single ``threading.Lock``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # metric_name -> list of float values
        self._histograms: dict[str, list[float]] = defaultdict(list)
        # counter_name -> int
        self._counters: dict[str, int] = defaultdict(int)

        # Periodic logging state
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, metric_name: str, value: float) -> None:
        """Append *value* to the histogram bucket named *metric_name*.

        Args:
            metric_name: Logical name for the measurement (e.g. "latency_ms").
            value: Numeric observation to record.
        """
        with self._lock:
            self._histograms[metric_name].append(value)

    def increment(self, counter_name: str) -> None:
        """Increment the named counter by 1.

        Args:
            counter_name: Logical name for the counter (e.g. "requests_total").
        """
        with self._lock:
            self._counters[counter_name] += 1

    def snapshot(self) -> dict[str, _SnapshotValue]:
        """Return current stats for all recorded metrics.

        Histogram entries contain::

            {
                "count": int,
                "mean": float,
                "p50": float,
                "p95": float,
                "p99": float,
            }

        Counter entries are plain ``int`` values.

        Returns:
            A dict mapping metric/counter name to its current stats.
            Returns an empty dict when nothing has been recorded.
        """
        with self._lock:
            result: dict[str, _SnapshotValue] = {}

            for name, values in self._histograms.items():
                if not values:
                    continue
                sorted_vals = sorted(values)
                count = len(sorted_vals)
                mean = sum(sorted_vals) / count
                result[name] = {
                    "count": count,
                    "mean": mean,
                    "p50": _percentile(sorted_vals, 50),
                    "p95": _percentile(sorted_vals, 95),
                    "p99": _percentile(sorted_vals, 99),
                }

            for name, count in self._counters.items():
                result[name] = count

            return result

    def start_periodic_logging(self, interval_seconds: int = 60) -> None:
        """Start a daemon thread that logs a snapshot every *interval_seconds*.

        The thread is idempotent: calling this method a second time while a
        thread is already running is a no-op.

        Args:
            interval_seconds: How often (in seconds) to emit a snapshot log.
                              Defaults to 60.
        """
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._logging_loop,
            args=(interval_seconds,),
            name="MetricsCollector-periodic",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the periodic logging thread to stop and wait for it to exit.

        Safe to call even if ``start_periodic_logging`` was never called.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _logging_loop(self, interval_seconds: int) -> None:
        """Target function for the periodic logging daemon thread."""
        while not self._stop_event.is_set():
            interrupted = False
            try:
                time.sleep(interval_seconds)
            except Exception:
                # Injected in tests (e.g. StopIteration) to break the sleep.
                # Emit one final snapshot before exiting so callers can assert
                # that at least one log was produced.
                interrupted = True

            if self._stop_event.is_set():
                break

            snap = self.snapshot()
            logger.info(json.dumps(snap, ensure_ascii=False))

            if interrupted:
                break
