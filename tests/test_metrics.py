"""
Tests for src/core/metrics.py.

Covers:
- record() + snapshot() — histogram stats (count, mean, p50, p95, p99)
- increment() + snapshot() — counter values
- snapshot() on empty collector — returns empty dict
- start_periodic_logging() / stop() — daemon thread lifecycle
- thread safety — concurrent records don't corrupt state
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_collector():
    """Import and return a fresh MetricsCollector instance."""
    from src.core.metrics import MetricsCollector
    return MetricsCollector()


# ---------------------------------------------------------------------------
# Unit: snapshot on empty collector
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_snapshot_empty():
    """snapshot() returns empty dict when nothing has been recorded."""
    mc = _make_collector()
    result = mc.snapshot()
    assert result == {}


# ---------------------------------------------------------------------------
# Unit: record() + snapshot() — histogram
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_record_and_snapshot_basic_stats():
    """snapshot() returns correct count and mean after recording values."""
    mc = _make_collector()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        mc.record("latency_ms", v)

    snap = mc.snapshot()
    assert "latency_ms" in snap
    hist = snap["latency_ms"]

    assert hist["count"] == 5
    assert abs(hist["mean"] - 3.0) < 1e-9


@pytest.mark.unit
def test_record_and_snapshot_percentiles():
    """snapshot() computes p50, p95, p99 for a known distribution."""
    mc = _make_collector()
    # Record 100 values: 1..100
    for v in range(1, 101):
        mc.record("response_time", float(v))

    snap = mc.snapshot()
    hist = snap["response_time"]

    assert hist["count"] == 100
    # p50 of 1..100 is 50 or 51 depending on interpolation — allow ±2
    assert 49 <= hist["p50"] <= 52
    # p95 should be near 95
    assert 94 <= hist["p95"] <= 96
    # p99 should be near 99
    assert 98 <= hist["p99"] <= 100


@pytest.mark.unit
def test_record_single_value_percentiles():
    """snapshot() handles a single recorded value without crashing."""
    mc = _make_collector()
    mc.record("lone_metric", 42.0)

    snap = mc.snapshot()
    hist = snap["lone_metric"]

    assert hist["count"] == 1
    assert hist["mean"] == 42.0
    assert hist["p50"] == 42.0
    assert hist["p95"] == 42.0
    assert hist["p99"] == 42.0


@pytest.mark.unit
def test_record_multiple_metrics_are_independent():
    """Different metric names maintain separate histogram buckets."""
    mc = _make_collector()
    mc.record("metric_a", 10.0)
    mc.record("metric_b", 99.0)

    snap = mc.snapshot()
    assert snap["metric_a"]["count"] == 1
    assert snap["metric_a"]["mean"] == 10.0
    assert snap["metric_b"]["count"] == 1
    assert snap["metric_b"]["mean"] == 99.0


# ---------------------------------------------------------------------------
# Unit: increment() + snapshot() — counters
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_increment_and_snapshot():
    """increment() increases named counter; snapshot() includes it."""
    mc = _make_collector()
    mc.increment("requests_total")
    mc.increment("requests_total")
    mc.increment("requests_total")

    snap = mc.snapshot()
    assert "requests_total" in snap
    assert snap["requests_total"] == 3


@pytest.mark.unit
def test_increment_default_starts_at_zero():
    """First increment on a new counter goes from 0 to 1."""
    mc = _make_collector()
    mc.increment("new_counter")

    snap = mc.snapshot()
    assert snap["new_counter"] == 1


@pytest.mark.unit
def test_counters_and_histograms_coexist():
    """Counters and histograms can have different names without collision."""
    mc = _make_collector()
    mc.record("latency", 5.0)
    mc.increment("errors")

    snap = mc.snapshot()
    assert isinstance(snap["latency"], dict)   # histogram dict
    assert isinstance(snap["errors"], int)     # plain integer counter


# ---------------------------------------------------------------------------
# Unit: periodic logging thread lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_periodic_logging_starts_and_stops_cleanly():
    """start_periodic_logging() spawns a daemon thread; stop() terminates it."""
    mc = _make_collector()

    # Patch sleep so the thread doesn't actually block
    with patch("src.core.metrics.time.sleep", return_value=None):
        mc.start_periodic_logging(interval_seconds=60)

        # Thread must be alive right after start
        assert mc._thread is not None
        assert mc._thread.is_alive()
        assert mc._thread.daemon is True

        mc.stop()

        # Give the thread up to 2 seconds to exit
        mc._thread.join(timeout=2.0)
        assert not mc._thread.is_alive()


@pytest.mark.unit
def test_stop_before_start_is_safe():
    """Calling stop() before start_periodic_logging() does not raise."""
    mc = _make_collector()
    mc.stop()  # must not raise


@pytest.mark.unit
def test_periodic_logging_calls_logger(caplog):
    """Periodic thread logs snapshot JSON at INFO level."""
    import logging
    mc = _make_collector()
    mc.record("req", 1.0)
    mc.increment("hits")

    log_calls = []

    # Replace the internal logger to capture calls
    mock_logger = MagicMock()
    mock_logger.info.side_effect = lambda msg, *a, **kw: log_calls.append(msg)

    # Use a very short interval and let one iteration run, then stop
    call_event = threading.Event()

    def patched_sleep(seconds):
        call_event.set()
        raise StopIteration  # break the loop after first sleep

    with patch("src.core.metrics.time.sleep", side_effect=patched_sleep):
        with patch("src.core.metrics.logger", mock_logger):
            mc.start_periodic_logging(interval_seconds=1)
            call_event.wait(timeout=3.0)
            mc.stop()

    assert mock_logger.info.called


# ---------------------------------------------------------------------------
# Integration: thread safety
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_thread_safety_concurrent_records():
    """Concurrent record() calls from many threads do not corrupt state."""
    mc = _make_collector()
    n_threads = 20
    records_per_thread = 50

    def worker():
        for _ in range(records_per_thread):
            mc.record("concurrent_metric", 1.0)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = mc.snapshot()
    assert snap["concurrent_metric"]["count"] == n_threads * records_per_thread


@pytest.mark.integration
def test_thread_safety_concurrent_increments():
    """Concurrent increment() calls do not lose counts."""
    mc = _make_collector()
    n_threads = 30
    increments_per_thread = 100

    def worker():
        for _ in range(increments_per_thread):
            mc.increment("shared_counter")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = mc.snapshot()
    assert snap["shared_counter"] == n_threads * increments_per_thread
