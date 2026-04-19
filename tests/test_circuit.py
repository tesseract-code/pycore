import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from pycore.circuit import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)


# -------------------------------------------------------------------
# Fixtures and helpers
# -------------------------------------------------------------------
@pytest.fixture
def breaker():
    """Default circuit breaker with low thresholds for testing."""
    return CircuitBreaker(
        failure_threshold=3,
        reset_timeout=0.2,          # short timeout for tests
        half_open_max_calls=2,
        rolling_window_size=10,
    )


class FakeService:
    """Simulates an external service with configurable success/failure."""

    def __init__(self, fail_count: int = 0):
        self.call_count = 0
        self.fail_count = fail_count
        self.lock = threading.Lock()

    def call(self) -> str:
        with self.lock:
            self.call_count += 1
            if self.call_count <= self.fail_count:
                raise RuntimeError("Service failure")
            return "success"


# -------------------------------------------------------------------
# Basic state transitions
# -------------------------------------------------------------------
def test_initial_state_closed(breaker):
    assert breaker.get_state().is_closed
    assert breaker.get_state().consecutive_failures == 0


def test_failures_open_circuit(breaker):
    # Fail 3 times
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with breaker.protected():
                raise RuntimeError("fail")

    snap = breaker.get_state()
    assert snap.is_open
    assert snap.consecutive_failures == 3

    # Further calls rejected
    with pytest.raises(CircuitBreakerOpenError):
        with breaker.protected():
            pass


def test_timeout_transitions_to_half_open(breaker):
    # Force open
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with breaker.protected():
                raise RuntimeError("fail")

    # Wait for reset timeout
    time.sleep(breaker.reset_timeout + 0.05)

    snap = breaker.get_state()
    assert snap.is_half_open
    assert snap.half_open_calls == 0


def test_success_in_half_open_closes_circuit(breaker):
    # Open then wait
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with breaker.protected():
                raise RuntimeError("fail")
    time.sleep(breaker.reset_timeout + 0.05)

    # Successful probe
    with breaker.protected():
        pass  # success

    snap = breaker.get_state()
    assert snap.is_closed
    assert snap.consecutive_failures == 0
    assert snap.half_open_calls == 0


def test_failure_in_half_open_reopens_circuit(breaker):
    # Open then wait
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with breaker.protected():
                raise RuntimeError("fail")
    time.sleep(breaker.reset_timeout + 0.05)

    # Failed probe
    with pytest.raises(RuntimeError):
        with breaker.protected():
            raise RuntimeError("fail again")

    snap = breaker.get_state()
    assert snap.is_open
    # Failures should be >= threshold (it will be 4 because we added one more)
    assert snap.consecutive_failures >= 3
    # half_open_calls must be reset
    assert snap.half_open_calls == 0


def test_concurrent_probes_in_half_open(breaker):
    breaker.half_open_max_calls = 2

    # Open circuit
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with breaker.protected():
                raise RuntimeError("fail")
    time.sleep(breaker.reset_timeout + 0.05)

    # Acquire two probe slots
    assert breaker.can_execute()   # slot 1
    assert breaker.can_execute()   # slot 2
    # Third attempt fails
    assert not breaker.can_execute()

    # Record one failure -> should reset half_open_calls
    breaker.record_failure()
    snap = breaker.get_state()
    assert snap.is_open
    assert snap.half_open_calls == 0


def test_race_condition_success_after_failure_in_half_open(breaker):
    """
    Simulate: two concurrent probes, first fails (reopens circuit),
    second succeeds but should NOT close circuit.
    """
    breaker.half_open_max_calls = 2

    # Open then wait
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with breaker.protected():
                raise RuntimeError("fail")
    time.sleep(breaker.reset_timeout + 0.05)

    # Acquire two probe slots via can_execute()
    assert breaker.can_execute()
    assert breaker.can_execute()

    # First probe fails -> circuit opens
    breaker.record_failure()
    snap_open = breaker.get_state()
    assert snap_open.is_open

    # Second probe (delayed) succeeds
    breaker.record_success()

    # Circuit should remain OPEN, not be closed by the late success
    snap_final = breaker.get_state()
    assert snap_final.is_open
    # failures remain high
    assert snap_final.consecutive_failures >= 3


# -------------------------------------------------------------------
# Decay behavior
# -------------------------------------------------------------------
def test_time_based_decay():
    cb = CircuitBreaker(
        failure_threshold=5,
        use_time_based_decay=True,
        decay_factor=0.5,
    )
    # Cause 4 failures (below threshold)
    for _ in range(4):
        with pytest.raises(RuntimeError):
            with cb.protected():
                raise RuntimeError("fail")
    assert cb.get_state().consecutive_failures == 4

    # Success reduces failures by half
    with cb.protected():
        pass
    assert cb.get_state().consecutive_failures == 2   # 4 * 0.5

    # Another success halves again
    with cb.protected():
        pass
    assert cb.get_state().consecutive_failures == 1

    # Another success floors at 0
    with cb.protected():
        pass
    assert cb.get_state().consecutive_failures == 0


# -------------------------------------------------------------------
# Manual reset
# -------------------------------------------------------------------
def test_reset_clears_state_and_window(breaker):
    # Generate some history
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with breaker.protected():
                raise RuntimeError("fail")
    assert breaker.get_state().is_open

    breaker.reset()
    snap = breaker.get_state()
    assert snap.is_closed
    assert snap.consecutive_failures == 0
    assert snap.total_calls == 0
    assert snap.rolling_window_size == 0


# -------------------------------------------------------------------
# Decorator interface
# -------------------------------------------------------------------
def test_decorator_usage():
    cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

    @cb
    def risky_func(should_fail: bool):
        if should_fail:
            raise ValueError("boom")
        return "ok"

    # Success
    assert risky_func(False) == "ok"

    # Fail twice to open
    with pytest.raises(ValueError):
        risky_func(True)
    with pytest.raises(ValueError):
        risky_func(True)

    # Now open
    with pytest.raises(CircuitBreakerOpenError):
        risky_func(False)


# -------------------------------------------------------------------
# Context manager usage with manual calls (discouraged)
# -------------------------------------------------------------------
def test_manual_usage_leak_if_forget_record(breaker):
    # This demonstrates why manual pattern is risky
    breaker.half_open_max_calls = 1

    # Open then wait
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with breaker.protected():
                raise RuntimeError("fail")
    time.sleep(breaker.reset_timeout + 0.05)

    # Acquire probe slot but never record outcome
    assert breaker.can_execute()
    # Simulate forgetting to call record_success/failure
    # The slot remains occupied
    snap = breaker.get_state()
    assert snap.half_open_calls == 1

    # Next attempt fails
    assert not breaker.can_execute()


# -------------------------------------------------------------------
# Metrics and snapshots
# -------------------------------------------------------------------
def test_snapshot_immutability(breaker):
    snap1 = breaker.get_state()
    assert snap1.is_closed

    # Modify breaker state
    with pytest.raises(RuntimeError):
        with breaker.protected():
            raise RuntimeError("fail")
    snap2 = breaker.get_state()
    assert snap1 != snap2
    assert snap2.consecutive_failures == 1


def test_health_metrics(breaker):
    # Success then failure
    with breaker.protected():
        pass
    with pytest.raises(RuntimeError):
        with breaker.protected():
            raise RuntimeError("fail")

    metrics = breaker.get_health_metrics()
    assert metrics.circuit_state == CircuitState.CLOSED
    assert metrics.is_healthy is True
    assert metrics.total_operations == 2
    assert metrics.error_rate == 0.5
    assert metrics.time_since_last_failure is not None
    assert metrics.time_since_last_success is not None


# -------------------------------------------------------------------
# Concurrent stress test
# -------------------------------------------------------------------
def test_concurrent_operations(breaker):
    """Ensure no deadlocks or state corruption under concurrency."""
    service = FakeService(fail_count=3)

    def worker():
        try:
            with breaker.protected():
                return service.call()
        except (RuntimeError, CircuitBreakerOpenError):
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(20)]
        results = [f.result() for f in as_completed(futures)]

    # After failures, circuit should open and some calls rejected
    snap = breaker.get_state()
    assert snap.is_open
    assert snap.consecutive_failures >= 3
    # No deadlocks occurred
    assert True


def test_many_threads_in_half_open(breaker):
    breaker.half_open_max_calls = 3
    # Open circuit
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with breaker.protected():
                raise RuntimeError("fail")
    time.sleep(breaker.reset_timeout + 0.05)

    # Concurrent probe attempts
    successes = 0
    failures = 0
    lock = threading.Lock()

    def probe():
        nonlocal successes, failures
        try:
            with breaker.protected():
                # Simulate some work
                time.sleep(0.01)
            with lock:
                successes += 1
        except CircuitBreakerOpenError:
            with lock:
                failures += 1
        except Exception:
            with lock:
                failures += 1

    threads = [threading.Thread(target=probe) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # At most half_open_max_calls should succeed
    assert successes == breaker.half_open_max_calls
    assert failures == 10 - breaker.half_open_max_calls


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------
def test_failure_threshold_one():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)
    with pytest.raises(RuntimeError):
        with cb.protected():
            raise RuntimeError("fail")
    assert cb.get_state().is_open


def test_half_open_max_calls_one():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1, half_open_max_calls=1)
    with pytest.raises(RuntimeError):
        with cb.protected():
            raise RuntimeError("fail")
    time.sleep(0.2)
    assert cb.can_execute() is True
    assert cb.can_execute() is False


def test_never_used_metrics():
    cb = CircuitBreaker()
    metrics = cb.get_health_metrics()
    assert metrics.time_since_last_failure is None
    assert metrics.time_since_last_success is None


def test_invalid_construction():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(reset_timeout=0)
    with pytest.raises(ValueError):
        CircuitBreaker(half_open_max_calls=0)
    with pytest.raises(ValueError):
        CircuitBreaker(decay_factor=1.5)