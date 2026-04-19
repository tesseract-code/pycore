"""
circuit_breaker.py — Thread-safe circuit breaker for fault-tolerant systems.

State machine
-------------
CLOSED    → Normal operation; all calls pass through.
OPEN      → Fault detected; calls are rejected immediately.
HALF_OPEN → Recovery probe; a limited number of test calls are permitted.

Transitions
-----------
CLOSED    → OPEN      when consecutive_failures ≥ failure_threshold
OPEN      → HALF_OPEN after reset_timeout seconds have elapsed
HALF_OPEN → CLOSED    on a successful probe call (only if still HALF_OPEN)
HALF_OPEN → OPEN      on a failed probe call

Usage — decorator (preferred)
------------------------------
    cb = CircuitBreaker(failure_threshold=3, reset_timeout=30.0)

    @cb
    def call_external_service() -> str: ...

Usage — context manager (recommended for manual control)
-----------------------
    with cb.protected():
        call_external_service()

Usage — manual (discouraged due to risk of leaking probe count)
--------------
    if cb.can_execute():
        try:
            result = call_external_service()
            cb.record_success()
        except Exception:
            cb.record_failure()
            raise
    else:
        raise CircuitBreakerOpenError(cb.get_state())
"""

from __future__ import annotations

import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from logging import getLogger
from time import monotonic
from typing import Any, Callable, Deque, Final, Generator, TypeVar

__all__ = (
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
    "CircuitBreakerSnapshot",
    "HealthMetrics",
)

logger = getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])
_NEVER: Final[float] = 0.0


class CircuitState(str, Enum):
    """Canonical circuit breaker states, ordered by increasing severity."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


@dataclass(frozen=True, slots=True)
class CircuitBreakerSnapshot:
    """Immutable snapshot of circuit breaker internals."""

    state: CircuitState
    consecutive_failures: int
    total_calls: int
    total_failures: int
    success_count: int
    failure_rate: float
    recent_success_rate: float
    recent_error_rate: float
    rolling_window_size: int
    last_failure_time: float
    last_success_time: float
    half_open_calls: int
    half_open_max_calls: int

    @property
    def is_open(self) -> bool:
        return self.state is CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        return self.state is CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        return self.state is CircuitState.HALF_OPEN

    def __str__(self) -> str:
        return (
            f"CircuitBreakerSnapshot("
            f"state={self.state!r}, "
            f"failures={self.consecutive_failures}/{self.total_failures} total, "
            f"error_rate={self.failure_rate:.1%})"
        )


@dataclass(frozen=True, slots=True)
class HealthMetrics:
    """Immutable health metrics snapshot."""

    circuit_state: CircuitState
    is_healthy: bool
    is_degraded: bool
    consecutive_failures: int
    total_operations: int
    error_rate: float
    recent_success_rate: float
    rolling_window_size: int
    time_since_last_failure: float | None
    time_since_last_success: float | None

    def __str__(self) -> str:
        return (
            f"HealthMetrics("
            f"state={self.circuit_state!r}, "
            f"healthy={self.is_healthy}, "
            f"error_rate={self.error_rate:.1%})"
        )


@dataclass(slots=True)
class _CircuitBreakerInternals:
    failures: int = 0
    last_failure_time: float = _NEVER
    last_success_time: float = _NEVER
    total_calls: int = 0
    total_failures: int = 0
    success_count: int = 0
    half_open_calls: int = 0
    rolling_window: Deque[bool] = field(
        default_factory=lambda: deque(maxlen=100)
    )


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit is OPEN or HALF_OPEN at capacity."""

    __slots__ = ("snapshot",)

    def __init__(self, snapshot: CircuitBreakerSnapshot) -> None:
        self.snapshot = snapshot
        super().__init__(
            f"Circuit breaker is {snapshot.state!r} "
            f"({snapshot.consecutive_failures} consecutive failures)"
        )


class CircuitBreaker:
    """Thread-safe circuit breaker."""

    __slots__ = (
        "failure_threshold",
        "reset_timeout",
        "half_open_max_calls",
        "rolling_window_size",
        "use_time_based_decay",
        "decay_factor",
        "_internals",
        "_lock",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        rolling_window_size: int = 100,
        use_time_based_decay: bool = False,
        decay_factor: float = 0.5,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be ≥ 1")
        if reset_timeout <= 0:
            raise ValueError("reset_timeout must be > 0")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be ≥ 1")
        if rolling_window_size < 1:
            raise ValueError("rolling_window_size must be ≥ 1")
        if not 0 < decay_factor <= 1:
            raise ValueError("decay_factor must be in (0, 1]")

        self.failure_threshold: Final[int] = failure_threshold
        self.reset_timeout: Final[float] = reset_timeout
        self.half_open_max_calls: Final[int] = half_open_max_calls
        self.rolling_window_size: Final[int] = rolling_window_size
        self.use_time_based_decay: Final[bool] = use_time_based_decay
        self.decay_factor: Final[float] = decay_factor

        self._internals = _CircuitBreakerInternals(
            rolling_window=deque(maxlen=rolling_window_size)
        )
        self._lock = threading.RLock()

    def _compute_state(self) -> CircuitState:
        s = self._internals
        if s.failures < self.failure_threshold:
            return CircuitState.CLOSED
        elapsed = monotonic() - s.last_failure_time
        return CircuitState.OPEN if elapsed <= self.reset_timeout else CircuitState.HALF_OPEN

    def can_execute(self) -> bool:
        with self._lock:
            state = self._compute_state()
            if state is CircuitState.CLOSED:
                return True
            if state is CircuitState.OPEN:
                return False

            # HALF_OPEN
            s = self._internals
            if s.half_open_calls < self.half_open_max_calls:
                s.half_open_calls += 1
                logger.debug(
                    "Circuit breaker HALF_OPEN probe %d/%d",
                    s.half_open_calls,
                    self.half_open_max_calls,
                )
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            s = self._internals
            state_before = self._compute_state()

            if state_before is CircuitState.HALF_OPEN:
                s.failures = 0
                s.half_open_calls = 0
                logger.info("Circuit breaker → CLOSED")
            elif state_before is CircuitState.CLOSED:
                if self.use_time_based_decay:
                    s.failures = max(0, int(s.failures * self.decay_factor))
                else:
                    s.failures = 0
            else:  # OPEN
                logger.debug("Success recorded while circuit OPEN – ignoring for state transition")

            s.last_success_time = monotonic()
            s.total_calls += 1
            s.success_count += 1
            s.rolling_window.append(True)

    def record_failure(self) -> None:
        with self._lock:
            s = self._internals
            s.failures += 1
            s.last_failure_time = monotonic()
            s.total_calls += 1
            s.total_failures += 1
            s.rolling_window.append(False)

            if s.failures >= self.failure_threshold:
                s.half_open_calls = 0
                logger.warning(
                    "Circuit breaker → OPEN after %d consecutive failures",
                    s.failures,
                )

    def get_state(self) -> CircuitBreakerSnapshot:
        with self._lock:
            s = self._internals
            state = self._compute_state()

            rolling_total = len(s.rolling_window)
            recent_successes = sum(s.rolling_window)
            recent_success_rate = recent_successes / rolling_total if rolling_total else 0.0
            failure_rate = s.total_failures / s.total_calls if s.total_calls else 0.0

            return CircuitBreakerSnapshot(
                state=state,
                consecutive_failures=s.failures,
                total_calls=s.total_calls,
                total_failures=s.total_failures,
                success_count=s.success_count,
                failure_rate=failure_rate,
                recent_success_rate=recent_success_rate,
                recent_error_rate=1.0 - recent_success_rate,
                rolling_window_size=rolling_total,
                last_failure_time=s.last_failure_time,
                last_success_time=s.last_success_time,
                half_open_calls=s.half_open_calls,
                half_open_max_calls=self.half_open_max_calls,
            )

    def get_health_metrics(self) -> HealthMetrics:
        snap = self.get_state()
        now = monotonic()
        return HealthMetrics(
            circuit_state=snap.state,
            is_healthy=snap.is_closed,
            is_degraded=snap.is_half_open,
            consecutive_failures=snap.consecutive_failures,
            total_operations=snap.total_calls,
            error_rate=snap.failure_rate,
            recent_success_rate=snap.recent_success_rate,
            rolling_window_size=snap.rolling_window_size,
            time_since_last_failure=(
                now - snap.last_failure_time
                if snap.last_failure_time != _NEVER
                else None
            ),
            time_since_last_success=(
                now - snap.last_success_time
                if snap.last_success_time != _NEVER
                else None
            ),
        )

    def reset(self) -> None:
        """Manually force the circuit back to CLOSED state (e.g., after a deployment)."""
        with self._lock:
            self._internals = _CircuitBreakerInternals(
                rolling_window=deque(maxlen=self.rolling_window_size)
            )
            logger.info("Circuit breaker manually reset → CLOSED")

    def __call__(self, func: _F) -> _F:
        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.can_execute():
                raise CircuitBreakerOpenError(self.get_state())
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception:
                self.record_failure()
                raise

        return _wrapper  # type: ignore[return-value]

    with_circuit_breaker = __call__

    @contextmanager
    def protected(self) -> Generator[None, None, None]:
        if not self.can_execute():
            raise CircuitBreakerOpenError(self.get_state())
        try:
            yield
            self.record_success()
        except Exception:
            self.record_failure()
            raise

    def __repr__(self) -> str:
        snap = self.get_state()
        return (
            f"{type(self).__name__}("
            f"state={snap.state!r}, "
            f"failures={snap.consecutive_failures}/{self.failure_threshold}, "
            f"reset_timeout={self.reset_timeout}s, "
            f"error_rate={snap.failure_rate:.1%})"
        )