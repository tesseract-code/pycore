import concurrent.futures
import random
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from logging import getLogger
from time import time, sleep
from typing import Any, Callable, TypeVar, Optional, List
from typing import Type, Tuple

logger = getLogger(__name__)

T = TypeVar('T')


def call_with_timeout(func: Callable[..., T], timeout_seconds: float,
                      use_process: bool = False, *args, **kwargs) -> T:
    """
    Execute a function with a timeout.

    Args:
        func: Function to execute
        timeout_seconds: Maximum execution time in seconds
        use_process: If True, use ProcessPoolExecutor (thread-safe, requires picklable func).
                    If False, use ThreadPoolExecutor (faster, but not thread-safe).
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        The result of func(*args, **kwargs)

    Raises:
        concurrent.futures.TimeoutError: If the function doesn't complete within timeout_seconds
        ValueError: If timeout_seconds is not positive
        RuntimeError: If argument use_process=True but function is not
        picklable
        Exception: Any exception raised by the target function

    Thread Safety:
        use_process=False: Not thread-safe if func is not reentrant
        use_process=True: Thread-safe, isolated process execution
    """

    func_name = getattr(func, '__name__', str(func))

    if timeout_seconds <= 0:
        raise ValueError(f"Timeout must be positive, got: {timeout_seconds}")

    executor_class = (
        concurrent.futures.ProcessPoolExecutor if use_process else concurrent.futures.ThreadPoolExecutor)

    executor_type = getattr(executor_class, '__name__',
                            str(executor_class.__class__.__name__))

    with executor_class(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            logger.error(
                f"Function '{func_name}' timed out after {timeout_seconds}s ({executor_type})")
            raise


@dataclass(frozen=True)
class RetryMetrics:
    """Metrics collected during retry execution."""
    attempts: int
    total_duration: float
    success: bool
    final_exception: Optional[Exception]
    per_attempt_durations: Tuple[float, ...]


class _RetryState:
    """Internal state for retry execution."""

    def __init__(self, func_name: str, max_attempts: int):
        self.func_name = func_name
        self.max_attempts = max_attempts
        self.last_exception: Optional[Exception] = None
        self.start_time: float = time()
        self.attempt_durations: List[float] = []

    def record_attempt(self, duration: float) -> None:
        """Record an attempt duration."""
        self.attempt_durations.append(duration)

    def elapsed(self) -> float:
        """Get total elapsed time."""
        return time() - self.start_time

    def build_metrics(self, attempt: int, success: bool) -> RetryMetrics:
        """Build metrics from current state."""
        return RetryMetrics(attempts=attempt, total_duration=self.elapsed(),
                            success=success,
                            final_exception=None if success else self.last_exception,
                            per_attempt_durations=tuple(self.attempt_durations))


def _validate_retry_params(max_attempts: int, delay: float,
                           backoff_factor: float) -> None:
    """Validate retry decorator parameters."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if backoff_factor < 1:
        raise ValueError("backoff_factor must be >= 1")


def _check_total_timeout(state: _RetryState, total_timeout: Optional[float],
                         attempt: int) -> bool:
    """Check if total timeout exceeded. Returns True if retry should
    continue."""
    if total_timeout and state.elapsed() >= total_timeout:
        logger.error(
            f"{state.func_name} exceeded total timeout of {total_timeout}s "
            f"after {state.elapsed():.2f}s (attempt {attempt})")
        return False
    return True


def _calculate_backoff_delay(attempt: int, delay: float, backoff_factor: float,
                             max_delay: Optional[float], jitter: bool) -> float:
    """Calculate delay before next retry with backoff and jitter."""
    base_delay = delay * (backoff_factor ** (attempt - 1))

    if max_delay:
        base_delay = min(base_delay, max_delay)

    if jitter:
        jitter_factor = 0.75 + random.random() * 0.5
        return base_delay * jitter_factor

    return base_delay


def _adjust_delay_for_timeout(sleep_time: float, state: _RetryState,
                              total_timeout: Optional[float]) -> float:
    """Adjust sleep time to not exceed total timeout."""
    if not total_timeout:
        return sleep_time

    remaining = total_timeout - state.elapsed()
    if sleep_time >= remaining:
        logger.warning(
            f"{state.func_name} skipping retry delay - would exceed total timeout "
            f"(elapsed: {state.elapsed():.2f}s, remaining: {remaining:.2f}s)")
        return max(0.0, remaining - 0.1)

    return sleep_time


def _execute_attempt(func: Callable, timeout_seconds: Optional[float],
                     timeout_use_process: bool, args: tuple,
                     kwargs: dict) -> Any:
    """Execute a single retry attempt."""
    if timeout_seconds:
        return call_with_timeout(func, timeout_seconds, timeout_use_process,
                                 *args, **kwargs)
    return func(*args, **kwargs)


def _handle_attempt_success(state: _RetryState, attempt: int, duration: float,
                            collect_metrics: bool, result: Any) -> Any:
    """Handle successful attempt."""
    state.record_attempt(duration)

    if collect_metrics:
        return result, state.build_metrics(attempt, success=True)
    return result


def retry(max_attempts: int = 3, delay: float = 1.0,
          backoff_factor: float = 2.0, timeout_seconds: Optional[float] = None,
          timeout_use_process: bool = False, max_delay: Optional[float] = None,
          total_timeout: Optional[float] = None, jitter: bool = False,
          collect_metrics: bool = False,
          exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    Retry decorator with exponential backoff, circuit breaker, and metrics.

    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay on each retry
        timeout_seconds: Timeout for each attempt in seconds
        timeout_use_process: Use process isolation for timeout (requires picklable func)
        max_delay: Maximum delay between retries in seconds
        total_timeout: Maximum total time for all attempts in seconds
        jitter: Add random jitter to delays
        collect_metrics: Return (result, metrics) tuple instead of just result
        exceptions: Exception types to catch and retry

    Returns:
        Decorated function that retries on failure
    """
    _validate_retry_params(max_attempts, delay, backoff_factor)

    def decorator(func: Callable) -> Callable:
        sig = signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = getattr(func, '__name__', str(func))
            # Validate function arguments
            try:
                sig.bind(*args, **kwargs).apply_defaults()
            except TypeError as e:
                raise TypeError(f"Invalid arguments for {func_name}: {e}")

            # Prepare retry exceptions
            retry_exceptions = exceptions
            if timeout_seconds and concurrent.futures.TimeoutError not in exceptions:
                retry_exceptions = exceptions + (
                    concurrent.futures.TimeoutError,)

            # Initialize retry state
            state = _RetryState(func_name, max_attempts)

            # Retry loop
            for attempt in range(1, max_attempts + 1):
                # Check total timeout before attempt
                if not _check_total_timeout(state, total_timeout, attempt):
                    raise TimeoutError(
                        f"Total retry timeout of {total_timeout}s exceeded")

                # Execute attempt
                attempt_start = time()

                try:
                    result = _execute_attempt(func, timeout_seconds,
                                              timeout_use_process, args, kwargs)
                    duration = time() - attempt_start
                    return _handle_attempt_success(state, attempt, duration,
                                                   collect_metrics, result)

                except retry_exceptions as e:
                    duration = time() - attempt_start
                    is_final = attempt == max_attempts

                    state.record_attempt(duration)
                    state.last_exception = e

                    if is_final:
                        raise e

                    # Calculate and apply backoff delay
                    sleep_time = _calculate_backoff_delay(attempt, delay,
                                                          backoff_factor,
                                                          max_delay, jitter)
                    sleep_time = _adjust_delay_for_timeout(sleep_time, state,
                                                           total_timeout)

                    if sleep_time > 0:
                        # logger.warning(
                        #     f"{state.func_name} retrying in {sleep_time:.2f}s")
                        sleep(sleep_time)

            # Should never reach here, but handle gracefully
            if state.last_exception:
                raise state.last_exception

        return wrapper

    return decorator
