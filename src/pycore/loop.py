"""
loop.py
High-Performance Asyncio Event Loop Management.
"""
import asyncio
import functools
import logging
import sys
import threading
from enum import Enum, unique, auto
from typing import Callable, Optional, TypeVar, ParamSpec, Awaitable, Protocol, Any

# Type definitions
P = ParamSpec('P')
T = TypeVar('T')

class LoopHandler(Protocol[P, T]):
    def __call__(self, *args: P.args, loop: asyncio.AbstractEventLoop, **kwargs: P.kwargs) -> T | Awaitable[T]:
        ...

@unique
class LoopPolicy(Enum):
    AUTO = auto()
    PROACTOR = auto()
    SELECTOR = auto()
    UVLOOP = auto()

@unique
class LoopStrategy(Enum):
    REUSE = auto()
    NEW_THREAD = auto()

class EventLoopManager:
    """Manages event loop creation and policy selection."""

    @classmethod
    def get_policy(cls) -> asyncio.AbstractEventLoopPolicy:
        if sys.platform == 'win32':
            return asyncio.WindowsProactorEventLoopPolicy()
        else:
            try:
                import uvloop
                return uvloop.EventLoopPolicy()
            except ImportError:
                return asyncio.DefaultEventLoopPolicy()

    @classmethod
    def create_loop(cls, policy: LoopPolicy = LoopPolicy.AUTO) -> asyncio.AbstractEventLoop:
        original_policy = asyncio.get_event_loop_policy()
        new_policy = original_policy

        if policy == LoopPolicy.AUTO:
            new_policy = cls.get_policy()
        elif policy == LoopPolicy.UVLOOP:
            try:
                import uvloop
                new_policy = uvloop.EventLoopPolicy()
            except ImportError:
                logging.warning("uvloop not found, falling back.")
                new_policy = cls.get_policy()
        elif sys.platform == 'win32':
            if policy == LoopPolicy.PROACTOR:
                new_policy = asyncio.WindowsProactorEventLoopPolicy()
            elif policy == LoopPolicy.SELECTOR:
                new_policy = asyncio.WindowsSelectorEventLoopPolicy()

        try:
            asyncio.set_event_loop_policy(new_policy)
            loop = new_policy.new_event_loop()
            return loop
        finally:
            asyncio.set_event_loop_policy(original_policy)

def with_event_loop(
        policy: LoopPolicy = LoopPolicy.AUTO,
        strategy: LoopStrategy = LoopStrategy.REUSE,
        timeout: Optional[float] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to inject an event loop into a function."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Try to reuse existing running loop
            try:
                loop = asyncio.get_running_loop()
                if strategy == LoopStrategy.REUSE:
                    return _exec_on_loop(func, loop, args, kwargs, timeout)
            except RuntimeError:
                pass

            # 2. Create new loop in this thread or separate thread
            if strategy == LoopStrategy.NEW_THREAD:
                return _exec_threaded(func, policy, args, kwargs, timeout)
            else:
                # Create loop, run, close
                loop = EventLoopManager.create_loop(policy)
                asyncio.set_event_loop(loop)
                try:
                    return _exec_on_loop(func, loop, args, kwargs, timeout)
                finally:
                    try:
                        loop.close()
                    except Exception as e:
                        logging.warning(f"Loop close error: {e}")
        return wrapper
    return decorator

def _exec_on_loop(func, loop, args, kwargs, timeout):
    """Helper to run sync or async function on a specific loop."""
    # Inject loop if function expects it
    if 'loop' in kwargs or 'loop' in func.__code__.co_varnames:
        kwargs['loop'] = loop

    if asyncio.iscoroutinefunction(func):
        coro = func(*args, **kwargs)
        if timeout:
            coro = asyncio.wait_for(coro, timeout)

        if loop.is_running():
            # If we are already in a loop, return the future/task
            return asyncio.create_task(coro)
        return loop.run_until_complete(coro)
    else:
        return func(*args, **kwargs)

def _exec_threaded(func, policy, args, kwargs, timeout):
    """Executes function in a separate thread with its own loop."""
    result = []
    exc = []

    def target():
        loop = EventLoopManager.create_loop(policy)
        asyncio.set_event_loop(loop)
        try:
            val = _exec_on_loop(func, loop, args, kwargs, timeout)
            result.append(val)
        except Exception as e:
            exc.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=target)
    t.start()
    t.join()

    if exc: raise exc[0]
    return result[0] if result else None