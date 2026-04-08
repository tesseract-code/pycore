"""
copy.py
=======
Multi-threaded memory copy utilities for high-throughput CPU→GPU transfers.

This module exists to saturate memory bandwidth when copying large image
buffers into mapped PBO memory.  A single ``ctypes.memmove`` call is
single-threaded; splitting the buffer into chunks and dispatching each chunk
to a worker thread can approach the system's peak memory bandwidth on
NUMA-aware hardware.

When to use which function
--------------------------
* :func:`parallel_copy` — explicit control over chunk size and executor.
  Use when you have measured the optimal chunk size for your workload.
* :func:`tuned_parallel_copy` — automatic strategy selection based on
  transfer size.  Prefer this for general-purpose call sites.

Executor lifecycle
------------------
A single :class:`~concurrent.futures.ThreadPoolExecutor` is shared across all
callers via :func:`get_global_executor`.  It is created lazily on first use
and registered for graceful shutdown at interpreter exit via :func:`atexit`.
Threads are named ``PBO_Uploader-N`` to make them identifiable in profilers
and debuggers.

Thread-safety
-------------
:func:`get_global_executor` uses a module-level global.  In CPython the GIL
makes the ``if _GLOBAL_EXECUTOR is None`` check safe against concurrent
initialisation from multiple threads, but this is an implementation detail.
Callers that spin up their own threads before the first call to
:func:`get_global_executor` should ensure the executor is created on the main
thread first to avoid any ambiguity.

Performance notes
-----------------
* The parallel path has a fixed overhead from task scheduling.  For small
  transfers this overhead exceeds the benefit, which is why
  :func:`parallel_copy` falls back to a single ``memmove`` when the payload
  fits in one chunk, and why :func:`tuned_parallel_copy` starts the parallel
  path at 8 MiB.
* Workers capture ``ctypes.memmove`` as a local reference to avoid a
  per-call global lookup inside the tight copy loop.
* ``executor.map`` is used instead of ``submit`` + ``as_completed`` because
  it preserves submission order and surfaces exceptions on the calling thread,
  making error attribution straightforward.
"""

from __future__ import annotations

import atexit
import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

import numpy as np

__all__ = [
    "get_global_executor",
    "parallel_copy",
    "tuned_parallel_copy",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default number of bytes assigned to each worker thread in
#: :func:`parallel_copy`.  4 MiB balances scheduling overhead against the
#: benefit of parallelism across typical L3 cache sizes.
DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024   # 4 MiB

#: Hard upper bound on the thread count used by :func:`get_global_executor`.
#: PBO uploads are memory-bandwidth-bound, not compute-bound; more than four
#: threads rarely improves throughput and increases context-switching overhead.
MAX_WORKERS_LIMIT = 4

# ---------------------------------------------------------------------------
# Global executor
# ---------------------------------------------------------------------------

# Lazily initialised by get_global_executor().  Module-private; callers must
# use get_global_executor() rather than accessing this directly so that the
# shutdown hook and the initialisation logic stay in one place.
_GLOBAL_EXECUTOR: Optional[ThreadPoolExecutor] = None


def get_global_executor(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    """
    Return the process-wide :class:`~concurrent.futures.ThreadPoolExecutor`.

    Creates the executor on the first call and reuses it thereafter.  The
    thread count is capped at :data:`MAX_WORKERS_LIMIT` because PBO transfers
    are memory-bandwidth-bound: additional threads beyond this point contend
    for the same memory bus without improving throughput.

    Args:
        max_workers: Number of worker threads.  When ``None`` (the default),
            the count is ``min(os.cpu_count(), MAX_WORKERS_LIMIT)``.  Ignored
            on all calls after the first — the pool size is fixed at creation.

    Returns:
        The shared :class:`~concurrent.futures.ThreadPoolExecutor` instance.

    Note:
        The executor is registered for shutdown via :mod:`atexit` at module
        import time.  Calling this function after the interpreter has begun
        its exit sequence (e.g. inside another ``atexit`` handler) may return
        a pool that is already shut down.
    """
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is None:
        if max_workers is None:
            # os.cpu_count() can return None on some platforms (e.g. certain
            # containers); fall back to 4 in that case.
            max_workers = min(os.cpu_count() or 4, MAX_WORKERS_LIMIT)

        _GLOBAL_EXECUTOR = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="PBO_Uploader",
        )
    return _GLOBAL_EXECUTOR


def _shutdown_executor() -> None:
    """
    Graceful shutdown hook registered with :mod:`atexit`.

    Called automatically when the interpreter begins its exit sequence.
    ``wait=False`` allows the process to exit without blocking for in-flight
    copy tasks to complete — acceptable here because partial PBO transfers at
    shutdown produce no persistent side-effects.
    """
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is not None:
        _GLOBAL_EXECUTOR.shutdown(wait=False)
        _GLOBAL_EXECUTOR = None


# Register at import time so the pool is always cleaned up, even if the
# caller never explicitly calls a shutdown function.
atexit.register(_shutdown_executor)


# ---------------------------------------------------------------------------
# Copy primitives
# ---------------------------------------------------------------------------

def parallel_copy(
    dst_ptr: Union[int, ctypes.c_void_p],
    src_data: np.ndarray,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    executor: Optional[ThreadPoolExecutor] = None,
) -> None:
    """
    Copy a NumPy array into a raw memory address using multiple threads.

    Splits ``src_data`` into contiguous byte chunks and copies each chunk
    with ``ctypes.memmove`` on a pool thread.  For payloads smaller than
    ``chunk_size`` a single-threaded ``memmove`` is used directly to avoid
    the scheduling overhead of the thread pool.

    The destination memory must remain valid and writable for the entire
    duration of the call.  Typical callers pass a mapped PBO address obtained
    from ``glMapBuffer`` / ``glMapBufferRange``; the PBO must not be unmapped
    until this function returns.

    Args:
        dst_ptr:    Destination memory address.  Accepts either a plain
                    ``int`` or a ``ctypes.c_void_p``.  A ``c_void_p`` with
                    ``value=None`` is rejected with :exc:`ValueError`.
        src_data:   Source array.  Must be C-contiguous (standard NumPy
                    arrays are contiguous by default; slices may not be).
        chunk_size: Byte size of each worker's slice.  Defaults to
                    :data:`DEFAULT_CHUNK_SIZE` (4 MiB).  Larger chunks reduce
                    scheduling overhead but may leave CPU cores idle at the
                    end of an uneven transfer.
        executor:   Pool to use.  When ``None`` (the default), the shared
                    global executor from :func:`get_global_executor` is used.

    Raises:
        ValueError: If ``dst_ptr`` is a ``ctypes.c_void_p`` whose ``.value``
                    is ``None`` (i.e. a null pointer).

    Note:
        NumPy releases the GIL for ``ctypes.data`` access.  The workers
        therefore run truly in parallel on CPython despite the GIL, making
        this approach effective for memory-bandwidth-bound transfers.
    """
    nbytes = src_data.nbytes

    # Normalise the destination pointer to a plain integer address.
    # c_void_p.value is None when the underlying pointer is NULL.
    if isinstance(dst_ptr, ctypes.c_void_p):
        dst_addr = dst_ptr.value
    else:
        dst_addr = dst_ptr

    if dst_addr is None:
        raise ValueError(
            "Destination pointer cannot be None (received a null c_void_p)."
        )

    # Fast path: payload fits in a single chunk — skip thread pool overhead.
    if nbytes <= chunk_size:
        ctypes.memmove(dst_addr, src_data.ctypes.data, nbytes)
        return

    executor = executor or get_global_executor()
    src_addr = src_data.ctypes.data

    # Capture memmove as a local to avoid a per-call global lookup inside the
    # worker closure.  This is measurable at high chunk counts.
    memmove = ctypes.memmove

    # Build the task list before submitting to avoid interleaving
    # list-construction overhead with worker execution.
    tasks: list[tuple[int, int, int]] = []
    offset = 0
    while offset < nbytes:
        size = min(chunk_size, nbytes - offset)
        tasks.append((dst_addr + offset, src_addr + offset, size))
        offset += size

    def _worker(args: tuple[int, int, int]) -> None:
        # Unpack inline rather than using args[0]/[1]/[2] to give the
        # interpreter a single UNPACK_SEQUENCE opcode instead of three
        # BINARY_SUBSCR calls.
        dst, src, n = args
        memmove(dst, src, n)

    # executor.map blocks until all chunks are complete and re-raises any
    # worker exception on the calling thread, which is the desired behaviour
    # for an upload pipeline where partial copies are not recoverable.
    list(executor.map(_worker, tasks))


def tuned_parallel_copy(
    dst_ptr: Union[int, ctypes.c_void_p],
    src_data: np.ndarray,
) -> None:
    """
    Copy ``src_data`` to ``dst_ptr`` using an automatically selected strategy.

    Chooses between a single ``memmove`` and :func:`parallel_copy` with an
    appropriate chunk size based on the total transfer size.  The thresholds
    are tuned for typical image workloads on desktop hardware:

    ==================  ========================================================
    Payload size        Strategy
    ==================  ========================================================
    ``< 8 MiB``         Single-threaded ``memmove`` — thread overhead exceeds
                        the bandwidth benefit at this scale.
    ``8 MiB – 64 MiB``  Parallel with 8 MiB chunks — two to eight workers,
                        suitable for HD/4K frames at moderate bit depths.
    ``≥ 64 MiB``        Parallel with 16 MiB chunks — limits task-list length
                        for very large allocations (e.g. 16-bit 8K frames).
    ==================  ========================================================

    Args:
        dst_ptr:  Destination memory address (``int`` or ``ctypes.c_void_p``).
                  Must remain valid for the duration of the call.
        src_data: Source NumPy array to copy.

    Note:
        The thresholds were chosen empirically.  Workloads with non-standard
        frame sizes or memory subsystem characteristics may benefit from
        calling :func:`parallel_copy` directly with a measured chunk size.
    """
    nbytes = src_data.nbytes

    if nbytes < 8 * 1024 * 1024:
        # Small transfer: scheduling overhead outweighs parallelism benefit.
        ctypes.memmove(dst_ptr, src_data.ctypes.data, nbytes)
        return
    elif nbytes < 64 * 1024 * 1024:
        # Mid-range: 8 MiB chunks keep 2–8 workers busy across typical frames.
        chunk_size = 8 * 1024 * 1024
    else:
        # Large transfer: wider chunks reduce task-list overhead for huge
        # allocations without significantly reducing parallelism.
        chunk_size = 16 * 1024 * 1024

    parallel_copy(dst_ptr, src_data, chunk_size=chunk_size)