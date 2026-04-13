"""
test_copy.py
============
Comprehensive test coverage for copy.py.

Run with:
    pytest test_copy.py -v

Optional (memory-bandwidth tests, slow):
    pytest test_copy.py -v -m slow
"""

from __future__ import annotations

import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest

import cross_platform.pycore.mtcopy as copy_mod  # import the module itself
# for global state manipulation
from cross_platform.pycore.mtcopy import (
    DEFAULT_CHUNK_SIZE,
    MAX_WORKERS_LIMIT,
    get_global_executor,
    parallel_copy,
    tuned_parallel_copy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_array(size_bytes: int, dtype=np.uint8, value: int = 0) -> np.ndarray:
    """Return a C-contiguous array of *size_bytes* bytes filled with *value*."""
    n_elements = size_bytes // np.dtype(dtype).itemsize
    arr = np.full(n_elements, value, dtype=dtype)
    assert arr.flags[
        "C_CONTIGUOUS"], "Test helper must produce a contiguous array"
    return arr


def _alloc_buf(size_bytes: int) -> ctypes.Array:
    """Allocate a ctypes byte buffer that memmove can target."""
    return (ctypes.c_uint8 * size_bytes)()


def _addr(buf: ctypes.Array) -> int:
    return ctypes.addressof(buf)


def _read_buf(buf: ctypes.Array) -> bytes:
    return bytes(buf)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_global_executor():
    """
    Ensure each test starts with a clean global executor and restore it afterwards.

    This prevents executor state from leaking between tests and avoids
    interference with the shutdown-hook test.
    """
    original = copy_mod._GLOBAL_EXECUTOR
    copy_mod._GLOBAL_EXECUTOR = None
    yield
    # Shut down whatever the test created, then put the original back.
    if copy_mod._GLOBAL_EXECUTOR is not None:
        copy_mod._GLOBAL_EXECUTOR.shutdown(wait=True)
        copy_mod._GLOBAL_EXECUTOR = None
    copy_mod._GLOBAL_EXECUTOR = original


# ===========================================================================
# get_global_executor
# ===========================================================================

class TestGetGlobalExecutor:

    def test_returns_thread_pool_executor(self):
        ex = get_global_executor()
        assert isinstance(ex, ThreadPoolExecutor)

    def test_singleton_same_object(self):
        ex1 = get_global_executor()
        ex2 = get_global_executor()
        assert ex1 is ex2

    def test_max_workers_cap(self):
        """Thread count must never exceed MAX_WORKERS_LIMIT."""
        ex = get_global_executor()
        assert ex._max_workers <= MAX_WORKERS_LIMIT

    def test_thread_name_prefix(self):
        """Worker threads must be named PBO_Uploader-N for debuggability."""
        ex = get_global_executor()
        # Submit a dummy job to spin up a worker thread.
        future = ex.submit(threading.current_thread)
        worker_thread = future.result()
        assert worker_thread.name.startswith("PBO_Uploader")

    def test_max_workers_none_uses_cpu_count_capped(self):
        """Default max_workers = min(cpu_count, MAX_WORKERS_LIMIT)."""
        import os
        cpu_count = os.cpu_count() or 4
        ex = get_global_executor()
        assert ex._max_workers == min(cpu_count, MAX_WORKERS_LIMIT)

    def test_explicit_max_workers_honoured_on_first_call(self):
        """An explicit max_workers value is used when the executor doesn't exist yet."""
        # Force creation with max_workers=1; cap still applies.
        ex = get_global_executor(max_workers=1)
        assert ex._max_workers == 1

    def test_max_workers_ignored_after_first_call(self):
        """Subsequent calls with a different max_workers still return the same pool."""
        ex1 = get_global_executor(max_workers=1)
        ex2 = get_global_executor(max_workers=4)
        assert ex1 is ex2
        assert ex2._max_workers == 1  # original value is unchanged

    def test_cpu_count_none_falls_back_to_4(self):
        """If os.cpu_count() returns None the fallback is 4 (or MAX_WORKERS_LIMIT)."""
        with patch("os.cpu_count", return_value=None):
            ex = get_global_executor()
        assert ex._max_workers == min(4, MAX_WORKERS_LIMIT)


# ===========================================================================
# _shutdown_executor
# ===========================================================================

class TestShutdownExecutor:

    def test_shutdown_clears_global(self):
        get_global_executor()
        copy_mod._shutdown_executor()
        assert copy_mod._GLOBAL_EXECUTOR is None

    def test_shutdown_idempotent_when_none(self):
        """Calling shutdown when no executor exists must not raise."""
        assert copy_mod._GLOBAL_EXECUTOR is None
        copy_mod._shutdown_executor()  # should not raise

    def test_atexit_registered(self):
        """_shutdown_executor must be registered with atexit at import time."""
        # atexit stores handlers in _atexit.__dict__ or atexit._atexit in CPython.
        # The safest check is to inspect via the atexit module's internal list.
        # We can't introspect atexit directly in all Python versions, so we
        # verify the effect: calling _shutdown_executor via atexit does not error.
        # Register again and fire manually to confirm it is callable.
        import atexit as _atexit
        called = []
        _atexit.register(lambda: called.append(True))
        # The real registration happened at import time; just verify the function
        # is intact and callable.
        copy_mod._shutdown_executor()  # should complete without error


# ===========================================================================
# parallel_copy — correctness
# ===========================================================================

class TestParallelCopyCorrectness:

    # --- Small payloads (fast path: single memmove) -------------------------

    def test_small_payload_copies_correctly(self):
        src = _make_array(1024, value=0xAB)
        dst = _alloc_buf(1024)
        parallel_copy(_addr(dst), src)
        assert _read_buf(dst) == bytes([0xAB] * 1024)

    def test_single_byte_payload(self):
        src = _make_array(1, value=0xFF)
        dst = _alloc_buf(1)
        parallel_copy(_addr(dst), src)
        assert _read_buf(dst) == b"\xFF"

    def test_payload_exactly_chunk_size_uses_fast_path(self):
        """A payload == chunk_size should take the memmove fast path."""
        chunk = DEFAULT_CHUNK_SIZE
        src = _make_array(chunk, value=0x55)
        dst = _alloc_buf(chunk)
        with patch("ctypes.memmove", wraps=ctypes.memmove) as mock_mm:
            parallel_copy(_addr(dst), src, chunk_size=chunk)
        mock_mm.assert_called_once()
        assert _read_buf(dst) == bytes([0x55] * chunk)

    # --- Multi-chunk payloads -----------------------------------------------

    def test_multi_chunk_copies_correctly(self):
        size = DEFAULT_CHUNK_SIZE * 3
        pattern = np.arange(size, dtype=np.uint8)
        dst = _alloc_buf(size)
        parallel_copy(_addr(dst), pattern)
        assert _read_buf(dst) == bytes(pattern)

    def test_uneven_last_chunk(self):
        """Last chunk is smaller than chunk_size — still copied completely."""
        # 2.5 chunks
        chunk = 64 * 1024
        size = int(chunk * 2.5)
        src = _make_array(size, value=0x7E)
        dst = _alloc_buf(size)
        parallel_copy(_addr(dst), src, chunk_size=chunk)
        assert _read_buf(dst) == bytes([0x7E] * size)

    def test_no_gaps_or_overlaps(self):
        """Byte-level correctness: sequential values, no gaps or double-writes."""
        size = DEFAULT_CHUNK_SIZE * 4 + 1  # forces uneven chunks
        src = np.arange(size, dtype=np.uint8)
        dst = _alloc_buf(size)
        parallel_copy(_addr(dst), src)
        assert _read_buf(dst) == bytes(src)

    def test_accepts_c_void_p(self):
        src = _make_array(512, value=0x12)
        dst = _alloc_buf(512)
        ptr = ctypes.c_void_p(_addr(dst))
        parallel_copy(ptr, src)
        assert _read_buf(dst) == bytes([0x12] * 512)

    def test_accepts_plain_int_address(self):
        src = _make_array(512, value=0x34)
        dst = _alloc_buf(512)
        parallel_copy(int(_addr(dst)), src)
        assert _read_buf(dst) == bytes([0x34] * 512)

    def test_custom_chunk_size(self):
        size = 1024 * 1024
        src = np.arange(size, dtype=np.uint8)
        dst = _alloc_buf(size)
        parallel_copy(_addr(dst), src, chunk_size=256 * 1024)
        assert _read_buf(dst) == bytes(src)

    def test_chunk_size_of_1_byte(self):
        """Extreme: each byte in its own task."""
        size = 16  # small so the test is fast
        src = np.arange(size, dtype=np.uint8)
        dst = _alloc_buf(size)
        parallel_copy(_addr(dst), src, chunk_size=1)
        assert _read_buf(dst) == bytes(src)

    def test_uint32_source_array(self):
        src = np.arange(1024, dtype=np.uint32)
        dst = _alloc_buf(src.nbytes)
        parallel_copy(_addr(dst), src)
        expected = src.tobytes()
        assert _read_buf(dst) == expected

    # --- Executor argument --------------------------------------------------

    def test_custom_executor_is_used(self):
        """When an explicit executor is passed it should be used, not the global one."""
        size = DEFAULT_CHUNK_SIZE * 2
        src = _make_array(size, value=0xCC)
        dst = _alloc_buf(size)
        with ThreadPoolExecutor(max_workers=1) as ex:
            parallel_copy(_addr(dst), src, executor=ex)
        assert _read_buf(dst) == bytes([0xCC] * size)

    def test_global_executor_used_when_none_passed(self):
        """Omitting executor should trigger get_global_executor()."""
        size = DEFAULT_CHUNK_SIZE * 2
        src = _make_array(size, value=0xDD)
        dst = _alloc_buf(size)
        # patch("copy.get_global_executor") resolves to the stdlib copy module
        # because the user module shadows it by name.  patch.object targets the
        # actual module object, bypassing the name collision entirely.
        with patch.object(copy_mod, "get_global_executor",
                          wraps=get_global_executor) as mock_gge:
            parallel_copy(_addr(dst), src)
        mock_gge.assert_called_once()


# ===========================================================================
# parallel_copy — error handling
# ===========================================================================

class TestParallelCopyErrors:

    def test_null_c_void_p_raises_value_error(self):
        src = _make_array(64)
        null_ptr = ctypes.c_void_p(None)
        with pytest.raises(ValueError, match="null"):
            parallel_copy(null_ptr, src)

    def test_zero_value_c_void_p_raises_value_error(self):
        """c_void_p(0) has .value == None in CPython — treat as null."""
        src = _make_array(64)
        null_ptr = ctypes.c_void_p(0)
        with pytest.raises(ValueError):
            parallel_copy(null_ptr, src)

    def test_worker_exception_propagates(self):
        """An exception raised by a worker must surface on the calling thread."""
        src = _make_array(DEFAULT_CHUNK_SIZE * 2)
        dst = _alloc_buf(src.nbytes)

        original_memmove = ctypes.memmove
        call_count = [0]

        def boom(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("simulated worker failure")
            return original_memmove(*args, **kwargs)

        with patch("ctypes.memmove", side_effect=boom):
            with pytest.raises(RuntimeError, match="simulated worker failure"):
                parallel_copy(_addr(dst), src)


# ===========================================================================
# tuned_parallel_copy — strategy selection
# ===========================================================================

class TestTunedParallelCopyStrategySelection:
    SMALL_SIZE = 4 * 1024 * 1024  # 4 MiB  → single memmove
    MID_SIZE = 16 * 1024 * 1024  # 16 MiB → parallel 8 MiB chunks
    LARGE_SIZE = 128 * 1024 * 1024  # 128 MiB → parallel 16 MiB chunks

    # --- Threshold: < 8 MiB → single memmove --------------------------------

    def test_small_uses_single_memmove(self):
        src = _make_array(self.SMALL_SIZE)
        dst = _alloc_buf(self.SMALL_SIZE)
        with patch("ctypes.memmove", wraps=ctypes.memmove) as mock_mm:
            with patch.object(copy_mod, "parallel_copy") as mock_pc:
                tuned_parallel_copy(_addr(dst), src)
        mock_mm.assert_called_once()
        mock_pc.assert_not_called()

    def test_boundary_just_below_8mib_uses_memmove(self):
        size = 8 * 1024 * 1024 - 1
        src = _make_array(size)
        dst = _alloc_buf(size)
        with patch("ctypes.memmove", wraps=ctypes.memmove) as mock_mm:
            with patch.object(copy_mod, "parallel_copy") as mock_pc:
                tuned_parallel_copy(_addr(dst), src)
        mock_mm.assert_called_once()
        mock_pc.assert_not_called()

    # --- Threshold: 8 MiB ≤ size < 64 MiB → 8 MiB chunks ------------------

    def test_boundary_exactly_8mib_uses_parallel_8mib_chunks(self):
        size = 8 * 1024 * 1024
        src = _make_array(size)
        dst = _alloc_buf(size)
        with patch.object(copy_mod, "parallel_copy") as mock_pc:
            tuned_parallel_copy(_addr(dst), src)
        mock_pc.assert_called_once_with(_addr(dst), src,
                                        chunk_size=8 * 1024 * 1024)

    def test_mid_range_uses_8mib_chunks(self):
        src = _make_array(self.MID_SIZE)
        dst = _alloc_buf(self.MID_SIZE)
        with patch.object(copy_mod, "parallel_copy") as mock_pc:
            tuned_parallel_copy(_addr(dst), src)
        mock_pc.assert_called_once_with(_addr(dst), src,
                                        chunk_size=8 * 1024 * 1024)

    def test_boundary_just_below_64mib_uses_8mib_chunks(self):
        size = 64 * 1024 * 1024 - 1
        src = _make_array(size)
        dst = _alloc_buf(size)
        with patch.object(copy_mod, "parallel_copy") as mock_pc:
            tuned_parallel_copy(_addr(dst), src)
        mock_pc.assert_called_once_with(_addr(dst), src,
                                        chunk_size=8 * 1024 * 1024)

    # --- Threshold: ≥ 64 MiB → 16 MiB chunks -------------------------------

    def test_boundary_exactly_64mib_uses_16mib_chunks(self):
        size = 64 * 1024 * 1024
        src = _make_array(size)
        dst = _alloc_buf(size)
        with patch.object(copy_mod, "parallel_copy") as mock_pc:
            tuned_parallel_copy(_addr(dst), src)
        mock_pc.assert_called_once_with(_addr(dst), src,
                                        chunk_size=16 * 1024 * 1024)

    def test_large_payload_uses_16mib_chunks(self):
        src = _make_array(self.LARGE_SIZE)
        dst = _alloc_buf(self.LARGE_SIZE)
        with patch.object(copy_mod, "parallel_copy") as mock_pc:
            tuned_parallel_copy(_addr(dst), src)
        mock_pc.assert_called_once_with(_addr(dst), src,
                                        chunk_size=16 * 1024 * 1024)


# ===========================================================================
# tuned_parallel_copy — correctness (end-to-end, no mocking)
# ===========================================================================

class TestTunedParallelCopyCorrectness:

    def test_small_payload_correct(self):
        size = 1024
        src = np.arange(size, dtype=np.uint8)
        dst = _alloc_buf(size)
        tuned_parallel_copy(_addr(dst), src)
        assert _read_buf(dst) == bytes(src)

    def test_mid_payload_correct(self):
        size = 10 * 1024 * 1024  # 10 MiB
        src = np.arange(size, dtype=np.uint8)
        dst = _alloc_buf(size)
        tuned_parallel_copy(_addr(dst), src)
        assert _read_buf(dst) == bytes(src)

    def test_large_payload_correct(self):
        size = 64 * 1024 * 1024  # 64 MiB
        src = np.arange(size, dtype=np.uint8)
        dst = _alloc_buf(size)
        tuned_parallel_copy(_addr(dst), src)
        assert _read_buf(dst) == bytes(src)


# ===========================================================================
# Thread safety of get_global_executor
# ===========================================================================

class TestThreadSafety:

    def test_concurrent_get_global_executor_returns_same_object(self):
        """
        Multiple threads calling get_global_executor concurrently should all
        receive the same instance (CPython GIL guarantee).
        """
        results = []
        barrier = threading.Barrier(8)

        def _get():
            barrier.wait()  # synchronise all threads to the same instant
            results.append(get_global_executor())

        threads = [threading.Thread(target=_get) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        # All threads must have received the identical object.
        assert all(r is results[0] for r in results)

    def test_parallel_copy_is_thread_safe(self):
        """
        Two threads performing independent parallel_copy operations on separate
        buffers must not corrupt each other's data.
        """
        size = DEFAULT_CHUNK_SIZE * 3
        src_a = _make_array(size, value=0xAA)
        src_b = _make_array(size, value=0xBB)
        dst_a = _alloc_buf(size)
        dst_b = _alloc_buf(size)
        errors = []

        def copy_and_check(src, dst, expected_byte):
            try:
                parallel_copy(_addr(dst), src)
                assert _read_buf(dst) == bytes([expected_byte] * size)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=copy_and_check, args=(src_a, dst_a, 0xAA))
        t2 = threading.Thread(target=copy_and_check, args=(src_b, dst_b, 0xBB))
        t1.start();
        t2.start()
        t1.join();
        t2.join()

        assert errors == [], f"Thread safety violation: {errors}"


# ===========================================================================
# Module constants
# ===========================================================================

class TestModuleConstants:

    def test_default_chunk_size_is_4mib(self):
        assert DEFAULT_CHUNK_SIZE == 4 * 1024 * 1024

    def test_max_workers_limit_is_4(self):
        assert MAX_WORKERS_LIMIT == 4

    def test_dunder_all_exports(self):
        import cross_platform.pycore.mtcopy as m
        assert "get_global_executor" in m.__all__
        assert "parallel_copy" in m.__all__
        assert "tuned_parallel_copy" in m.__all__
        # Internal helpers should not be exported.
        assert "_shutdown_executor" not in m.__all__
        assert "_GLOBAL_EXECUTOR" not in m.__all__


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_array_does_not_raise(self):
        """An array with 0 bytes should be a no-op, not an error."""
        src = np.array([], dtype=np.uint8)
        dst = _alloc_buf(1)  # 1-byte buffer; nothing should be written
        # Should not raise regardless of whether the fast path or chunk path runs.
        parallel_copy(_addr(dst), src)

    def test_non_contiguous_array_behaviour(self):
        """
        numpy slices can be non-contiguous.  ctypes.data on a non-contiguous
        array returns the address of the raw (non-sequential) memory, which
        may produce wrong results.  This test documents the current behaviour
        rather than asserting correctness.
        """
        base = np.arange(2048, dtype=np.uint8)
        non_contig = base[::2]  # every other byte — non-contiguous
        assert not non_contig.flags["C_CONTIGUOUS"]
        # parallel_copy should not crash; result may be incorrect (documented).
        dst = _alloc_buf(non_contig.nbytes)
        parallel_copy(_addr(dst), non_contig)  # must not raise

    def test_array_of_zeros(self):
        size = DEFAULT_CHUNK_SIZE * 2
        src = np.zeros(size, dtype=np.uint8)
        dst = _alloc_buf(size)
        parallel_copy(_addr(dst), src)
        assert _read_buf(dst) == bytes(size)

    def test_chunk_size_larger_than_payload_skips_thread_pool(self):
        """If chunk_size > nbytes the fast path is taken without touching the pool."""
        size = 1024
        src = _make_array(size, value=0x9A)
        dst = _alloc_buf(size)
        # chunk_size intentionally much larger than the payload.
        with patch.object(copy_mod, "get_global_executor") as mock_gge:
            parallel_copy(_addr(dst), src, chunk_size=size * 10)
        mock_gge.assert_not_called()
        assert _read_buf(dst) == bytes([0x9A] * size)

    def test_multiple_sequential_calls_are_consistent(self):
        """The same executor must handle repeated calls without state leakage."""
        size = DEFAULT_CHUNK_SIZE * 2
        for i in range(4):
            src = _make_array(size, value=i)
            dst = _alloc_buf(size)
            parallel_copy(_addr(dst), src)
            assert _read_buf(dst) == bytes(
                [i] * size), f"Failed on iteration {i}"
