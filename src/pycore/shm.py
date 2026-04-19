import multiprocessing
from multiprocessing import shared_memory
from typing import Optional, Tuple, List, Dict

import numpy as np

from pycore.log.ctx import with_logger


def alloc_shm_buffer(required_size: int, dtype: np.dtype = np.uint8):
    # Note: cleanup of the buffer and view are the callers responsibility
    new_shm = shared_memory.SharedMemory(create=True, size=required_size)

    # Create NumPy view
    new_view = np.ndarray(
        shape=(new_shm.size,),
        dtype=dtype,
        buffer=new_shm.buf
    )
    return new_shm, new_view


@with_logger
class SharedMemoryRingBuffer:
    def __init__(self, buffer_count: int = 4):
        self.buffer_count = buffer_count

        # Internal state tracking
        self.buffers: List[Optional[shared_memory.SharedMemory]] = [
                                                                       None] * buffer_count
        self.names: List[Optional[str]] = [None] * buffer_count
        self.numpy_views: List[Optional[np.ndarray]] = [None] * buffer_count

        self.cursor = multiprocessing.Value('i', 0)

    def alloc_buffer(self, required_size: int) -> (
            Tuple)[str, np.ndarray]:
        """
        Returns a name and a shared_memory object.
        Guarantees the buffer is at least `required_size`.
        """
        # with self._lock:
        idx = self.cursor.value % self.buffer_count
        self.cursor.value += self.cursor.value

        existing_shm = self.buffers[idx]

        # 1. Evaluate Existing Buffer
        if existing_shm is not None:
            if existing_shm.size >= required_size:
                # Happy path: reuse existing
                return self.names[idx], self.numpy_views[idx]

            # Too small: Clean up before re-allocating
            self._logger.debug(
                f"Resizing Ring Slot {idx}: {existing_shm.size} < {required_size}")
            self._release_slot(idx)

        # 2. Allocate New Buffer (if slot is empty or was just cleared)
        self._allocate_slot(idx, required_size)

        return self.names[idx], self.numpy_views[idx]

    def _release_slot(self, idx: int) -> None:
        """
        Safely closes, unlinks, and clears references for a specific slot.
        Assumes lock is held by caller.
        """
        shm = self.buffers[idx]
        if shm is None:
            return

        # Critical: Remove NumPy view reference first to allow GC
        self.numpy_views[idx] = None
        name = shm.name

        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            # Common race condition: already unlinked. Safe to ignore.
            pass
        except Exception as e:
            self._logger.warning(f"Error freeing SHM slot {idx} ({name}): {e}")
        finally:
            # Ensure internal state is reset regardless of OS errors
            self.buffers[idx] = None
            self.names[idx] = None

    def _allocate_slot(self, idx: int, required_size: int) -> None:
        """
        Allocates a new SHM and NumPy view for a specific slot.
        Assumes lock is held by caller.
        """
        try:
            # Allocate 1.5x to prevent jittery resizing
            alloc_size = int(required_size * 1.5)
            new_shm, new_view = alloc_shm_buffer(alloc_size)

            # Update state
            self.buffers[idx] = new_shm
            self.names[idx] = new_shm.name
            self.numpy_views[idx] = new_view

            self._logger.debug(
                f"Allocated Ring Slot {idx}: {new_shm.name} ({alloc_size} bytes)")

        except Exception as e:
            self._logger.error(f"Failed to allocate SHM for slot {idx}: {e}")
            # Rollback: Try to clean up if SHM was created but view failed
            if 'new_shm' in locals():
                try:
                    new_shm.close()
                    new_shm.unlink()
                except:
                    pass
            raise

    def cleanup(self):
        """Destroys all managed buffers."""
        self._logger.info("Cleaning up SharedMemoryRingBuffer resources")
        # with self._lock:
        for idx in range(self.buffer_count):
            self._release_slot(idx)

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass


def cleanup_shm_cache(shm_cache: Dict[str, shared_memory.SharedMemory],
                      unlink: bool = False,
                      raise_on_error: bool = False):
    """
    Cleans up a dictionary of SHM objects.
    """
    errors = []
    # List conversion prevents RuntimeError if dict changes during iteration
    for name, shm in list(shm_cache.items()):
        try:
            shm.close()
            if unlink:
                shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            if raise_on_error:
                raise
            errors.append(f"{name}: {e}")

    shm_cache.clear()

    if errors:
        # Depending on app policy, might want to log these errors
        pass
