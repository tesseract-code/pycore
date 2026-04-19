import asyncio
import hashlib
import hmac
import logging
import pickle
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import Executor
from pathlib import Path
from typing import Any, Callable, Self

logger = logging.getLogger(__name__)

# HMAC key for signing pickled state. Override via environment variable or
# dependency injection in production; never hard-code a real secret here.
_DEFAULT_HMAC_KEY = b"dead-beef"
_HMAC_DIGEST = "sha256"
# Derive _HMAC_SIZE from the digest algorithm
_HMAC_SIZE: int = hashlib.new(_HMAC_DIGEST).digest_size


def is_picklable(obj: Any) -> bool:
    """
    Check if an object can be pickled.

    Args:
        obj: Object to check

    Returns:
        True if picklable, False otherwise
    """
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError, AttributeError, RecursionError):
        return False


def is_func_picklable(func: Callable, *args, **kwargs) -> bool:
    """
    Check if a function and its arguments are all picklable.

    Args:
        func: Function to check
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        True if all are picklable, False otherwise
    """
    return (
        is_picklable(func)
        and all(is_picklable(a) for a in args)
        and all(is_picklable(v) for v in kwargs.values())
    )


def _sign(data: bytes, key: bytes) -> bytes:
    """Return HMAC-SHA256 signature of *data*."""
    return hmac.new(key, data, _HMAC_DIGEST).digest()


def _pack(data: bytes, key: bytes) -> bytes:
    """Prepend an HMAC signature to *data*."""
    return _sign(data, key) + data


def _unpack(blob: bytes, key: bytes) -> bytes:
    """
    Verify the HMAC signature and return the payload.

    Raises:
        ValueError: If the signature is missing or does not match.
    """
    if len(blob) < _HMAC_SIZE:
        raise ValueError("Data too short to contain a valid signature.")
    sig, data = blob[:_HMAC_SIZE], blob[_HMAC_SIZE:]
    expected = _sign(data, key)
    if not hmac.compare_digest(sig, expected):
        raise ValueError("HMAC verification failed: data may be corrupt or tampered.")
    return data


class AutoPickle(ABC):
    """
    Base class for automatic serialisation/deserialisation of object state.

    Subclasses must implement :meth:`on_state_restored` (or leave it as a
    no-op) and may declare :attr:`_pickle_exclude` as a frozenset of
    attribute names that should be omitted from serialised state.

    Saved files are HMAC-signed.  Pass a *hmac_key* that is consistent
    across save and load calls; the default key is intentionally insecure
    and must be replaced in production.
    """

    # Subclasses override this with a frozenset of names to exclude.
    # Using frozenset avoids the mutable shared class-attribute bug.
    _pickle_exclude: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Ensure each subclass owns its own frozenset rather than inheriting
        # a mutable reference that could be shared across sibling classes.
        if "_pickle_exclude" not in cls.__dict__:
            cls._pickle_exclude = frozenset()

    @abstractmethod
    def on_state_restored(self) -> None:
        """
        Hook called after state restoration.

        Subclasses must implement this explicitly to avoid silently operating
        on a half-initialised object. A no-op implementation is fine:

            def on_state_restored(self) -> None:
                pass
        """

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def _should_exclude_from_pickle(self, key: str, value: Any) -> bool:
        """
        Return True if *key*/*value* should be omitted from serialised state.

        Excludes:
        - Names listed in ``_pickle_exclude``
        - Callable values (bound methods, lambdas)
        - asyncio tasks/futures and weakrefs (inherently non-serialisable)

        Note: leading-underscore names are NOT automatically excluded so
        that subclasses can serialise private state they own.
        """
        return (
            key in self._pickle_exclude
            or callable(value)
            or isinstance(value, (asyncio.Task, asyncio.Future, weakref.ref))
        )

    @property
    def pickle_state(self) -> dict[str, Any]:
        """
        Return a dictionary of the serialisable instance state.

        Attributes that are not picklable are skipped with a warning rather
        than raising, so that a single bad attribute does not block saving.
        """
        state: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if self._should_exclude_from_pickle(key, value):
                continue
            try:
                pickle.dumps(value)
                state[key] = value
            except (pickle.PicklingError, TypeError, AttributeError, RecursionError):
                logger.warning(
                    "Skipping unpicklable attribute '%s' on %s.",
                    key,
                    type(self).__name__,
                )
        return state

    # ------------------------------------------------------------------
    # Async persistence
    # ------------------------------------------------------------------

    async def save_to_disk(
        self,
        filepath: str | Path,
        *,
        hmac_key: bytes = _DEFAULT_HMAC_KEY,
        executor: Executor | None = None,
        mkdir: bool = True,
    ) -> None:
        """
        Asynchronously save signed, pickled state to *filepath*.

        Args:
            filepath:  Destination path.
            hmac_key:  Secret used to sign the payload.
            executor:  Thread-pool executor for the blocking I/O call.
                       Defaults to the running loop's default executor.
            mkdir:     If True, create parent directories as needed.
        """
        filepath = Path(filepath)
        if mkdir:
            filepath.parent.mkdir(parents=True, exist_ok=True)

        blob = _pack(pickle.dumps(self.pickle_state), hmac_key)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, filepath.write_bytes, blob)

    async def load_from_disk(
        self,
        filepath: str | Path,
        *,
        hmac_key: bytes = _DEFAULT_HMAC_KEY,
        executor: Executor | None = None,
    ) -> None:
        """
        Asynchronously load and verify state from *filepath*.

        Args:
            filepath:  Source path.
            hmac_key:  Secret used to verify the payload signature.
            executor:  Thread-pool executor for the blocking I/O call.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
            ValueError:        If the HMAC signature is invalid.
        """
        filepath = Path(filepath)
        # FIX 2: Replace the TOCTOU-prone exists()+read_bytes() pattern with a
        # single read_bytes() call inside try/except.  The original code checked
        # filepath.exists() and then called filepath.read_bytes() as two
        # separate operations; another process could delete the file in the
        # window between them, causing an unhandled OSError instead of the
        # documented FileNotFoundError.
        loop = asyncio.get_running_loop()
        try:
            blob = await loop.run_in_executor(executor, filepath.read_bytes)
        except FileNotFoundError:
            raise FileNotFoundError(f"State file not found: {filepath}")

        data = _unpack(blob, hmac_key)
        state = pickle.loads(data)
        self.restore_state(state)

    # ------------------------------------------------------------------
    # State restoration
    # ------------------------------------------------------------------

    def restore_state(self, state: dict[str, Any]) -> None:
        """
        Restore instance state from a deserialised dictionary onto an already
        initialised object.

        Only keys that correspond to attributes already present in
        ``__dict__`` are applied, preventing a crafted state file from
        injecting arbitrary new attributes.  Use this when loading onto a
        live object (e.g. ``load_from_disk``).

        Args:
            state: Dictionary of state to restore.
        """
        known = set(self.__dict__)
        for key, value in state.items():
            if key in known:
                setattr(self, key, value)
            else:
                logger.warning(
                    "Ignoring unknown attribute '%s' during state restore on %s.",
                    key,
                    type(self).__name__,
                )
        self.on_state_restored()

    def _restore_state_from_new(self, state: dict[str, Any]) -> None:
        """
        Restore state onto an uninitialised object created via ``__new__``.

        ``__init__`` has not run so ``__dict__`` is empty — the known-keys
        guard in :meth:`restore_state` would block every attribute.  This
        method sets all keys unconditionally and is intentionally private;
        only :meth:`from_bytes` should call it.

        Args:
            state: Dictionary of state to restore.
        """
        for key, value in state.items():
            setattr(self, key, value)
        self.on_state_restored()

    # ------------------------------------------------------------------
    # Byte-level helpers
    # ------------------------------------------------------------------

    def to_bytes(self, *, hmac_key: bytes = _DEFAULT_HMAC_KEY) -> bytes:
        """Serialise and sign the current state, returning raw bytes."""
        return _pack(pickle.dumps(self.pickle_state), hmac_key)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        hmac_key: bytes = _DEFAULT_HMAC_KEY,
    ) -> Self:
        """
        Create an instance from signed, serialised bytes.

        Note: ``__init__`` is intentionally bypassed.  Subclasses that
        require initialisation before ``on_state_restored`` runs should
        override ``on_state_restored`` accordingly.

        Args:
            data:     Bytes produced by :meth:`to_bytes`.
            hmac_key: Secret used to verify the signature.

        Returns:
            A new instance of the calling class with state restored.

        Raises:
            ValueError: If the HMAC signature is invalid.
        """
        instance = cls.__new__(cls)
        payload = _unpack(data, hmac_key)
        state = pickle.loads(payload)
        instance._restore_state_from_new(state)
        return instance