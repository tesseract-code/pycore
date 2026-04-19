"""
bidict.py
=========
A bidirectional dictionary with mutual-exclusivity enforcement.

Overview
--------
`BidirectionalDict` extends `collections.UserDict` so that
every forward mapping ``key → value`` is mirrored by a reverse mapping
``value → key``.  Both directions are kept consistent on every mutation.

Mutual exclusivity
------------------
Keys and values share a single *combined namespace*: a token may appear as
a key **or** as a value, but never as both simultaneously, and never in more
than one role at a time.  Attempting to insert a pair that would violate this
constraint raises `ValueError` immediately, before any state is changed.

Lookup behaviour
----------------
__getitem__` (``bd[token]``) first searches the forward mapping, then
falls back to the reverse mapping.  This means ``bd['a']`` and ``bd[1]`` both
work on ``BidirectionalDict({'a': 1})``.  Callers that need to be explicit
about which direction they are searching should use `get_value` or
`get_key` instead.

 Warning:

    Because `__getitem__` and `__contains__` cover *both*
    directions while `keys`, `values`, `items`, and
    iteration cover only the *forward* mapping, some standard dict invariants
    do not hold::

        bd = BidirectionalDict({'a': 1})
        len(bd)          # 1
        list(bd.keys())  # ['a']
        1 in bd          # True   ← value found via reverse lookup
        bd[1]            # 'a'    ← reverse lookup

    This is intentional, but callers should be aware of it.

Thread safety
-------------
`BidirectionalDict` is **not** thread-safe.  Callers that share an
instance across threads must provide their own synchronisation.

Typical usage
-------------
::

    from bidict import BidirectionalDict

    bd = BidirectionalDict({'left': 'right', 'up': 'down'})

    bd['left']           # 'right'  — forward lookup
    bd['right']          # 'left'   — reverse lookup
    bd.get_key('down')   # 'up'     — explicit reverse lookup
    bd.get_value('up')   # 'down'   — explicit forward lookup

    bd['diagonal'] = 'diagonal'   # ValueError: key == value violates exclusivity
"""

from __future__ import annotations

from collections import UserDict
from typing import Any, Iterable


__all__ = ["BidirectionalDict"]

# Sentinel used as a default in _validate_pair to distinguish "no argument
# supplied" from any real key value, including None.
_SENTINEL = object()


class BidirectionalDict(UserDict):
    """
    A bidirectional dictionary that allows lookup by both key and value.

    Every insertion maintains two invariants:

    1. **Bidirectionality** — for every ``key → value`` in the forward
       mapping there is a corresponding ``value → key`` in the reverse
       mapping, and vice versa.

    2. **Mutual exclusivity** — the sets of keys and values are disjoint.
       A token cannot occupy both roles simultaneously.

    Parameters
    ----------
    data:
        Optional initial data.  Accepts a :class:`dict`, another
        :class:`BidirectionalDict`, or an iterable of ``(key, value)``
        pairs.  All normal :class:`dict` keyword-argument initialisation
        is also supported.

    Raises
    ------
    TypeError
        If more than one positional argument is supplied.
    ValueError
        If the initial data violates mutual exclusivity.
    """

    # Explicitly suppress hashing.  __eq__ is defined below, and mutable
    # containers should not be hashable.
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        data: dict | BidirectionalDict | Iterable[tuple[Any, Any]] | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        # Initialise the reverse mapping before calling super().__init__ so
        # that __setitem__ (called during update) can always write to it.
        self._reverse: dict[Any, Any] = {}

        # super().__init__() without arguments initialises self.data = {}
        # without triggering any update calls, giving us a clean slate.
        super().__init__()

        if data is not None:
            self.update(data)

        if kwargs:
            self.update(kwargs)

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _validate_pair(
        self,
        key: Any,
        value: Any,
        *,
        replacing_key: Any = _SENTINEL,
    ) -> None:
        """
        Assert that inserting ``key → value`` preserves mutual exclusivity.

        Called by :meth:`__setitem__` *after* the old reverse entry for
        ``key`` has already been removed, so self._reverse no longer
        contains the stale mapping for ``key``.

        Parameters
        ----------
        key:
            The key being inserted.
        value:
            The value being inserted.
        replacing_key:
            When updating an existing key, pass the key being replaced so
            that its current value is not treated as a conflict.  Defaults
            to a private sentinel that never equals any real key.

        Raises
        ------
        ValueError
            On any mutual-exclusivity violation.
        """
        # A token cannot be its own reverse — that would mean key == value,
        # which creates an ambiguous round-trip.
        if key == value:
            raise ValueError(
                f"Key and value must be distinct; got key={key!r} == value={value!r}."
            )

        # The proposed *value* must not already be a *key* in the forward
        # mapping (unless it is the key we are currently replacing, in which
        # case its old entry has already been cleaned up).
        if value in self.data and value != replacing_key:
            raise ValueError(
                f"Cannot use {value!r} as a value: it already exists as a key."
            )

        # The proposed *key* must not already be a *value* in the reverse
        # mapping.  (The old reverse entry for this key was removed before
        # this call, so a hit here means a *different* key maps to our key.)
        if key in self._reverse:
            raise ValueError(
                f"Cannot use {key!r} as a key: it already exists as a value "
                f"(mapped from key {self._reverse[key]!r})."
            )

        # The proposed *value* must not already be a *value* for a different
        # key.  (Same-key re-assignment is fine; __setitem__ removed the old
        # reverse entry before calling us.)
        if value in self._reverse:
            raise ValueError(
                f"Cannot use {value!r} as a value: it is already the value "
                f"for key {self._reverse[value]!r}."
            )

    # ------------------------------------------------------------------
    # Core mapping protocol
    # ------------------------------------------------------------------

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Insert or update the mapping ``key → value``.

        If *key* already exists its old reverse entry is removed before
        validation so that updating ``bd['a'] = 2`` on ``{'a': 1}`` does
        not falsely report ``1`` as a conflict.

        Raises
        ------
        ValueError
            If the new pair would violate mutual exclusivity.
        """
        # Remove the stale reverse entry for this key so that validation
        # does not see the old value as a conflict.
        if key in self.data:
            del self._reverse[self.data[key]]

        self._validate_pair(key, value, replacing_key=key)

        self.data[key] = value
        self._reverse[value] = key

    def __delitem__(self, key: Any) -> None:
        """
        Remove ``key`` and its corresponding reverse entry.

        Raises
        ------
        KeyError
            If *key* does not exist in the forward mapping.
        """
        if key not in self.data:
            raise KeyError(key)

        del self._reverse[self.data[key]]
        del self.data[key]

    def __getitem__(self, key: Any) -> Any:
        """
        Return the value associated with *key*.

        Searches the forward mapping first; if *key* is not found there,
        falls back to the reverse mapping so that values can be used as
        lookup tokens.

        Raises
        ------
        KeyError
            If *key* is not found in either mapping.
        """
        if key in self.data:
            return self.data[key]
        if key in self._reverse:
            return self._reverse[key]
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        """
        Return ``True`` if *key* exists in the forward **or** reverse mapping.

        .. note::

            This covers both directions, while :meth:`keys` and iteration
            cover only the forward mapping.  See the module docstring for
            the implications.
        """
        return key in self.data or key in self._reverse

    # ------------------------------------------------------------------
    # Explicit directional accessors
    # ------------------------------------------------------------------

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Return the mapping for *key* (checked in both directions), or
        *default* if not found.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def get_key(self, value: Any, default: Any = None) -> Any:
        """
        Return the key that maps to *value*, or *default* if not found.

        This is an explicit reverse lookup and never searches the forward
        mapping, making it unambiguous regardless of the overlap rules.
        """
        return self._reverse.get(value, default)

    def get_value(self, key: Any, default: Any = None) -> Any:
        """
        Return the value for *key*, or *default* if not found.

        This is an explicit forward lookup and never searches the reverse
        mapping.
        """
        return self.data.get(key, default)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def update(  # type: ignore[override]
        self,
        other: dict | BidirectionalDict | Iterable[tuple[Any, Any]] | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        """
        Update the dictionary with key-value pairs from *other* and *kwargs*.

        Accepts a :class:`dict`, another :class:`BidirectionalDict`, or any
        iterable of ``(key, value)`` pairs.  Each pair is inserted via
        :meth:`__setitem__`, so mutual-exclusivity is enforced on every item.

        Raises
        ------
        ValueError
            If any incoming pair would violate mutual exclusivity.
        """
        if other is not None:
            pairs = other.items() if hasattr(other, "items") else other
            for key, value in pairs:
                self[key] = value

        for key, value in kwargs.items():
            self[key] = value

    def clear(self) -> None:
        """Remove all entries from both the forward and reverse mappings."""
        self.data.clear()
        self._reverse.clear()

    def pop(self, key: Any, *args: Any) -> Any:
        """
        Remove *key* and return its value.

        Parameters
        ----------
        key:
            The key to remove.
        default:
            Returned if *key* is absent.  If omitted and *key* is absent,
            :exc:`KeyError` is raised.

        Raises
        ------
        TypeError
            If more than one default argument is provided.
        KeyError
            If *key* is absent and no default was supplied.
        """
        if len(args) > 1:
            raise TypeError(
                f"pop expected at most 2 arguments, got {1 + len(args)}"
            )
        try:
            value = self.data[key]
        except KeyError:
            if args:
                return args[0]
            raise

        # Delegate to __delitem__ so reverse cleanup is always in one place.
        del self[key]
        return value

    def popitem(self) -> tuple[Any, Any]:
        """
        Remove and return an arbitrary ``(key, value)`` pair.

        Raises
        ------
        KeyError
            If the dictionary is empty.
        """
        if not self.data:
            raise KeyError("popitem(): dictionary is empty")

        # Peek at the key, then use __delitem__ so reverse cleanup stays in
        # one place rather than being duplicated here.
        key = next(iter(self.data))
        value = self.data[key]
        del self[key]
        return key, value

    # ------------------------------------------------------------------
    # View methods (forward mapping only)
    # ------------------------------------------------------------------

    def keys(self):
        """Return a view of the forward keys."""
        return self.data.keys()

    def values(self):
        """Return a view of the forward values."""
        return self.data.values()

    def items(self):
        """Return a view of the forward ``(key, value)`` pairs."""
        return self.data.items()

    def reverse_items(self):
        """Return a view of the reverse ``(value, key)`` pairs."""
        return self._reverse.items()

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> BidirectionalDict:
        """
        Return a shallow copy.

        The copy is independent: mutations to the original do not affect the
        copy and vice versa.  Values themselves are *not* deep-copied.
        """
        new = BidirectionalDict()
        # Directly assign the internal dicts to avoid re-validating every
        # pair; the source is already guaranteed to be consistent.
        new.data = self.data.copy()
        new._reverse = self._reverse.copy()
        return new

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self.data)!r})"

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison.

        Two :class:`BidirectionalDict` instances are equal when both their
        forward *and* reverse mappings are equal.  A plain :class:`dict` is
        considered equal when it matches the forward mapping only.
        """
        if isinstance(other, BidirectionalDict):
            return self.data == other.data and self._reverse == other._reverse
        if isinstance(other, dict):
            return self.data == other
        return NotImplemented