"""
Pytest test suite for autopickle.py
"""

import hashlib
import weakref

import pytest

from pycore.autopickle import (
    AutoPickle,
    _DEFAULT_HMAC_KEY,
    _HMAC_SIZE,
    _pack,
    _sign,
    _unpack,
    is_func_picklable,
    is_picklable,
)


# ---------------------------------------------------------------------------
# Concrete AutoPickle subclasses used across tests
# ---------------------------------------------------------------------------


class Simple(AutoPickle):
    """Minimal subclass with two plain attributes."""

    def __init__(self, x: int = 0, label: str = "default"):
        self.x = x
        self.label = label
        self.restored_count = 0

    def on_state_restored(self) -> None:
        self.restored_count += 1


class WithExcludes(AutoPickle):
    """Subclass that excludes a sensitive field from serialisation."""

    _pickle_exclude = frozenset({"secret"})

    def __init__(self):
        self.public = "visible"
        self.secret = "top-secret"

    def on_state_restored(self) -> None:
        pass


class WithPrivate(AutoPickle):
    """Subclass that serialises a private (leading-underscore) attribute."""

    def __init__(self, value: int):
        self._internal = value

    def on_state_restored(self) -> None:
        pass


class WithCallable(AutoPickle):
    """Subclass that holds a callable attribute (should be excluded)."""

    def __init__(self):
        self.data = 42
        self.handler = lambda: None  # non-picklable callable

    def on_state_restored(self) -> None:
        pass


class CallbackTracker(AutoPickle):
    """Records that on_state_restored was called exactly once per restore."""

    def __init__(self):
        self.value = 99
        self.hooks: list[str] = []

    def on_state_restored(self) -> None:
        self.hooks.append("restored")


class WithWeakref(AutoPickle):
    """Subclass holding a weakref (must be excluded from serialisation)."""

    class _Target:
        """Minimal weakly-referenceable target."""

    def __init__(self, target: object):
        self.data = "present"
        self.ref = weakref.ref(target)

    def on_state_restored(self) -> None:
        pass


# ---------------------------------------------------------------------------
# is_picklable
# ---------------------------------------------------------------------------


class TestIsPicklable:
    def test_primitive_types_are_picklable(self):
        for obj in (
        1, 3.14, "hello", b"bytes", True, None, (1, 2), [3, 4], {"k": 5}):
            assert is_picklable(obj) is True

    def test_lambda_is_not_picklable(self):
        assert is_picklable(lambda: None) is False

    def test_module_level_function_is_picklable(self):
        # Functions defined at module level can be pickled by reference.
        import math

        assert is_picklable(math.sqrt) is True

    def test_nested_function_is_not_picklable(self):
        def inner():
            pass

        assert is_picklable(inner) is False

    def test_class_instance_is_picklable(self):
        # Must use a module-level class; locally-defined classes cannot be
        # pickled because pickle resolves them by qualified name.
        obj = Simple(x=1)
        assert is_picklable(obj) is True

    def test_unpicklable_object(self):
        # threading.Lock cannot be pickled
        import threading

        assert is_picklable(threading.Lock()) is False


# ---------------------------------------------------------------------------
# is_func_picklable
# ---------------------------------------------------------------------------


class TestIsFuncPicklable:
    def test_all_picklable(self):
        import math

        assert is_func_picklable(math.sqrt, 4.0) is True

    def test_func_not_picklable(self):
        assert is_func_picklable(lambda: None) is False

    def test_args_not_picklable(self):
        import math
        import threading

        assert is_func_picklable(math.sqrt, threading.Lock()) is False

    def test_kwargs_not_picklable(self):
        import math
        import threading

        assert is_func_picklable(math.sqrt, lock=threading.Lock()) is False

    def test_all_picklable_with_kwargs(self):
        import math

        assert is_func_picklable(math.pow, 2.0, exp=3.0) is True


# ---------------------------------------------------------------------------
# _sign / _pack / _unpack
# ---------------------------------------------------------------------------


class TestHMAC:
    def test_sign_returns_correct_size(self):
        sig = _sign(b"data", _DEFAULT_HMAC_KEY)
        assert len(sig) == _HMAC_SIZE

    def test_sign_is_deterministic(self):
        assert _sign(b"data", b"key") == _sign(b"data", b"key")

    def test_sign_differs_for_different_keys(self):
        assert _sign(b"data", b"key1") != _sign(b"data", b"key2")

    def test_sign_differs_for_different_data(self):
        assert _sign(b"aaa", b"key") != _sign(b"bbb", b"key")

    def test_pack_prepends_signature(self):
        data = b"payload"
        blob = _pack(data, _DEFAULT_HMAC_KEY)
        assert len(blob) == _HMAC_SIZE + len(data)
        assert blob[_HMAC_SIZE:] == data

    def test_unpack_returns_original_data(self):
        data = b"hello world"
        blob = _pack(data, _DEFAULT_HMAC_KEY)
        assert _unpack(blob, _DEFAULT_HMAC_KEY) == data

    def test_unpack_raises_on_wrong_key(self):
        blob = _pack(b"data", b"correct-key")
        with pytest.raises(ValueError, match="HMAC verification failed"):
            _unpack(blob, b"wrong-key")

    def test_unpack_raises_on_tampered_payload(self):
        blob = bytearray(_pack(b"original", _DEFAULT_HMAC_KEY))
        blob[-1] ^= 0xFF  # flip last byte of payload
        with pytest.raises(ValueError, match="HMAC verification failed"):
            _unpack(bytes(blob), _DEFAULT_HMAC_KEY)

    def test_unpack_raises_on_tampered_signature(self):
        blob = bytearray(_pack(b"data", _DEFAULT_HMAC_KEY))
        blob[0] ^= 0xFF  # flip first byte of signature
        with pytest.raises(ValueError, match="HMAC verification failed"):
            _unpack(bytes(blob), _DEFAULT_HMAC_KEY)

    def test_unpack_raises_on_too_short_blob(self):
        with pytest.raises(ValueError, match="Data too short"):
            _unpack(b"\x00" * (_HMAC_SIZE - 1), _DEFAULT_HMAC_KEY)

    def test_unpack_exact_minimum_length(self):
        # A blob that is exactly _HMAC_SIZE bytes has an empty payload.
        blob = _pack(b"", _DEFAULT_HMAC_KEY)
        assert len(blob) == _HMAC_SIZE
        assert _unpack(blob, _DEFAULT_HMAC_KEY) == b""

    def test_hmac_size_matches_digest(self):
        # Ensures _HMAC_SIZE is derived from the algorithm, not hard-coded.
        expected = hashlib.new("sha256").digest_size
        assert _HMAC_SIZE == expected


# ---------------------------------------------------------------------------
# AutoPickle.pickle_state
# ---------------------------------------------------------------------------


class TestPickleState:
    def test_plain_attributes_are_included(self):
        obj = Simple(x=7, label="hi")
        state = obj.pickle_state
        assert state["x"] == 7
        assert state["label"] == "hi"

    def test_callable_is_excluded(self):
        obj = WithCallable()
        assert "handler" not in obj.pickle_state
        assert obj.pickle_state["data"] == 42

    def test_excluded_field_is_absent(self):
        obj = WithExcludes()
        state = obj.pickle_state
        assert "public" in state
        assert "secret" not in state

    def test_private_attr_is_included(self):
        obj = WithPrivate(value=55)
        assert obj.pickle_state["_internal"] == 55

    def test_weakref_is_excluded(self):
        target = WithWeakref._Target()
        obj = WithWeakref(target)
        state = obj.pickle_state
        assert "ref" not in state
        assert "data" in state

    def test_unpicklable_attr_is_skipped_with_warning(self, caplog):
        import threading

        obj = Simple(x=1)
        obj.bad = threading.Lock()  # dynamically attach an unpicklable attr
        import logging

        with caplog.at_level(logging.WARNING):
            state = obj.pickle_state
        assert "bad" not in state
        assert any("bad" in msg for msg in caplog.messages)

    def test_restored_count_not_in_pickle_state_initially(self):
        # restored_count starts at 0 and IS serialisable; it should be present.
        obj = Simple(x=3)
        assert "restored_count" in obj.pickle_state


# ---------------------------------------------------------------------------
# AutoPickle.to_bytes / from_bytes
# ---------------------------------------------------------------------------


class TestToFromBytes:
    def test_round_trip(self):
        original = Simple(x=42, label="round-trip")
        blob = original.to_bytes()
        restored = Simple.from_bytes(blob)
        assert restored.x == 42
        assert restored.label == "round-trip"

    def test_on_state_restored_called_once(self):
        obj = CallbackTracker()
        assert obj.hooks == []
        restored = CallbackTracker.from_bytes(obj.to_bytes())
        assert restored.hooks == ["restored"]

    def test_wrong_key_raises(self):
        blob = Simple(x=1).to_bytes(hmac_key=b"secret")
        with pytest.raises(ValueError, match="HMAC verification failed"):
            Simple.from_bytes(blob, hmac_key=b"wrong")

    def test_custom_key_round_trip(self):
        key = b"my-production-key"
        original = Simple(x=99)
        blob = original.to_bytes(hmac_key=key)
        restored = Simple.from_bytes(blob, hmac_key=key)
        assert restored.x == 99

    def test_excluded_fields_absent_after_round_trip(self):
        obj = WithExcludes()
        restored = WithExcludes.from_bytes(obj.to_bytes())
        # 'secret' was not serialised so it won't exist on the restored object
        assert not hasattr(restored, "secret")
        assert restored.public == "visible"

    def test_private_attr_survives_round_trip(self):
        obj = WithPrivate(value=7)
        restored = WithPrivate.from_bytes(obj.to_bytes())
        assert restored._internal == 7

    def test_from_bytes_bypasses_init(self):
        # from_bytes must not call __init__; if it did, restored_count
        # would start at 0 and then be incremented to 1 by __init__ *plus*
        # on_state_restored, giving 2. With correct behaviour it starts at
        # whatever was serialised (0) and is incremented only by
        # on_state_restored → 1.
        obj = Simple(x=5)
        assert obj.restored_count == 0
        restored = Simple.from_bytes(obj.to_bytes())
        assert restored.restored_count == 1  # only from on_state_restored


# ---------------------------------------------------------------------------
# AutoPickle.restore_state
# ---------------------------------------------------------------------------


class TestRestoreState:
    def test_known_keys_are_applied(self):
        target = Simple(x=0, label="old")
        target.restore_state({"x": 99, "label": "new"})
        assert target.x == 99
        assert target.label == "new"

    def test_unknown_keys_are_ignored_with_warning(self, caplog):
        target = Simple(x=1)
        import logging

        with caplog.at_level(logging.WARNING):
            target.restore_state({"x": 2, "injected_attr": "evil"})
        assert not hasattr(target, "injected_attr")
        assert any("injected_attr" in msg for msg in caplog.messages)

    def test_on_state_restored_called(self):
        obj = CallbackTracker()
        obj.restore_state({"value": 5})
        assert "restored" in obj.hooks

    def test_partial_restore_leaves_unmentioned_attrs_intact(self):
        obj = Simple(x=1, label="keep")
        obj.restore_state({"x": 2})  # label not in state dict
        assert obj.x == 2
        assert obj.label == "keep"


# ---------------------------------------------------------------------------
# AutoPickle.save_to_disk / load_from_disk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDiskPersistence:
    async def test_round_trip(self, tmp_path):
        path = tmp_path / "state.pkl"
        original = Simple(x=7, label="disk")
        await original.save_to_disk(path)
        target = Simple()
        await target.load_from_disk(path)
        assert target.x == 7
        assert target.label == "disk"

    async def test_mkdir_creates_parents(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "state.pkl"
        await Simple(x=1).save_to_disk(nested, mkdir=True)
        assert nested.exists()

    async def test_mkdir_false_does_not_create_parents(self, tmp_path):
        nested = tmp_path / "nonexistent" / "state.pkl"
        with pytest.raises(FileNotFoundError):
            await Simple(x=1).save_to_disk(nested, mkdir=False)

    async def test_load_missing_file_raises(self, tmp_path):
        target = Simple()
        with pytest.raises(FileNotFoundError, match="State file not found"):
            await target.load_from_disk(tmp_path / "missing.pkl")

    async def test_load_wrong_key_raises(self, tmp_path):
        path = tmp_path / "state.pkl"
        await Simple(x=3).save_to_disk(path, hmac_key=b"correct")
        target = Simple()
        with pytest.raises(ValueError, match="HMAC verification failed"):
            await target.load_from_disk(path, hmac_key=b"wrong")

    async def test_load_tampered_file_raises(self, tmp_path):
        path = tmp_path / "state.pkl"
        await Simple(x=3).save_to_disk(path)
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
        target = Simple()
        with pytest.raises(ValueError, match="HMAC verification failed"):
            await target.load_from_disk(path)

    async def test_on_state_restored_called_after_load(self, tmp_path):
        path = tmp_path / "state.pkl"
        obj = CallbackTracker()
        await obj.save_to_disk(path)
        target = CallbackTracker()
        await target.load_from_disk(path)
        assert target.hooks.count("restored") == 1

    async def test_custom_hmac_key_round_trip(self, tmp_path):
        key = b"custom-secret"
        path = tmp_path / "state.pkl"
        await Simple(x=55).save_to_disk(path, hmac_key=key)
        target = Simple()
        await target.load_from_disk(path, hmac_key=key)
        assert target.x == 55

    async def test_str_path_accepted(self, tmp_path):
        path = str(tmp_path / "state.pkl")
        await Simple(x=11).save_to_disk(path)
        target = Simple()
        await target.load_from_disk(path)
        assert target.x == 11


# ---------------------------------------------------------------------------
# __init_subclass__ isolation
# ---------------------------------------------------------------------------


class TestInitSubclass:
    def test_each_subclass_has_own_exclude_set(self):
        class A(AutoPickle):
            def on_state_restored(self):
                pass

        class B(AutoPickle):
            def on_state_restored(self):
                pass

        # Mutating A's set must not affect B's set
        A._pickle_exclude = frozenset({"a_only"})
        assert "a_only" not in B._pickle_exclude

    def test_declared_exclude_is_preserved(self):
        assert "secret" in WithExcludes._pickle_exclude

    def test_subclass_without_declaration_gets_empty_set(self):
        assert Simple._pickle_exclude == frozenset()

    def test_abstract_method_enforced(self):
        with pytest.raises(TypeError):
            class Bad(AutoPickle):
                pass  # missing on_state_restored

            Bad()  # instantiation must fail


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_state_round_trip(self):
        class Empty(AutoPickle):
            def on_state_restored(self):
                pass

        obj = Empty()
        restored = Empty.from_bytes(obj.to_bytes())
        assert isinstance(restored, Empty)

    def test_nested_data_structures(self):
        obj = Simple()
        obj.nested = {"list": [1, 2, 3], "dict": {"a": 1}}
        blob = obj.to_bytes()
        restored = Simple.from_bytes(blob)
        assert restored.nested == {"list": [1, 2, 3], "dict": {"a": 1}}

    def test_large_payload_round_trip(self):
        obj = Simple()
        obj.big = list(range(10_000))
        restored = Simple.from_bytes(obj.to_bytes())
        assert restored.big == list(range(10_000))

    def test_to_bytes_produces_bytes(self):
        assert isinstance(Simple(x=1).to_bytes(), bytes)
