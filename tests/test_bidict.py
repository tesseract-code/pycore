"""
test_bidict.py
==============
Pytest coverage for BidirectionalDict.

Run with:
    pytest test_bidict.py -v
"""

from __future__ import annotations

import pytest

from pycore.bidict import BidirectionalDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make() -> BidirectionalDict:
    """Return a small populated instance for use in multiple tests."""
    return BidirectionalDict({"a": 1, "b": 2, "c": 3})


# ===========================================================================
# Construction
# ===========================================================================

class TestConstruction:

    def test_empty(self):
        bd = BidirectionalDict()
        assert len(bd) == 0
        assert bd.data == {}
        assert bd._reverse == {}

    def test_from_dict(self):
        bd = BidirectionalDict({"a": 1, "b": 2})
        assert bd["a"] == 1
        assert bd["b"] == 2

    def test_from_list_of_pairs(self):
        bd = BidirectionalDict([("a", 1), ("b", 2)])
        assert bd["a"] == 1
        assert bd._reverse[1] == "a"

    def test_from_another_bidirectional_dict(self):
        src = _make()
        bd = BidirectionalDict(src)
        assert bd == src
        assert bd is not src

    def test_from_kwargs(self):
        bd = BidirectionalDict(a=1, b=2)
        assert bd["a"] == 1
        assert bd["b"] == 2

    def test_from_dict_and_kwargs_merged(self):
        bd = BidirectionalDict({"a": 1}, b=2)
        assert bd["a"] == 1
        assert bd["b"] == 2

    def test_too_many_positional_args_raises(self):
        with pytest.raises(TypeError):
            BidirectionalDict({"a": 1}, {"b": 2})

    def test_reverse_mapping_populated_at_construction(self):
        bd = BidirectionalDict({"a": 1, "b": 2})
        assert bd._reverse == {1: "a", 2: "b"}


# ===========================================================================
# Mutual exclusivity — construction
# ===========================================================================

class TestMutualExclusivityAtConstruction:

    def test_value_equals_existing_key_raises(self):
        """{'a': 1, 1: 2} — 1 is both a key and a value."""
        with pytest.raises(ValueError, match="already exists as a value"):
            BidirectionalDict({"a": 1, 1: 2})

    def test_key_equals_existing_value_raises(self):
        """{'a': 1, 'b': 'a'} — 'a' is both a key and proposed value."""
        with pytest.raises(ValueError):
            BidirectionalDict({"a": 1, "b": "a"})

    def test_key_equals_value_raises(self):
        """{'a': 'a'} — a token cannot map to itself."""
        with pytest.raises(ValueError, match="must be distinct"):
            BidirectionalDict({"a": "a"})

    def test_cycle_raises(self):
        """{'a': 'b', 'b': 'a'} — creates an ambiguous cycle."""
        with pytest.raises(ValueError):
            BidirectionalDict({"a": "b", "b": "a"})


# ===========================================================================
# Injectivity — construction
# ===========================================================================

class TestInjectivityAtConstruction:

    def test_duplicate_values_raises(self):
        """{'a': 1, 'b': 1} — two keys mapping to the same value."""
        with pytest.raises(ValueError, match="already the value"):
            BidirectionalDict({"a": 1, "b": 1})

    def test_unique_values_accepted(self):
        bd = BidirectionalDict({"a": 1, "b": 2, "c": 3})
        assert len(bd) == 3


# ===========================================================================
# __setitem__
# ===========================================================================

class TestSetItem:

    def test_insert_new_pair(self):
        bd = _make()
        bd["d"] = 4
        assert bd["d"] == 4
        assert bd._reverse[4] == "d"

    def test_update_existing_key_removes_old_reverse(self):
        bd = BidirectionalDict({"a": 1})
        bd["a"] = 99
        assert bd["a"] == 99
        assert bd._reverse[99] == "a"
        assert 1 not in bd._reverse

    def test_update_existing_key_to_already_used_value_raises(self):
        bd = BidirectionalDict({"a": 1, "b": 2})
        with pytest.raises(ValueError):
            bd["a"] = 2  # 2 is already the value for 'b'

    def test_duplicate_value_raises(self):
        bd = BidirectionalDict({"a": 1})
        with pytest.raises(ValueError, match="already the value"):
            bd["b"] = 1

    def test_value_already_a_key_raises(self):
        bd = BidirectionalDict({"a": 1})
        with pytest.raises(ValueError, match="already exists as a key"):
            bd["x"] = "a"

    def test_key_already_a_value_raises(self):
        bd = BidirectionalDict({"a": 1})
        with pytest.raises(ValueError, match="already exists as a value"):
            bd[1] = "z"

    def test_key_equals_value_raises(self):
        bd = BidirectionalDict()
        with pytest.raises(ValueError, match="must be distinct"):
            bd["x"] = "x"

    def test_state_unchanged_after_failed_insert(self):
        """A failed __setitem__ must not leave the dict in a partial state."""
        bd = BidirectionalDict({"a": 1})
        try:
            bd["b"] = 1
        except ValueError:
            pass
        assert dict(bd.data) == {"a": 1}
        assert dict(bd._reverse) == {1: "a"}


# ===========================================================================
# __delitem__
# ===========================================================================

class TestDelItem:

    def test_delete_removes_forward_and_reverse(self):
        bd = _make()
        del bd["a"]
        assert "a" not in bd.data
        assert 1 not in bd._reverse

    def test_delete_missing_key_raises(self):
        bd = _make()
        with pytest.raises(KeyError):
            del bd["z"]

    def test_delete_by_value_token_raises(self):
        """Values are not first-class keys in the forward map."""
        bd = _make()
        with pytest.raises(KeyError):
            del bd[1]


# ===========================================================================
# __getitem__
# ===========================================================================

class TestGetItem:

    def test_forward_lookup(self):
        bd = _make()
        assert bd["a"] == 1

    def test_reverse_lookup(self):
        """A value used as a key returns the original key."""
        bd = _make()
        assert bd[1] == "a"

    def test_missing_key_raises(self):
        bd = _make()
        with pytest.raises(KeyError):
            _ = bd["z"]

    def test_forward_takes_priority(self):
        """If a token is somehow both a forward key and a reverse token,
        the forward mapping wins.  (This should not arise in practice due
        to mutual exclusivity, but the priority is documented.)*"""
        # Construct a pathological state manually for this edge-case test.
        bd = BidirectionalDict()
        bd.data = {"x": "y"}
        bd._reverse = {"x": "z"}  # x is both a forward key and a reverse token
        assert bd["x"] == "y"  # forward wins


# ===========================================================================
# __contains__
# ===========================================================================

class TestContains:

    def test_forward_key_found(self):
        assert "a" in _make()

    def test_value_found_via_reverse(self):
        assert 1 in _make()

    def test_absent_token_not_found(self):
        assert "z" not in _make()

    def test_empty_dict(self):
        assert "a" not in BidirectionalDict()


# ===========================================================================
# get / get_key / get_value
# ===========================================================================

class TestGetMethods:

    def test_get_forward(self):
        assert _make().get("a") == 1

    def test_get_reverse(self):
        assert _make().get(1) == "a"

    def test_get_missing_returns_default(self):
        assert _make().get("z", "default") == "default"

    def test_get_missing_returns_none_by_default(self):
        assert _make().get("z") is None

    def test_get_key_returns_key_for_value(self):
        assert _make().get_key(1) == "a"

    def test_get_key_missing_returns_default(self):
        assert _make().get_key(99, "default") == "default"

    def test_get_key_does_not_search_forward(self):
        """get_key('a') must return None, not 1 — it only searches reverse."""
        assert _make().get_key("a") is None

    def test_get_value_returns_value_for_key(self):
        assert _make().get_value("a") == 1

    def test_get_value_missing_returns_default(self):
        assert _make().get_value("z", "default") == "default"

    def test_get_value_does_not_search_reverse(self):
        """get_value(1) must return None, not 'a' — it only searches forward."""
        assert _make().get_value(1) is None


# ===========================================================================
# update
# ===========================================================================

class TestUpdate:

    def test_update_from_dict(self):
        bd = _make()
        bd.update({"d": 4, "e": 5})
        assert bd["d"] == 4
        assert bd._reverse[5] == "e"

    def test_update_from_pairs(self):
        bd = BidirectionalDict()
        bd.update([("a", 1), ("b", 2)])
        assert bd["a"] == 1

    def test_update_from_kwargs(self):
        bd = BidirectionalDict()
        bd.update(x=10, y=20)
        assert bd["x"] == 10

    def test_update_violating_injectivity_raises(self):
        bd = BidirectionalDict({"a": 1})
        with pytest.raises(ValueError):
            bd.update({"b": 1})  # duplicate value

    def test_update_violating_mutual_exclusivity_raises(self):
        bd = BidirectionalDict({"a": 1})
        with pytest.raises(ValueError):
            bd.update({1: 99})  # 1 already exists as a value


# ===========================================================================
# clear
# ===========================================================================

class TestClear:

    def test_clear_empties_both_mappings(self):
        bd = _make()
        bd.clear()
        assert len(bd) == 0
        assert bd._reverse == {}

    def test_clear_allows_reuse(self):
        bd = _make()
        bd.clear()
        bd["x"] = 99
        assert bd["x"] == 99


# ===========================================================================
# pop
# ===========================================================================

class TestPop:

    def test_pop_returns_value(self):
        bd = _make()
        assert bd.pop("a") == 1

    def test_pop_removes_forward_and_reverse(self):
        bd = _make()
        bd.pop("a")
        assert "a" not in bd.data
        assert 1 not in bd._reverse

    def test_pop_missing_raises(self):
        with pytest.raises(KeyError):
            _make().pop("z")

    def test_pop_missing_with_default(self):
        assert _make().pop("z", "default") == "default"

    def test_pop_too_many_args_raises(self):
        with pytest.raises(TypeError):
            _make().pop("a", "x", "y")


# ===========================================================================
# popitem
# ===========================================================================

class TestPopItem:

    def test_popitem_returns_pair(self):
        bd = _make()
        key, value = bd.popitem()
        assert isinstance(key, str)
        assert isinstance(value, int)

    def test_popitem_removes_both_mappings(self):
        bd = BidirectionalDict({"a": 1})
        bd.popitem()
        assert len(bd) == 0
        assert bd._reverse == {}

    def test_popitem_empty_raises(self):
        with pytest.raises(KeyError, match="empty"):
            BidirectionalDict().popitem()

    def test_popitem_reduces_length(self):
        bd = _make()
        original_len = len(bd)
        bd.popitem()
        assert len(bd) == original_len - 1


# ===========================================================================
# copy
# ===========================================================================

class TestCopy:

    def test_copy_is_equal(self):
        bd = _make()
        assert bd.copy() == bd

    def test_copy_is_independent(self):
        bd = _make()
        cp = bd.copy()
        cp["d"] = 4
        assert "d" not in bd

    def test_copy_reverse_is_independent(self):
        bd = _make()
        cp = bd.copy()
        del cp["a"]
        assert 1 in bd._reverse

    def test_copy_is_bidirectional_dict(self):
        assert isinstance(_make().copy(), BidirectionalDict)


# ===========================================================================
# View methods
# ===========================================================================

class TestViewMethods:

    def test_keys_covers_forward_only(self):
        bd = _make()
        assert set(bd.keys()) == {"a", "b", "c"}
        assert 1 not in bd.keys()

    def test_values_covers_forward_only(self):
        bd = _make()
        assert set(bd.values()) == {1, 2, 3}
        assert "a" not in bd.values()

    def test_items_covers_forward_only(self):
        bd = _make()
        assert set(bd.items()) == {("a", 1), ("b", 2), ("c", 3)}

    def test_reverse_items(self):
        bd = _make()
        assert set(bd.reverse_items()) == {(1, "a"), (2, "b"), (3, "c")}

    def test_len_counts_forward_pairs_only(self):
        bd = _make()
        assert len(bd) == 3  # not 6, even though 6 tokens are reachable


# ===========================================================================
# __eq__ and __hash__
# ===========================================================================

class TestEquality:

    def test_equal_to_itself(self):
        bd = _make()
        assert bd == bd

    def test_equal_to_identical_bidict(self):
        assert _make() == _make()

    def test_not_equal_to_different_bidict(self):
        assert _make() != BidirectionalDict({"x": 10})

    def test_equal_to_plain_dict_with_same_forward(self):
        assert _make() == {"a": 1, "b": 2, "c": 3}

    def test_not_equal_to_plain_dict_with_different_entries(self):
        assert _make() != {"a": 1, "b": 2}

    def test_not_equal_to_unrelated_type(self):
        assert _make() != "not a dict"

    def test_not_hashable(self):
        """Mutable containers must not be hashable."""
        with pytest.raises(TypeError):
            hash(_make())


# ===========================================================================
# __repr__
# ===========================================================================

class TestRepr:

    def test_repr_contains_class_name(self):
        assert repr(_make()).startswith("BidirectionalDict(")

    def test_repr_contains_data(self):
        bd = BidirectionalDict({"a": 1})
        assert "'a'" in repr(bd)
        assert "1" in repr(bd)

    def test_empty_repr(self):
        assert repr(BidirectionalDict()) == "BidirectionalDict({})"


# ===========================================================================
# Invariant integrity — internal consistency checks
# ===========================================================================

class TestInvariantIntegrity:

    def _assert_consistent(self, bd: BidirectionalDict) -> None:
        """Assert forward and reverse mappings are perfect mirrors."""
        assert len(bd.data) == len(bd._reverse), (
            "forward and reverse mappings have different lengths"
        )
        for key, value in bd.data.items():
            assert bd._reverse[value] == key, (
                f"reverse[{value!r}] = {bd._reverse.get(value)!r}, expected {key!r}"
            )
        for value, key in bd._reverse.items():
            assert bd.data[key] == value, (
                f"data[{key!r}] = {bd.data.get(key)!r}, expected {value!r}"
            )

    def test_consistency_after_construction(self):
        self._assert_consistent(_make())

    def test_consistency_after_insert(self):
        bd = _make()
        bd["d"] = 4
        self._assert_consistent(bd)

    def test_consistency_after_update(self):
        bd = _make()
        bd["a"] = 99
        self._assert_consistent(bd)

    def test_consistency_after_delete(self):
        bd = _make()
        del bd["a"]
        self._assert_consistent(bd)

    def test_consistency_after_pop(self):
        bd = _make()
        bd.pop("b")
        self._assert_consistent(bd)

    def test_consistency_after_popitem(self):
        bd = _make()
        bd.popitem()
        self._assert_consistent(bd)

    def test_consistency_after_clear_and_reinsert(self):
        bd = _make()
        bd.clear()
        bd["x"] = 100
        self._assert_consistent(bd)

    def test_key_and_value_sets_are_disjoint(self):
        """Mutual exclusivity: no token appears in both roles."""
        bd = _make()
        bd["d"] = 4
        keys = set(bd.data.keys())
        values = set(bd.data.values())
        assert keys.isdisjoint(values), (
            f"Keys and values overlap: {keys & values}"
        )

    def test_values_are_unique(self):
        """Injectivity: every value maps back to exactly one key."""
        bd = _make()
        values = list(bd.data.values())
        assert len(values) == len(set(values)), "Duplicate values found"