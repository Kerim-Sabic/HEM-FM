from hemfm.external_lock import _canonical_hash


def test_external_lock_hash_is_order_invariant():
    assert _canonical_hash({"a": 1, "b": 2}) == _canonical_hash({"b": 2, "a": 1})

