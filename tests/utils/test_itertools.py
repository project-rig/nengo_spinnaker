from nengo_spinnaker.utils import itertools as nsitertools


def test_flatten():
    """Test flattening of nested iterables."""
    assert list(nsitertools.flatten([1, 2, [3, 4, 5, [[6]]]])) == \
        [1, 2, 3, 4, 5, 6]
