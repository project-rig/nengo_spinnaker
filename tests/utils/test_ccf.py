from nengo_spinnaker.utils.ccf import minimise
import pytest


@pytest.mark.parametrize(
    "on_set",
    [{(0b00, 0b11), (0b01, 0b11)},  # 0 and 0 => 0, 0 and 1 => X
     {(0b10, 0b11), (0b11, 0b11)},  # 1 and 1 => 1, 0 and 1 => X
     {(0b0, 0b1), (0b0, 0b0)},  # 0 and X => X
     {(0b1, 0b1), (0b0, 0b0)},  # 1 and X => X
     {(0b0, 0b0), (0b0, 0b0)},  # X and X => X
     ])
def test_leaf_with_no_offset(on_set):
    """Check that minimising a leaf with no off-set yields a single entry which
    combining the elements of the on-set.
    """
    minimised = set(minimise(on_set, {}))

    # Check that all of the original keys match against the minimised set
    for key, _ in on_set:
        assert any(key & mask == ekey for ekey, mask in minimised)


@pytest.mark.parametrize(
    "on_set, off_set, expected",
    [({(0b00, 0b11), (0b01, 0b11)},
      {(0b10, 0b11)},
      {(0b00, 0b10)}),
     ({(0b01, 0b11), (0b11, 0b11)},
      {(0b00, 0b11)},
      {(0b01, 0b01)}),
     ])
def test_choose_column_and_empty_column(on_set, off_set, expected):
    """Test that a column containing more 0s or 1s is chosen and that the empty
    other column is handled appropriately."""
    assert set(minimise(on_set, off_set)) == expected


def test_no_rechoose_column():
    """Test that a column is not repeatedly chosen (leading to infinite
    recursion).
    """
    on_set = {(0b000, 0b111),
              (0b001, 0b111),
              (0b010, 0b111),
              (0b111, 0b111)}
    off_set = {(0b011, 0b111)}

    assert set(minimise(on_set, off_set)) == {
        (0b000, 0b101),
        (0b001, 0b111),
        (0b111, 0b111),
    }


def test_with_xs():
    """Test that a column containing Xs isn't selected as the column on which
    to split.
    """
    on_set = {(0b1010, 0b1110)}
    off_set = {(0b1000, 0b1110),
               (0b0110, 0b1110)}
    minimised = set(minimise(on_set, off_set))
