import pytest
import struct
import tempfile

from nengo_spinnaker.regions import ListRegion


@pytest.mark.parametrize(
    "fstr, items, expected_length",
    [("I", [1, 2, 3], 12),
     ("b", [1, 2], 2),
     ("h", [1, 2], 4)]
)
def test_sizeof(items, fstr, expected_length):
    # Create the region
    lr = ListRegion(fstr)

    # Check the reported size is correct
    assert lr.sizeof(items) == expected_length


@pytest.mark.parametrize(
    "fstr, items",
    [("I", [1, 2, 3]),
     ("b", [1, 2]),
     ("h", [1, 2])]
)
def test_write_subregion_to_file(fstr, items):
    # Create the region
    lr = ListRegion(fstr)

    # Write the items to file
    with tempfile.TemporaryFile() as fp:
        lr.write_subregion_to_file(fp, items)

        # Read back the data
        fp.seek(0)
        data = fp.read()

    # Check that the items were correct
    reconstructed_items = list(
        struct.unpack("<{}{}".format(len(items), fstr), data))

    assert reconstructed_items == items
