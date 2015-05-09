import pytest

from nengo_spinnaker.regions.region import Region


class TestRegion(object):
    @pytest.mark.parametrize("original_size", [1, 2, 3, 4, 99])
    def test_sizeof_padded(self, original_size):
        """Create a region which takes up an incomplete number of words and
        ensure that this method correctly pads the size to the next number of
        words.
        """
        expected_slice = slice(0, 100)

        class MyRegion(Region):
            def sizeof(self, sl):
                assert sl == expected_slice
                return original_size

        r = MyRegion()
        actual_size = r.sizeof_padded(expected_slice)

        assert actual_size >= original_size and actual_size % 4 == 0
