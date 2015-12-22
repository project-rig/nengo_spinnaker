import pytest

from nengo_spinnaker.utils import neurons


@pytest.mark.parametrize(
    "slices, expected",
    [([slice(0, 10)], 4),
     ([slice(0, 33), slice(33, 35)], 12)
     ])
def test_get_bytes_for_unpacked_spike_vector(slices, expected):
    assert neurons.get_bytes_for_unpacked_spike_vector(slices) == expected
