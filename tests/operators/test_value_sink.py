import mock
import numpy as np
import pytest
import struct
import tempfile

from nengo_spinnaker.operators import ValueSink
from nengo_spinnaker.operators.value_sink import SystemRegion


def test_value_sink_init():
    probe = mock.Mock(name="Probe")
    probe.size_in = 3
    probe.sample_every = 0.0043

    v = ValueSink(probe, 0.001)
    assert v.probe is probe
    assert v.size_in == 3
    assert v.sample_every == 4


@pytest.mark.parametrize("timestep, input_slice", [(1000, slice(0, 10)),
                                                   (2000, slice(10, 100))])
def test_system_region(timestep, input_slice):
    """Create a system region, check that the size is reported correctly and
    that the values are written out correctly.
    """
    region = SystemRegion(timestep, input_slice)

    # This region should always require 12 bytes
    assert region.sizeof() == 12

    # Determine what we expect the system region to work out as.
    expected_data = struct.pack("<3I", timestep,
                                input_slice.stop - input_slice.start,
                                input_slice.start)

    fp = tempfile.TemporaryFile()
    region.write_subregion_to_file(fp)
    fp.seek(0)

    assert fp.read() == expected_data
