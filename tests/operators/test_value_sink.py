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


@pytest.mark.parametrize("size_in, n_steps", [(1, 2), (2, 1)])
def test_value_sink_after_simulation_not_already_probed(size_in, n_steps):
    """Test that the post-simulation function reads back from the recording
    region and formats the data correctly.
    """
    probe = mock.Mock()
    probe.size_in = size_in
    probe.sample_every = None

    # Create a value sink with a dummy recording region
    v = ValueSink(probe, 0.001)
    v.recording_region_mem = mock.Mock()
    v.recording_region_mem.read.return_value = \
        b'\xff\xff\x00\x00\x00\x00\xff\xff'

    # Create a mock simulator which can be modified to store the probe data
    sim = mock.Mock()
    sim.data = dict()
    v.after_simulation(None, sim, n_steps)

    assert sim.data[probe].shape == (n_steps, size_in)


def test_value_sink_after_simulation_already_probed():
    """Test that the post-simulation function reads back from the recording
    region and formats the data correctly and that the existing probe data is
    extended when probing has already occurred.
    """
    probe = mock.Mock()
    probe.size_in = 2
    probe.sample_every = None

    # Create a value sink with a dummy recording region
    v = ValueSink(probe, 0.001)
    v.recording_region_mem = mock.Mock()
    v.recording_region_mem.read.return_value = \
        b'\xff\xff\x00\x00\x00\x00\xff\xff'

    # Create a mock simulator which can be modified to store the probe data
    sim = mock.Mock()
    sim.data = dict()
    sim.data[probe] = np.zeros((10, 2))
    v.after_simulation(None, sim, 1)

    assert sim.data[probe].shape == (11, 2)
    assert np.all(sim.data[probe][:10] == np.zeros((10, 2)))


@pytest.mark.parametrize("timestep, size_in", [(1000, 3), (2000, 9)])
def test_system_region(timestep, size_in):
    """Create a system region, check that the size is reported correctly and
    that the values are written out correctly.
    """
    region = SystemRegion(timestep, size_in)

    # This region should always require 8 bytes
    assert region.sizeof() == 8

    # Determine what we expect the system region to work out as.
    expected_data = struct.pack("<2I", timestep, size_in)

    fp = tempfile.TemporaryFile()
    region.write_subregion_to_file(fp)
    fp.seek(0)

    assert fp.read() == expected_data
