import mock
import numpy as np
import pytest
from rig.place_and_route import Cores, SDRAM
import six
import struct
import tempfile

from nengo_spinnaker.builder.builder import Model, ObjectPort
from nengo_spinnaker.builder.model import SignalParameters
from nengo_spinnaker.builder.node import NodeTransmissionParameters
from nengo_spinnaker.operators import SDPReceiver
from nengo_spinnaker.operators.sdp_receiver import SystemRegion


class TestSystemRegion(object):
    @pytest.mark.parametrize("machine_timestep, size_out",
                             [(1000, 3), (2000, 1)])
    def test_all(self, machine_timestep, size_out):
        # Create a system region
        region = SystemRegion(machine_timestep, size_out)

        # Check that the size is correct
        assert region.sizeof(slice(None)) == 8

        # Write the data to a file and check that it is correct
        fp = tempfile.TemporaryFile()
        region.write_region_to_file(fp)

        fp.seek(0)
        assert fp.read() == struct.pack("<2I", machine_timestep, size_out)
