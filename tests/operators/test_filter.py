import pytest
import struct
import tempfile

from nengo_spinnaker.operators.filter import SystemRegion


class TestSystemRegion(object):
    def test_sizeof(self):
        # Create a system region, assert that the size is reported correctly.
        sr = SystemRegion(size_in=5, size_out=10, machine_timestep=1000,
                          transmission_delay=1, interpacket_pause=1)

        # Should always be 5 words
        assert sr.sizeof(slice(None)) == 20

    @pytest.mark.parametrize(
        "size_in, size_out, machine_timestep, transmission_delay, "
        "interpacket_pause",
        [(5, 16, 2000, 6, 1),
         (10, 1, 1000, 1, 2)]
    )
    def test_write_subregion_to_file(self, size_in, size_out, machine_timestep,
                                     transmission_delay, interpacket_pause):
        # Create the region
        sr = SystemRegion(size_in=size_in, size_out=size_out,
                          machine_timestep=machine_timestep,
                          transmission_delay=transmission_delay,
                          interpacket_pause=interpacket_pause)

        # Write the region to file, assert the values are sane
        fp = tempfile.TemporaryFile()
        sr.write_subregion_to_file(fp, slice(None))

        fp.seek(0)
        assert struct.unpack("<5I", fp.read()) == (
            size_in, size_out, machine_timestep, transmission_delay,
            interpacket_pause
        )
