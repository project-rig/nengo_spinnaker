import tempfile
import pytest
import struct

from nengo_spinnaker.operators.value_source import SystemRegion


class TestSystemRegion(object):
    def test_sizeof(self):
        # Create a region
        sr = SystemRegion(1000, True, 1)

        # Check the size is correct
        assert sr.sizeof(slice(0, 5)) == 24

    @pytest.mark.parametrize(
        "timestep, periodic, n_steps, vertex_slice, n_blocks, block_length, "
        "last_block_length",
        [(1000, True, 2000, slice(0, 10), 1, 1280, 720),
         (1000, False, 8000, slice(0, 10), 6, 1280, 320),
         ]
    )
    def test_write_subregion_to_file(self, timestep, periodic, n_steps,
                                     vertex_slice, n_blocks, block_length,
                                     last_block_length):
        # Create the region
        sr = SystemRegion(timestep, periodic, n_steps)

        # Write to file
        fp = tempfile.TemporaryFile()
        sr.write_subregion_to_file(fp, vertex_slice)

        fp.seek(0)
        assert struct.unpack("<6I", fp.read()) == (
            timestep, vertex_slice.stop - vertex_slice.start,
            0x1 if periodic else 0x0, n_blocks, block_length, last_block_length
        )
