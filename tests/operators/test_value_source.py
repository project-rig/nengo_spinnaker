import mock
import numpy as np
import tempfile
import pytest
import struct

from nengo_spinnaker.operators.value_source import (
    SystemRegion, get_transform_keys)


def test_get_transform_keys():
    """Check that the keys required to transmit data are correctly extracted.
    """
    # Create a transform with an empty row
    transform = np.zeros((5, 10))
    transform[0:4, 0] = [1, 2, 3, 4]
    transform[1, 0] = 0.0

    # Check that the expected transform and signals list are returned
    signal = mock.Mock()
    t_pars = mock.Mock(spec_set=["transform"])
    t_pars.transform = transform

    new_transform, signal_args = get_transform_keys(signal, t_pars)
    assert np.array_equal(
        new_transform,
        transform[np.array([True, False, True, True, False])]
    )

    # Check that the returned signals are correct
    assert signal_args == [
        (signal, dict(index=0)),
        (signal, dict(index=2)),
        (signal, dict(index=3)),
    ]


class TestSystemRegion(object):
    def test_sizeof(self):
        # Create a region
        sr = SystemRegion(1000, True, 1)

        # Check the size is correct
        assert sr.sizeof(slice(0, 5)) == 24

    @pytest.mark.parametrize(
        "timestep, periodic, n_steps, vertex_slice, n_blocks, block_length, "
        "last_block_length",
        [(1000, True, 2000, slice(0, 10), 3, 512, 464),
         (1000, False, 8000, slice(0, 10), 15, 512, 320),
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
