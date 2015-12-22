import mock
import numpy as np
import pytest
import struct
import tempfile

from nengo_spinnaker.builder.model import SignalParameters
from nengo_spinnaker.operators.filter import (SystemRegion,
                                              get_transforms_and_keys,
                                              ParallelFilterSlice)
from nengo_spinnaker.builder.ensemble import EnsembleTransmissionParameters
from nengo_spinnaker.builder.node import PassthroughNodeTransmissionParameters


class TestParallelFilterSlice(object):
    def test_accepts_signal(self):
        # Create a series of vertex slices
        pfss = [
            ParallelFilterSlice(slice(0, 16), slice(None)),
            ParallelFilterSlice(slice(8, 24), slice(None)),
            ParallelFilterSlice(slice(16, 32), slice(None)),
        ]

        # Create a series of transmission parameters, of current known types.
        ens_0_to_8 = np.random.uniform(size=(100, 10))
        ens_0_to_8[:, 8:] = 0.0
        tp0 = EnsembleTransmissionParameters(ens_0_to_8, 1.0)

        ptn_20_to_26 = np.zeros((32, 6))
        ptn_20_to_26[20:26, :] = np.eye(6)
        tp1 = PassthroughNodeTransmissionParameters(ptn_20_to_26)

        # Check that `accepts_signal` responds correctly
        assert pfss[0].accepts_signal(None, tp0)
        assert not pfss[1].accepts_signal(None, tp0)
        assert not pfss[2].accepts_signal(None, tp0)

        assert not pfss[0].accepts_signal(None, tp1)
        assert pfss[1].accepts_signal(None, tp1)
        assert pfss[2].accepts_signal(None, tp1)

    def test_transmits_signal(self):
        # Create a series of transmission parameters
        ptn_2_to_8 = np.zeros((32, 6))
        ptn_2_to_8[2:8, :] = np.eye(6)
        tp0 = PassthroughNodeTransmissionParameters(ptn_2_to_8.T)
        tp0_slice = set(range(2, 8))

        ptn_20_to_26 = np.zeros((32, 6))
        ptn_20_to_26[20:26, :] = np.eye(6)
        tp1 = PassthroughNodeTransmissionParameters(ptn_20_to_26.T)
        tp1_slice = set(range(20, 26))

        signal_parameter_slices = [(tp0, tp0_slice),
                                   (tp1, tp1_slice),
                                   ]

        # Create a series of vertex slices
        pfss = [
            ParallelFilterSlice(slice(None), slice(0, 16), {},
                                signal_parameter_slices),
            ParallelFilterSlice(slice(None), slice(8, 24), {},
                                signal_parameter_slices),
            ParallelFilterSlice(slice(None), slice(16, 32), {},
                                signal_parameter_slices),
        ]

        # Check that `transmits_signal` responds correctly
        assert pfss[0].transmits_signal(None, tp0)
        assert not pfss[1].transmits_signal(None, tp0)
        assert not pfss[2].transmits_signal(None, tp0)

        assert not pfss[0].transmits_signal(None, tp1)
        assert pfss[1].transmits_signal(None, tp1)
        assert pfss[2].transmits_signal(None, tp1)


class TestSystemRegion(object):
    def test_sizeof(self):
        # Create a system region
        sr = SystemRegion(n_dims=512, machine_timestep=1000)

        # Should always be 6 words
        assert sr.sizeof() == 6 * 4

    @pytest.mark.parametrize(
        "n_dims, in_slice, out_slice, machine_timestep, vector_address",
        [(256, slice(0, 10), slice(10, 12), 1000, 0x67800000),
         (100, slice(5, 7), slice(0, 4), 2000, 0x67880000),
         ]
    )
    def test_write_subregion_to_file(self, n_dims, in_slice, out_slice,
                                     machine_timestep, vector_address):
        # Create the region
        sr = SystemRegion(n_dims=n_dims, machine_timestep=machine_timestep)

        # Store the address of the shared vector in SDRAM
        sr.shared_vector_address = vector_address

        # Write the region to file, assert the values are sane
        fp = tempfile.TemporaryFile()
        sr.write_subregion_to_file(fp, in_slice=in_slice, out_slice=out_slice)

        fp.seek(0)
        assert struct.unpack("<6I", fp.read()) == (
            machine_timestep,
            n_dims,
            in_slice.start,
            in_slice.stop - in_slice.start,
            out_slice.stop - out_slice.start,
            vector_address,
        )


def test_get_transforms_and_keys():
    """Test that the complete transform matrix is constructed correctly and
    that appropriate keys are assigned.
    """
    # Create 2 mock signals and associated connections
    sig_a_ks_0 = mock.Mock()
    sig_a_ks_1 = mock.Mock()
    sig_a_kss = {
        0: sig_a_ks_0,
        1: sig_a_ks_1,
    }

    sig_a_ks = mock.Mock()
    sig_a_ks.side_effect = lambda index: sig_a_kss[index]
    sig_a = SignalParameters(keyspace=sig_a_ks)

    conn_a = PassthroughNodeTransmissionParameters(np.eye(2))

    sig_b_ks_0 = mock.Mock()
    sig_b_kss = {
        0: sig_b_ks_0,
    }

    sig_b_ks = mock.Mock()
    sig_b_ks.side_effect = lambda index: sig_b_kss[index]
    sig_b = SignalParameters(keyspace=sig_b_ks)

    conn_b = PassthroughNodeTransmissionParameters(np.array([[0.5, 0.5]]))
    transform_b = conn_b.transform

    # Create the dictionary type that will be used
    pars = [(sig_a, conn_a), (sig_b, conn_b)]

    # Get the transforms and keys
    transforms, keys, signal_parameter_slices = get_transforms_and_keys(pars)

    # Check that the transforms and keys are correct
    assert set(keys) == set([sig_a_ks_0, sig_a_ks_1, sig_b_ks_0])
    assert transforms.shape == (len(keys), 2)
    assert (np.all(transforms[0] == transform_b) or
            np.all(transforms[2] == transform_b))

    # Check that the signal parameter slices are correct
    for (par, sl) in signal_parameter_slices:
        if par == conn_a:
            assert sl == set(range(0, 2)) or sl == set(range(1, 3))
        else:
            assert par == conn_b
            assert sl == set(range(0, 1)) or sl == set(range(2, 3))


@pytest.mark.parametrize("latching", [False, True])
def test_get_transforms_and_keys_removes_zeroed_rows(latching):
    """Check that zeroed rows (those that would always result in zero valued
    packets) are removed, and the keys miss this value as well.
    """
    ks = mock.Mock()
    transform = np.ones((10, 5))
    transform[1, :] = 0.0
    transform[4:7, :] = 0.0
    transform[:, 1] = 0.0

    # Create a signal and keyspace
    sig = mock.Mock()
    sig.keyspace = ks
    sig.latching = latching
    sig = SignalParameters(keyspace=ks, latching=latching)

    # Create a mock connection
    conn = PassthroughNodeTransmissionParameters(transform)

    signals_connections = [(sig, conn)]

    # Get the transform and keys
    t, keys, _ = get_transforms_and_keys(signals_connections)

    if not latching:
        # Check the transform is correct
        assert np.all(t ==
                      np.vstack((transform[0], transform[2:4], transform[7:])))

        # Check the keys were called for correctly
        ks.assert_has_calls([mock.call(index=0),
                             mock.call(index=2),
                             mock.call(index=3),
                             mock.call(index=7),
                             mock.call(index=8),
                             mock.call(index=9)])
    else:
        # Check the transform is correct
        assert np.all(t == t)

        # Check the keys were called for correctly
        ks.assert_has_calls([mock.call(index=0),
                             mock.call(index=1),
                             mock.call(index=2),
                             mock.call(index=3),
                             mock.call(index=4),
                             mock.call(index=5),
                             mock.call(index=6),
                             mock.call(index=7),
                             mock.call(index=8),
                             mock.call(index=9)])


def test_get_transforms_and_keys_nothing():
    """Check that no transform and no keys are returned for empty connection
    sets.
    """
    tr, keys, _ = get_transforms_and_keys([])

    assert keys == list()
    assert tr.ndim == 2
