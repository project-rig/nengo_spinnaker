import mock
import numpy as np
import pytest
import struct
import tempfile

from nengo_spinnaker.operators.filter import (SystemRegion,
                                              get_transforms_and_keys)


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


def test_get_transforms_and_keys():
    """Test that the complete transform matrix is constructed correctly and
    that appropriate keys are assigned.
    """
    # Create 2 mock signals and associated connections
    sig_a = mock.Mock(name="signal A")
    sig_a_ks_0 = mock.Mock()
    sig_a_ks_1 = mock.Mock()
    sig_a_kss = {
        0: sig_a_ks_0,
        1: sig_a_ks_1,
    }

    sig_a.keyspace = mock.Mock()
    sig_a.keyspace.side_effect = lambda index: sig_a_kss[index]

    conn_a = mock.Mock(name="connection A")
    transform_a = np.eye(2)

    sig_b = mock.Mock(name="signal B")
    sig_b_ks_0 = mock.Mock()
    sig_b_kss = {
        0: sig_b_ks_0,
    }

    sig_b.keyspace = mock.Mock()
    sig_b.keyspace.side_effect = lambda index: sig_b_kss[index]

    conn_b = mock.Mock(name="connection B")
    transform_b = np.array([[0.5, 0.5]])

    # Create the dictionary type that will be used
    signals_connections = {
        sig_a: [conn_a],
        sig_b: [conn_b],
    }

    # Get the transforms and keys
    with mock.patch("nengo_spinnaker.operators.filter.full_transform") as ft:
        def full_transform(conn, allow_scalars):
            assert not allow_scalars
            if conn is conn_a:
                return transform_a
            elif conn is conn_b:
                return transform_b
            else:
                assert False, "Unexpected connection."

        ft.side_effect = full_transform

        transforms, keys = get_transforms_and_keys(signals_connections)

    # Check that the transforms and keys are correct
    assert set(keys) == set([sig_a_ks_0, sig_a_ks_1, sig_b_ks_0])
    assert transforms.shape == (len(keys), 2)
    assert (np.all(transforms[0] == transform_b) or
            np.all(transforms[2] == transform_b))


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

    # Create a mock connection
    conn = mock.Mock()

    signals_connections = {sig: [conn]}

    # Get the transform and keys
    with mock.patch("nengo_spinnaker.operators.filter.full_transform") as ft:
        ft.return_value = transform
        t, keys = get_transforms_and_keys(signals_connections)

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


@pytest.mark.xfail
def test_get_transforms_and_keys_no_connection():
    """Test that an identity matrix of the size of the signal is used when the
    no connection is provided.
    """
    sig_a = mock.Mock()
    sig_a.width = 5
    sigs_conns = {sig_a: []}

    # Get the signals and keys
    transform, keys = get_transforms_and_keys(sigs_conns)
    assert np.all(transform == np.eye(sig_a.width))


def test_get_transforms_and_keys_multiple_connections_fails():
    """Filters can't deal with multiple connections being represented by one
    signal.
    """
    sig_a = mock.Mock()
    sig_a.width = 5
    sigs_conns = {sig_a: [mock.Mock() for _ in range(3)]}

    # Get the signals and keys
    with pytest.raises(NotImplementedError):
        get_transforms_and_keys(sigs_conns)


def test_get_transforms_and_keys_noting():
    """Check that no transform and no keys are returned for empty connection
    sets.
    """
    tr, keys = get_transforms_and_keys(dict())

    assert keys == list()
    assert tr.ndim == 2
