import mock
import numpy as np
import pytest
from rig.place_and_route import Cores
import struct
import tempfile

from nengo_spinnaker.builder import Model
from nengo_spinnaker.builder.model import SignalParameters, OutputPort
from nengo_spinnaker.operators.filter import (SystemRegion, Filter, Regions,
                                              get_transforms_and_keys)
from nengo_spinnaker.builder.ensemble import EnsembleTransmissionParameters
from nengo_spinnaker.builder.node import PassthroughNodeTransmissionParameters


class TestFilter(object):
    @pytest.mark.parametrize(
        "size_in, n_expected_slices",
        [(32, 1), (64, 1), (256, 2), (512, 4), (1024, 8), (2056, 17)]
    )
    def test_init(self, size_in, n_expected_slices):
        """Check that the filter is broken into column-slices correctly."""
        f = Filter(size_in)
        assert len(f.groups) == n_expected_slices
        assert all(g.size_in < 528 for g in f.groups)

    def test_make_vertices_no_outgoing_signals(self):
        """Test that no vertices or constraints result if there are no outgoing
        signals.
        """
        # Create a small filter operator
        filter_op = Filter(3)

        # Create an empty model
        m = Model()

        # Make vertices using the model
        netlistspec = filter_op.make_vertices(m, 10000)
        assert len(netlistspec.vertices) == 0
        assert netlistspec.load_function is None
        assert netlistspec.before_simulation_function is None
        assert netlistspec.after_simulation_function is None
        assert netlistspec.constraints is None

    def test_make_vertices_one_group_many_cores_1_chip(self):
        """Test that many vertices are returned if the matrix has many rows and
        that there is an appropriate constraint forcing the co-location of the
        vertices.
        """
        # Create a small filter operator
        filter_op = Filter(3)

        # Create a model and add some connections which will cause packets to
        # be transmitted from the filter operator.
        m = Model()
        signal_parameters = SignalParameters(False, 3, m.keyspaces["nengo"])
        signal_parameters.keyspace.length = 32

        transmission_parameters = \
            PassthroughNodeTransmissionParameters(np.ones((32*3, 3)))
        m.connection_map.add_connection(
            filter_op, OutputPort.standard, signal_parameters,
            transmission_parameters, None, None, None
        )

        # Make vertices using the model
        netlistspec = filter_op.make_vertices(m, 10000)
        assert len(netlistspec.vertices) == 2  # Two vertices

        for vx in netlistspec.vertices:
            assert "filter" in vx.application

            assert vx.resources[Cores] == 1

            assert vx.regions[Regions.system].column_slice == slice(0, 3)

            keys_region = vx.regions[Regions.keys]
            assert keys_region.keyspaces == [
                signal_parameters.keyspace(index=i) for i in range(32*3)]
            assert len(keys_region.fields) == 1
            assert keys_region.partitioned == True

            assert vx.regions[Regions.transform].matrix.shape == (32*3, 3)


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
