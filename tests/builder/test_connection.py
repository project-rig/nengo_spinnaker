import mock
import nengo
import numpy as np

from nengo_spinnaker.builder.builder import Model
from nengo_spinnaker.builder.model import InputPort, OutputPort
from nengo_spinnaker.builder.connection import (
    generic_source_getter,
    generic_sink_getter,
    build_generic_reception_params,
    EnsembleTransmissionParameters,
    PassthroughNodeTransmissionParameters,
    NodeTransmissionParameters
)


def test_generic_source_getter():
    """Test the generic source object getter, this should just return the
    object that is associated with the connection pre_obj.
    """
    # Create the connection
    conn = mock.Mock(name="connection", spec_set=["pre_obj", "post_obj"])
    conn.pre_obj = mock.Mock(name="pre object")
    conn.post_obj = mock.Mock(name="post object")

    # Create the model
    pre_int = mock.Mock(name="pre intermediate")
    post_int = mock.Mock(name="post intermediate")

    model = Model()
    model.object_operators[conn.pre_obj] = pre_int
    model.object_operators[conn.post_obj] = post_int

    # Get the source
    spec = generic_source_getter(model, conn)
    assert spec.target.obj is pre_int
    assert spec.target.port is OutputPort.standard


def test_generic_sink_getter():
    """Test the generic sink object getter, this should just return the
    object that is associated with the connection pre_obj.
    """
    # Create the connection
    conn = mock.Mock(name="connection", spec_set=["pre_obj", "post_obj"])
    conn.pre_obj = mock.Mock(name="pre object")
    conn.post_obj = mock.Mock(name="post object")

    # Create the model
    pre_int = mock.Mock(name="pre intermediate")
    post_int = mock.Mock(name="post intermediate")

    model = Model()
    model.object_operators[conn.pre_obj] = pre_int
    model.object_operators[conn.post_obj] = post_int

    # Get the sink
    spec = generic_sink_getter(model, conn)
    assert spec.target.obj is post_int
    assert spec.target.port is InputPort.standard


def test_build_standard_reception_params():
    # Create the test network
    with nengo.Network():
        a = nengo.Node(lambda t: [t, t], size_in=0, size_out=2)
        b = nengo.Node(lambda t, x: None, size_in=1, size_out=0)
        a_b = nengo.Connection(a[0], b, synapse=0.03)

    # Build the transmission parameters
    params = build_generic_reception_params(None, a_b)
    assert params.filter is a_b.synapse


class TestEnsembleTransmissionParameters(object):
    def test_eq_ne(self):
        """Create a series of EnsembleTransmissionParameters and ensure that
        they only report equal when they are.
        """
        class MyETP(EnsembleTransmissionParameters):
            pass

        tp1 = EnsembleTransmissionParameters(np.ones((3, 3)), np.eye(3))
        tp2 = EnsembleTransmissionParameters(np.ones((1, 1)), np.eye(1))
        tp3 = EnsembleTransmissionParameters(np.eye(3), np.eye(3))
        tp4 = MyETP(np.ones((3, 3)), np.eye(3))

        assert tp1 != tp2
        assert tp1 != tp3
        assert tp1 != tp4

        tp5 = EnsembleTransmissionParameters(np.ones((3, 3)), np.eye(3))
        assert tp1 == tp5

        tp6 = EnsembleTransmissionParameters(np.ones((3, 1)), np.ones((3, 1)))
        assert tp1 == tp6


class TestNodeTransmissionParameters(object):
    def test_eq_ne(self):
        class MyNTP(NodeTransmissionParameters):
            pass

        # NodeTransmissionParameters are only equivalent if they are of the
        # same type, they share the same pre_slice and transform.
        pars = [
            (NodeTransmissionParameters, (slice(0, 5), None, np.ones((5, 5)))),
            (NodeTransmissionParameters, (slice(None), None, np.ones((5, 5)))),
            (NodeTransmissionParameters, (slice(0, 5), None, np.eye(5))),
            (NodeTransmissionParameters, (slice(0, 5), None, np.ones((1, 1)))),
            (NodeTransmissionParameters,
             (slice(0, 5), lambda x: x, np.ones((5, 5)))),
            (MyNTP, (slice(0, 5), None, np.ones((5, 5)))),
        ]
        ntps = [cls(*args) for cls, args in pars]

        # Check the inequivalence works
        for a in ntps:
            for b in ntps:
                if a is not b:
                    assert a != b

        # Check that equivalence works
        for a, b in zip(ntps, [cls(*args) for cls, args in pars]):
            assert a is not b
            assert a == b
