import mock
import nengo
import numpy as np

from nengo_spinnaker.builder.builder import InputPort, Model, OutputPort
from nengo_spinnaker.builder.connection import (
    generic_source_getter,
    generic_sink_getter,
    build_generic_connection_params
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


def test_build_standard_connection_params():
    # Create the test network
    with nengo.Network():
        a = nengo.Node(lambda t: [t, t], size_in=0, size_out=2)
        b = nengo.Node(lambda t, x: None, size_in=1, size_out=0)
        a_b = nengo.Connection(a[0], b)

    # Build the connection parameters
    params = build_generic_connection_params(None, a_b)
    assert params.decoders is None
    assert params.transform == 1.0
    assert params.eval_points is None
    assert params.solver_info is None
