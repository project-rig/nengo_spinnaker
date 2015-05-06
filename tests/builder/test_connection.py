import nengo
import numpy as np

from nengo_spinnaker.builder.connection import build_generic_connection_params


def test_build_standard_connection_params():
    # Create the test network
    with nengo.Network():
        a = nengo.Node(lambda t: [t, t], size_in=0, size_out=2)
        b = nengo.Node(lambda t, x: None, size_in=1, size_out=0)
        a_b = nengo.Connection(a[0], b)

    # Build the connection parameters
    params = build_generic_connection_params(None, a_b)
    assert params.decoders is None
    assert np.all(params.transform == [[1.0, 0.0]])
    assert params.eval_points is None
    assert params.solver_info is None
