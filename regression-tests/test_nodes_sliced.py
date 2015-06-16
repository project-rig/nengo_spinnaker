"""More complex function of time Node example.
"""
import nengo
import nengo_spinnaker
import numpy as np
import pytest


@pytest.mark.parametrize("f_of_t", [True, False])
def test_nodes_sliced(f_of_t):
    # Create a model with a single function of time node which returns a 4D
    # vector, apply preslicing on some connections from it and ensure that this
    # slicing plays nicely with the functions attached to the connections.
    def out_fun_1(val):
        assert val.size == 2
        return val * 2

    with nengo.Network() as model:
        # Create the input node and an ensemble
        in_node = nengo.Node(lambda t: [0.1, 1.0, 0.2, -1.0], size_out=4)
        in_node_2 = nengo.Node(0.25)

        ens = nengo.Ensemble(400, 4)
        ens2 = nengo.Ensemble(200, 2)

        # Create the connections
        nengo.Connection(in_node[::2], ens[[1, 3]], transform=.5,
                         function=out_fun_1)
        nengo.Connection(in_node_2[[0, 0]], ens2)

        # Probe the ensemble to ensure that the values are correct
        p = nengo.Probe(ens, synapse=0.05)
        p2 = nengo.Probe(ens2, synapse=0.05)

    # Mark the input as being a function of time if desired
    if f_of_t:
        nengo_spinnaker.add_spinnaker_params(model.config)
        model.config[in_node].function_of_time = True

    # Run the simulator for 1.0 s and check that the last probed values are in
    # range
    sim = nengo_spinnaker.Simulator(model)
    with sim:
        sim.run(1.0)

    # Check the final values
    assert -0.05 < sim.data[p][-1, 0] < 0.05
    assert 0.05 < sim.data[p][-1, 1] < 0.15
    assert -0.05 < sim.data[p][-1, 2] < 0.05
    assert 0.15 < sim.data[p][-1, 3] < 0.25

    assert 0.20 < sim.data[p2][-1, 0] < 0.30
    assert 0.20 < sim.data[p2][-1, 1] < 0.30


if __name__ == "__main__":
    test_nodes_sliced(True)
    test_nodes_sliced(False)
