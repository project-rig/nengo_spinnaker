import logging
logging.basicConfig(level=logging.DEBUG)

import nengo
import nengo_spinnaker
import numpy as np


def test_node_input_received_from_board():
    """Test that Nodes do receive data from a running simulation.
    """
    # Node just maintains a list of received values
    class NodeCallable(object):
        def __init__(self):
            self.received_values = []

        def __call__(self, t, x):
            self.received_values.append(x)

    nc = NodeCallable()

    with nengo.Network("Test Network") as network:
        # Ensemble representing a constant 0.5
        a = nengo.Node(0.5)
        b = nengo.Ensemble(100, 1)
        nengo.Connection(a, b)

        # Feeds into the target Node with some transforms.  The transforms
        # could be combined in a single connection but we use two here to check
        # that this works!
        node = nengo.Node(nc, size_in=2, size_out=0)
        nengo.Connection(b, node[0], transform=0.5, synapse=0.05)
        nengo.Connection(b, node[1], transform=-1.0, synapse=0.05)

    # Create the simulate and simulate
    sim = nengo_spinnaker.Simulator(network)

    # Run the simulation for long enough to ensure that the decoded value is
    # with +/-20% of the input value.
    with sim:
        sim.run(2.0)

    # All we can really check is that the received values aren't all zero, that
    # the last few are within the expected range.
    vals = np.array(nc.received_values)
    offset = int(0.05 * 3 / sim.dt)
    print(vals[offset:])
    assert np.any(vals != np.zeros(vals.shape))
    assert (np.all(+0.20 <= vals[offset:, 0]) and
            np.all(+0.30 >= vals[offset:, 0]) and
            np.all(-0.40 >= vals[offset:, 1]) and
            np.all(-0.60 <= vals[offset:, 1]))


if __name__ == "__main__":
    test_node_input_received_from_board()
