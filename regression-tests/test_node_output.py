import logging
logging.basicConfig(level=logging.DEBUG)

import nengo
import nengo_spinnaker
import numpy as np


def test_node_output_transmitted_to_board():
    with nengo.Network("Test Network") as network:
        a = nengo.Node(lambda t: 0.5)
        b = nengo.Ensemble(100, 2)
        nengo.Connection(a, b, transform=[[-1.0], [1.0]])
        p = nengo.Probe(b, synapse=0.05)

    # Create the simulate and simulate
    sim = nengo_spinnaker.Simulator(network)

    # Run the simulation for long enough to ensure that the decoded value is
    # with +/-20% of the input value.
    with sim:
        sim.run(2.0)

    # Check that the value was decoded as expected
    index = int(p.synapse.tau * 2.5 / sim.dt)
    data = sim.data[p]
    assert(np.all(+0.40 <= data[index:, 1]) and
           np.all(+0.60 >= data[index:, 1]) and
           np.all(-0.60 <= data[index:, 0]) and
           np.all(-0.40 >= data[index:, 0]))


if __name__ == "__main__":
    test_node_output_transmitted_to_board()
