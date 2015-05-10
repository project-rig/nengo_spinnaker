import logging
logging.basicConfig(level=logging.DEBUG)

import nengo
import nengo_spinnaker
import numpy as np


def test_constant_node_ensemble_and_value_probe():
    with nengo.Network("Test Network") as network:
        # a = nengo.Node(0.5)
        b = nengo.Ensemble(100, 1)
        # nengo.Connection(a, b, synapse=0.05)
        p = nengo.Probe(b, synapse=0.01)

    # Create the simulate and simulate
    sim = nengo_spinnaker.SpiNNakerSimulator(network)

    # Run the simulation for long enough to ensure that the decoded value is
    # with +/-10% of the input value.
    sim.run(2.0)

    # Check that the value was decoded as expected
    print(sim.data[p])
    assert (np.all(0.45 <= sim.data[p][int(0.001 * 0.05 * 2.5):]) and
            np.all(sim.data[p][int(0.001 * 0.05 * 2.5):] <= 0.55))


if __name__ == "__main__":

    test_constant_node_ensemble_and_value_probe()
