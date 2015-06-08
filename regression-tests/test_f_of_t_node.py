import logging
logging.basicConfig(level=logging.DEBUG)

import nengo
import nengo_spinnaker
import numpy as np


def test_function_of_time_node():
    with nengo.Network("Test Network") as network:
        a = nengo.Node(lambda t: 0.6 if t < 1.0 else -0.4)

        b = nengo.Ensemble(200, 1)
        p_a = nengo.Probe(a, synapse=0.05)
        p_b = nengo.Probe(b, synapse=0.05)

        c = nengo.Ensemble(200, 1)
        p_c = nengo.Probe(c, synapse=0.05)

        nengo.Connection(a, b)
        nengo.Connection(a, c, function=lambda x: x**2)

    # Mark `a` as a function of time Node
    nengo_spinnaker.add_spinnaker_params(network.config)
    network.config[a].function_of_time = True

    # Create the simulate and simulate
    sim = nengo_spinnaker.Simulator(network, period=1.0)

    # Run the simulation for long enough to ensure that the decoded value is
    # with +/-20% of the input value.
    with sim:
        sim.run(2.0)

    # Check that the values are decoded as expected
    index10 = int(p_b.synapse.tau * 3 / sim.dt)
    index11 = 1.0 / sim.dt
    index20 = index11 + int(p_b.synapse.tau * 3 / sim.dt)
    data = sim.data[p_b]

    assert (np.all(+0.44 <= data[index10:index11, 0]) and
            np.all(+0.72 >= data[index10:index11, 0]) and
            np.all(-0.32 >= data[index20:, 0]) and
            np.all(-0.48 <= data[index20:, 0]))


if __name__ == "__main__":
    test_function_of_time_node()
