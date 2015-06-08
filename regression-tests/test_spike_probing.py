import logging
logging.basicConfig(level=logging.DEBUG)

import nengo
import nengo_spinnaker
import numpy as np


def test_probe_ensemble_spikes():
    with nengo.Network("Test Network") as network:
        a = nengo.Node(lambda t: -1.0 if t > 1.0 else 1.0)

        # Create a 2 neuron ensemble with opposing encoders
        b = nengo.Ensemble(2, 1)
        b.encoders = [[-1.0], [1.0]]
        b.max_rates = [100, 100]
        b.intercepts = [0.1, -0.1]

        nengo.Connection(a, b, synapse=None)

        # Probe the spikes
        p_n0 = nengo.Probe(b.neurons[0])
        p_n1 = nengo.Probe(b.neurons[1], sample_every=0.002)
        p_spikes = nengo.Probe(b.neurons)

    # Mark the input Node as a function of time
    nengo_spinnaker.add_spinnaker_params(network.config)
    network.config[a].function_of_time = True

    # Create the simulator and run for 2 s
    sim = nengo_spinnaker.Simulator(network)
    with sim:
        sim.run(2.0)

    # Check that the neurons spiked as expected
    assert not np.any(sim.data[p_n0][:1.0/sim.dt])  # Neuron 0
    assert np.any(sim.data[p_n1][:1.0/p_n1.sample_every])  # Neuron 1
    assert np.any(sim.data[p_n0][1.0/sim.dt:])
    assert not np.any(sim.data[p_n1][1.0/p_n1.sample_every:])


if __name__ == "__main__":
    test_probe_ensemble_spikes()
