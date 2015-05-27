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
        p_spikes = nengo.Probe(b.neurons)

    # Mark the input Node as a function of time
    nengo_spinnaker.add_spinnaker_params(network.config)
    network.config[a].function_of_time = True

    # Create the simulator and run for 2 s
    sim = nengo_spinnaker.Simulator(network)
    sim.run(2.0)

    # Check that the neurons spiked as expected
    data = sim.data[p_spikes]
    assert not np.any(data[:1.0/sim.dt, 0])  # Neuron 0 didn't spike at all
    assert np.any(data[:1.0/sim.dt, 1])  # Neuron 1 did spike
    assert np.any(data[1.0/sim.dt:, 0])  # Neuron 0 did spike
    assert not np.any(data[1.0/sim.dt:, 1])  # Neuron 1 didn't spike at all


if __name__ == "__main__":
    test_probe_ensemble_spikes()
