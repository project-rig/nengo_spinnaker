import logging
logging.basicConfig(level=logging.DEBUG)

import nengo
import nengo_spinnaker
import numpy as np


def test_probe_ensemble_voltages():
    with nengo.Network("Test Network") as network:
        # Create an Ensemble with 2 neurons that have known gain and bias. The
        # result is that we know how the membrane voltage should change over
        # time even with no external stimulus.
        ens = nengo.Ensemble(2, 1)
        ens.bias = [0.5, 1.0]
        ens.gain = [0.0, 0.0]

        # Add the voltage probe
        probe = nengo.Probe(ens.neurons, "voltage")

    # Compute the rise time to 95%
    max_t = -ens.neuron_type.tau_rc * np.log(0.05)

    # Run the simulation for this period of time
    sim = nengo_spinnaker.Simulator(network)
    with sim:
        sim.run(max_t)

    # Compute the ideal voltage curves
    c = 1.0 - np.exp(-sim.trange() / ens.neuron_type.tau_rc)
    ideal = np.dot(ens.bias[:, np.newaxis], c[np.newaxis, :]).T

    # Assert that the ideal curves match the retrieved curves well
    assert np.allclose(ideal, sim.data[probe], atol=1e-3)


if __name__ == "__main__":
    test_probe_ensemble_voltages()
