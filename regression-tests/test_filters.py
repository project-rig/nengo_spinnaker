import numpy as np
import nengo
import nengo_spinnaker
import pytest

import logging
logging.basicConfig(level=logging.DEBUG)


def test_none_filter():
    with nengo.Network() as network:
        inp = nengo.Node([1.0, 0.5])
        probe = nengo.Probe(inp, synapse=None)

    # Simulate the network
    sim = nengo_spinnaker.Simulator(network)
    with sim:
        sim.run(0.1)

    # Check that the probed value is as expected, calculate the mean of the
    # received data and ensure this is close to the given input (this accounts
    # for dropped packets).
    mean = np.mean(sim.data[probe], axis=0)
    assert np.all(0.9 * inp.output <= mean)
    assert np.all(1.1 * inp.output >= mean)


@pytest.mark.parametrize("tau", [0.01, 0.05, 0.1])
def test_lowpass_filter(tau):
    with nengo.Network() as network:
        inp = nengo.Node([1.0, -0.4])
        probe = nengo.Probe(inp, synapse=tau)

    # Simulate the network
    sim = nengo_spinnaker.Simulator(network)
    with sim:
        sim.run(tau * 3)

    # Check that the probed value is near the expected value
    assert np.allclose(
        sim.data[probe],
        (np.array([inp.output]).T * (1.0 - np.exp(-sim.trange()/tau))).T,
        atol=0.10
    )


@pytest.mark.parametrize("num, den, t, rmse, tolerance",
                         [([1.0], [0.001, 0.11, 1.0], 0.4, 0.05, 0.1)])
def test_lti_filter(num, den, t, rmse, tolerance):
    """Test the LTI filter."""
    # Create the network
    with nengo.Network() as network:
        step = nengo.Node([1.0, -0.4])
        probe = nengo.Probe(step, synapse=nengo.LinearFilter(num, den))

    # Simulate with reference Nengo
    nsim = nengo.Simulator(network)
    nsim.run(t)

    # Simulate with SpiNNaker
    ssim = nengo_spinnaker.Simulator(network)
    with ssim:
        ssim.run(t)

    # Calculate the error
    error = nsim.data[probe] - ssim.data[probe]
    assert np.sqrt(np.mean(np.square(error))) <= rmse
    assert np.allclose(nsim.data[probe], ssim.data[probe], atol=tolerance)


if __name__ == "__main__":
    test_none_filter()
    test_lowpass_filter(0.01)
    test_lti_filter([0.1, 1.0], [0.01, 0.15, 1.0], 1.0, 0.02, 0.03)
