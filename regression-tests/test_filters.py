import numpy as np
import nengo
import nengo_spinnaker
import pytest


def test_none_filter():
    with nengo.Network() as network:
        inp = nengo.Node([1.0, 0.5])
        probe = nengo.Probe(inp, synapse=None)

    # Simulate the network
    sim = nengo_spinnaker.Simulator(network)
    with sim:
        sim.run(0.1)

    # Check that the probed value is as expected
    assert np.all(sim.data[probe][:] == inp.output)


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
        sim.data[probe].T,
        inp.output.reshape((2, 1)) * (1.0 - np.exp(-sim.trange()/tau)),
        atol=0.01
    )


if __name__ == "__main__":
    # test_none_filter()
    test_lowpass_filter(0.01)
