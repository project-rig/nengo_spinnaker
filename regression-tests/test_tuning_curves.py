import nengo
import nengo_spinnaker
import numpy as np


def test_tuning_curves_lif():
    """This tests that the neuron model on SpiNNaker doesn't diverge too far
    from the reference LIF model.
    """
    with nengo.Network() as model:
        a = nengo.Ensemble(100, 1)
        a.bias = np.linspace(0.0, 40.0, a.n_neurons)
        a.gain = np.ones(a.n_neurons)
        p = nengo.Probe(a.neurons)


    # Run with Nengo neurons
    nsim = nengo.Simulator(model)
    nsim.run(1.0)

    # Run on SpiNNaker
    ssim = nengo_spinnaker.Simulator(model)
    with ssim:
        ssim.run(1.0)

    # Calculate the rates
    n_rates = np.sum(nsim.data[p] != 0, axis=0)
    s_rates = np.sum(ssim.data[p] != 0, axis=0)

    # Calculate the RMSE
    rmse = np.sqrt(np.mean(np.square(n_rates - s_rates)))
    assert rmse < 5.0  # Nothing special about 5.0


if __name__ == "__main__":
    test_tuning_curves_lif()
