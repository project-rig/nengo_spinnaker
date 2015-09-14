import nengo
import nengo_spinnaker
import numpy as np
import pytest
import time


@pytest.mark.parametrize("timescale", [0.5, 0.1]) # Half-time, 10th time
@pytest.mark.parametrize("f_of_t", [True, False])
def test_time_scaling(timescale, f_of_t):
    """Test that for various time-scales the model results in the same
    behaviour but takes different times to compute.
    """
    # Create a model to test
    with nengo.Network() as model:
        inp = nengo.Node(lambda t: -0.75 if t < 1.0 else .25)
        ens = nengo.Ensemble(100, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens, synapse=0.01)

    # Mark function of time Node if necessary
    if f_of_t:
        nengo_spinnaker.add_spinnaker_params(model.config)
        model.config[inp].function_of_time = True

    # Perform the simulation
    runtime = 2.0
    sim = nengo_spinnaker.Simulator(model, timescale=timescale)
    with sim:
        start = time.time()
        sim.run(runtime)
        duration = time.time() - start

    # Assert that the output is reasonable
    assert np.allclose(sim.data[p][100:900], -0.75, atol=1e-1)
    assert np.allclose(sim.data[p][1500:1900], +0.25, atol=1e-1)

    # Assert that the simulation took a reasonable amount of time
    assert 0.8 <= (timescale * duration) / runtime <= 1.2
