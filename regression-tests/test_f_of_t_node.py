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

    assert np.all(+0.44 <= data[index10:index11, 0])
    assert np.all(+0.72 >= data[index10:index11, 0])
    assert np.all(-0.32 >= data[index20:, 0])
    assert np.all(-0.48 <= data[index20:, 0])


def test_constant_node():
    with nengo.Network("Test Network") as model:
        in_val = nengo.Node([0.5])
        pn = nengo.Node(size_in=1)
        ens = nengo.Ensemble(100, 1)

        nengo.Connection(in_val, pn)
        nengo.Connection(pn, ens)

        probe = nengo.Probe(ens, synapse=0.05)

    sim = nengo_spinnaker.Simulator(model)
    with sim:
        sim.run(0.5)

    assert 0.45 < sim.data[probe][-1] < 0.55


def test_f_of_t_node_with_outgoing_function():
    with nengo.Network() as model:
        a = nengo.Node(0.0)
        b = nengo.Node(size_in=2)
        c = nengo.Ensemble(100, 2)

        output = nengo.Probe(c, synapse=0.01)

        nengo.Connection(a, b, function=lambda x: [np.sin(x), 1], synapse=None)
        nengo.Connection(b, c, synapse=None)

    # Simulate and ensure that the output is [0, 1] towards the end of the
    # simulation
    sim = nengo_spinnaker.Simulator(model)
    with sim:
        sim.run(0.5)

    assert -0.05 < sim.data[output][-1, 0] < 0.05
    assert 0.95 < sim.data[output][-1, 1] < 1.05


if __name__ == "__main__":
    test_function_of_time_node()
    test_constant_node()
    test_f_of_t_node_with_outgoing_function()
