import nengo
import nengo_spinnaker
import numpy as np
import pytest


@pytest.mark.parametrize("source", ("ensemble", "node", "f_of_t_node"))
def test_global_inhibition(source):
    with nengo.Network("Test Network") as network:
        # Create a 2-d ensemble representing a constant value
        input_value = nengo.Node([0.25, -0.3])
        ens = nengo.Ensemble(200, 2)
        nengo.Connection(input_value, ens)
        p_ens = nengo.Probe(ens, synapse=0.05)

        # The gate should be open initially and then closed after 1s
        gate_control = nengo.Node(lambda t: 0.0 if t < 1.0 else 1.0)

        if source == "ensemble":
            # Create another ensemble which will act to gate `ens`
            gate_driver = nengo.Ensemble(100, 1)
            gate_driver.intercepts = [0.3] * gate_driver.n_neurons
            gate_driver.encoders = [[1.0]] * gate_driver.n_neurons
            nengo.Connection(gate_control, gate_driver)
            nengo.Connection(gate_driver, ens.neurons,
                             transform=[[-10.0]] * ens.n_neurons)
        elif source == "node" or source == "f_of_t_node":
            nengo.Connection(gate_control, ens.neurons,
                             transform=[[-10.0]] * ens.n_neurons)

    # Mark appropriate Nodes as functions of time if desired
    if source != "node":
        nengo_spinnaker.add_spinnaker_params(network.config)
        network.config[gate_control].function_of_time = True

    # Create the simulate and simulate
    sim = nengo_spinnaker.Simulator(network)

    # Run the simulation for long enough to ensure that the decoded value is
    # with +/-20% of the input value.
    with sim:
        sim.run(2.0)

    # Check that the values are decoded as expected
    index10 = int(p_ens.synapse.tau * 4 / sim.dt)
    index11 = 1.0 / sim.dt
    index20 = index11 + int(p_ens.synapse.tau * 4 / sim.dt)
    data = sim.data[p_ens]

    assert (np.all(+0.20 <= data[index10:index11, 0]) and
            np.all(+0.30 >= data[index10:index11, 0]) and
            np.all(+0.05 >= data[index20:, 0]) and
            np.all(-0.05 <= data[index20:, 0]))
    assert (np.all(-0.36 <= data[index10:index11, 1]) and
            np.all(-0.24 >= data[index10:index11, 1]) and
            np.all(+0.05 >= data[index20:, 1]) and
            np.all(-0.05 <= data[index20:, 1]))


def test_global_inhibition_from_optimised_out_passthrough_node():
    with nengo.Network("Test Network") as network:
        # Create a 2-d ensemble representing a constant value
        input_value = nengo.Node([0.25, -0.3])
        ens = nengo.Ensemble(200, 2)
        nengo.Connection(input_value, ens)
        p_ens = nengo.Probe(ens, synapse=0.05)

        # The gate should be open initially and then closed after 1s
        gate_control = nengo.Node(lambda t: 0.0 if t < 1.0 else 1.0)

        # Add the passthrough Node
        gate_ptn = nengo.Node(size_in=ens.n_neurons)
        nengo.Connection(gate_ptn, ens.neurons)

        # Create the control and 
        nengo.Connection(gate_control, gate_ptn,
                         transform=[[-10.0]] * ens.n_neurons)

    # Mark appropriate Nodes as functions of time
    nengo_spinnaker.add_spinnaker_params(network.config)
    network.config[gate_control].function_of_time = True

    # Create the simulate and simulate
    sim = nengo_spinnaker.Simulator(network)

    # Run the simulation for long enough to ensure that the decoded value is
    # with +/-20% of the input value.
    with sim:
        sim.run(2.0)

    # Check that the values are decoded as expected
    index10 = int(p_ens.synapse.tau * 4 / sim.dt)
    index11 = 1.0 / sim.dt
    index20 = index11 + int(p_ens.synapse.tau * 4 / sim.dt)
    data = sim.data[p_ens]

    assert (np.all(+0.20 <= data[index10:index11, 0]) and
            np.all(+0.30 >= data[index10:index11, 0]) and
            np.all(+0.05 >= data[index20:, 0]) and
            np.all(-0.05 <= data[index20:, 0]))
    assert (np.all(-0.36 <= data[index10:index11, 1]) and
            np.all(-0.24 >= data[index10:index11, 1]) and
            np.all(+0.05 >= data[index20:, 1]) and
            np.all(-0.05 <= data[index20:, 1]))


if __name__ == "__main__":
    test_global_inhibition("ensemble")
    test_global_inhibition("node")
    test_global_inhibition("f_of_t_node")
    test_global_inhibition_from_optimised_out_passthrough_node()
