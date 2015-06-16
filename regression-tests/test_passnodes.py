import logging
logging.basicConfig(level=logging.DEBUG)

import nengo
import nengo_spinnaker
import numpy as np


def test_probe_passnodes():
    """Test that pass nodes are left on SpiNNaker and that they may be probed.
    """
    class ValueReceiver(object):
        def __init__(self):
            self.ts = list()
            self.values = list()

        def __call__(self, t, x):
            self.ts.append(t)
            self.values.append(x[:])

    with nengo.Network("Test Network") as net:
        # Create an input Node which is a function of time only
        input_node = nengo.Node(lambda t: -0.33 if t < 1.0 else 0.10,
                                label="my input")

        # 3D ensemble array to represent this value
        ens = nengo.networks.EnsembleArray(500, 3, label="reps")

        # Pipe the input to the array and probe the output of the array
        nengo.Connection(input_node, ens.input,
                         transform=[[1.0], [0.0], [-1.0]])
        p_ens = nengo.Probe(ens.output, synapse=0.05)

        # Also add a node connected to the end of the ensemble array to ensure
        # that multiple things correctly receive values from the filter.
        receiver = ValueReceiver()
        n_receiver = nengo.Node(receiver, size_in=3)
        nengo.Connection(ens.output, n_receiver, synapse=0.05)

    # Mark the input Node as being a function of time
    nengo_spinnaker.add_spinnaker_params(net.config)
    net.config[input_node].function_of_time = True

    # Create the simulate and simulate
    sim = nengo_spinnaker.Simulator(net)

    # Run the simulation for long enough to ensure that the decoded value is
    # with +/-20% of the input value.
    with sim:
        sim.run(2.0)

    # Check that the values are decoded as expected
    index10 = int(p_ens.synapse.tau * 3 / sim.dt)
    index11 = 1.0 / sim.dt
    index20 = index11 + index10
    data = sim.data[p_ens]

    assert (np.all(-0.25 >= data[index10:index11, 0]) and
            np.all(-0.40 <= data[index10:index11, 0]) and
            np.all(+0.05 <= data[index20:, 0]) and
            np.all(+0.15 >= data[index20:, 0]))
    assert np.all(-0.05 <= data[:, 1]) and np.all(+0.05 >= data[:, 1])
    assert (np.all(+0.25 <= data[index10:index11, 2]) and
            np.all(+0.40 >= data[index10:index11, 2]) and
            np.all(-0.05 >= data[index20:, 2]) and
            np.all(-0.15 <= data[index20:, 2]))

    # Check that values came into the node correctly
    assert +0.05 <= receiver.values[-1][0] <= +0.15
    assert -0.05 >= receiver.values[-1][2] >= -0.15

if __name__ == "__main__":
    test_probe_passnodes()
