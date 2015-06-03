import logging
logging.basicConfig(level=logging.DEBUG)

import math
import nengo
import nengo_spinnaker
import numpy as np

from nengo.processes import WhiteSignal

def test_white_signal():
    model = nengo.Network()
    with model:
        # Create communication channel
        pre = nengo.Ensemble(60, dimensions=2)
        post = nengo.Ensemble(60, dimensions=2)
        nengo.Connection(pre, post)

        inp = nengo.Node(WhiteSignal(60, high=5), size_out=2)
        nengo.Connection(inp, pre)

        # Probe signal and ensemble at end of channel
        inp_p = nengo.Probe(pre, synapse=0.01)
        post_p = nengo.Probe(post, synapse=0.01)

    nengo_spinnaker.add_spinnaker_params(model.config)
    model.config[inp].function_of_time = True

    sim = nengo_spinnaker.Simulator(model)
    with sim:
        sim.run(2.0)

    # Read data
    in_data = sim.data[inp_p]
    post_data = sim.data[post_p]

    # Calculate RMSD
    error = np.power(in_data - post_data, 2.0)

    # Assert it's less than arbitrary limit
    assert math.sqrt(np.mean(error)) < 0.1


if __name__=="__main__":
    test_white_signal()
