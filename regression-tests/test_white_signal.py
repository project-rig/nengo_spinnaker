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
        inp = nengo.Node(WhiteSignal(60, high=5), size_out=2)
        inp_p = nengo.Probe(inp)

    nengo_spinnaker.add_spinnaker_params(model.config)
    model.config[inp].function_of_time = True

    sim = nengo_spinnaker.Simulator(model)
    with sim:
        sim.run(0.2)

    # Read data
    in_data = sim.data[inp_p]

    # Check that the input value actually changes
    assert np.any(in_data[5] != in_data[6:])


if __name__=="__main__":
    test_white_signal()
