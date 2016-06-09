# Import modules
import csv
import numpy as np
import nengo
import nengo_spinnaker

# Import classes
from nengo.processes import WhiteNoise
from nengo_spinnaker.utils import profiling

# Import functions
from six import iteritems

# Parameters to profile
dimensions = 1
ensemble_size = 200

model = nengo.Network()
with model:
    # Create standard communication channel network with white noise input
    inp = nengo.Node(WhiteNoise(), label="inp")
    inp_p = nengo.Probe(inp)

    pre = nengo.Ensemble(ensemble_size, dimensions=dimensions, label="pre")
    pre_p = nengo.Probe(pre, synapse=0.01)
    nengo.Connection(inp, pre)
    
    post = nengo.Ensemble(ensemble_size, dimensions=dimensions, label="post")
    posts_p = nengo.Probe(post, synapse = 0.01)
    nengo.Connection(pre, post,
                     function=lambda x: np.random.random(dimensions))
    
    # Setup SpiNNaker-specific options to supply white noise from on
    # chip and profile the ensemble at the start of the channel
    nengo_spinnaker.add_spinnaker_params(model.config)
    model.config[inp].function_of_time = True
    model.config[pre].profile = True

# Create a SpiNNaker simulator and run model
sim = nengo_spinnaker.Simulator(model)
with sim:
    sim.run(10.0)

# Read profiler data
profiler_data = sim.profiler_data[pre]

# Open CSV file and create writer
with open("profile_communication_channel.csv", "wb") as csv_file:
    csv_writer = csv.writer(csv_file)

    # Loop through cores simulating pre-ensemble
    for i, (neuron_slice, core_data) in enumerate(iteritems(profiler_data)):
        # If this is the first row, write header with extra
        # columns for number of neurons and dimensions
        if i == 0:
            profiling.write_csv_header(core_data, csv_writer,
                                       ["Num neurons", "Dimensions"])

        # Write a row from the profiler data
        profiling.write_csv_row(core_data, 10.0, 0.001, csv_writer,
                                [neuron_slice[1] - neuron_slice[0], dimensions])