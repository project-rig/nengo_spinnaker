import numpy as np
import matplotlib.pyplot as plt

import nengo
import nengo_spinnaker
from nengo.processes import WhiteSignal

dimensions = 4
spinnaker = True

model = nengo.Network()
with model:
    num_neurons = dimensions * 30
    inp = nengo.Node(WhiteSignal(num_neurons, high=5), size_out=dimensions)
    pre = nengo.Ensemble(num_neurons, dimensions=dimensions)
    nengo.Connection(inp, pre)
    post = nengo.Ensemble(num_neurons, dimensions=dimensions)
    conn = nengo.Connection(pre, post, function=lambda x: np.random.random(dimensions))
    inp_p = nengo.Probe(inp)
    pre_p = nengo.Probe(pre, synapse=0.01)
    post_p = nengo.Probe(post, synapse=0.01)

    error = nengo.Ensemble(num_neurons, dimensions=dimensions)
    error_p = nengo.Probe(error, synapse=0.03)

    # Error = actual - target = post - pre
    nengo.Connection(post, error)
    nengo.Connection(pre, error, transform=-1)

    # Add the learning rule to the connection
    conn.learning_rule_type = nengo.PES()

    # Connect the error into the learning rule
    nengo.Connection(error, conn.learning_rule)

if spinnaker:
    sim = nengo_spinnaker.Simulator(model)
else:
    sim = nengo.Simulator(model)

sim.run(20.0)

figure, axes = plt.subplots(dimensions + 1, sharex=True)

for a, d in zip(axes, range(dimensions)):
    a.plot(sim.trange(), sim.data[inp_p].T[d], c='k', label='Input')
    a.plot(sim.trange(), sim.data[pre_p].T[d], c='b', label='Pre')
    a.plot(sim.trange(), sim.data[post_p].T[d], c='r', label='Post')

    a.set_ylabel("Dimensions 1")
    a.legend()

axes[dimensions].plot(sim.trange(), sim.data[error_p], c='b')
axes[dimensions].set_ylim(-1, 1)
axes[dimensions].set_ylabel("Error")
#axes[dimensions].legend(("Error[0]", "Error[1]"), loc='best');

plt.show()