import nengo
import nengo_spinnaker
import nengo_spinnaker.utils.learning
import numpy as np

n_neurons = 100
D_in = 4
D_out = 2
seed = 10
learning_rate = 1e-5
decimal_scale = 100

times = []

output_data = []
def output(t, x):
    times.append(t)
    output_data.append(x.copy()/decimal_scale)
    return x

stim_data = []
def stim_f(t, x):
    stim_data.append(x)

model = nengo.Network()
with model:

    #stim = nengo.Node(nengo.processes.WhiteSignal(high=0.5, rms=0.5, period=10),
    #                  size_out=D_in)
    phi = (np.sqrt(5)-1)/2
    rate = 2*np.pi
    stim = nengo.Node(lambda t: [np.sin(rate*t), np.sin(rate*t/phi),
                                 np.sin(rate*t/(phi*2)), np.sin(rate*t/(phi*3))])

    pre = nengo.Ensemble(n_neurons, D_in, seed=seed)
    post = nengo.Node(output, size_in=D_out, size_out=D_out)

    c = nengo.Connection(pre, post,
                         function=lambda x: np.zeros(D_out),
                         learning_rule_type=nengo.PES(learning_rate=learning_rate*decimal_scale))

    error = nengo.Node(lambda t, x: x, size_in=D_out, size_out=D_out)
    nengo.Connection(post, error, transform=1.0/decimal_scale)
    nengo.Connection(stim[0:2], error, transform=-1)
    nengo.Connection(stim, pre)
    nengo.Connection(error, c.learning_rule)

    stim_d = nengo.Node(stim_f, size_in=D_in, size_out=0)
    nengo.Connection(stim, stim_d)

sim = nengo_spinnaker.Simulator(model)

sim.run(10)

import pylab
output_data = np.array(output_data)
stim_data = np.array(stim_data)
pylab.plot(times, output_data[:,0], label='output')
pylab.plot(times, stim_data[:,0], label='stim')
pylab.legend(loc='best')

pylab.show()
