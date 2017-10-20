import nengo
import nengo_spinnaker
import nengo_spinnaker.utils.learning
import numpy as np
import timeit

n_neurons = 100
D_in = 4
D_out = 2
seed = 10
learning_rate = 1e-5
decimal_scale = 100
spinn = True
T = 10
test2 = False

times = []
times3 = []

output_data = []
def output(t, x):
    times.append(t)
    output_data.append(x.copy()/decimal_scale)
    return x
output3_data = []
def output3(t, x):
    times3.append(t)
    output3_data.append(x.copy()/decimal_scale)
    return x


stim_data = []
def stim_f(t, x):
    stim_data.append(x)

model = nengo.Network()
with model:

    #stim = nengo.Node(nengo.processes.WhiteSignal(high=0.5, rms=0.5, period=10),
    #                  size_out=D_in)
    phi = (np.sqrt(5)-1)/2
    phi = 1.0
    rate = 2*np.pi
    stim = nengo.Node(lambda t: [np.sin(rate*t), np.sin(rate*t/phi),
                                 np.sin(rate*t/(phi*2)), np.sin(rate*t/(phi*3))])

    pre = nengo.Ensemble(n_neurons, D_in, seed=seed)
    post_ens = nengo.Ensemble(n_neurons=500, dimensions=D_out, radius=decimal_scale)
    post = nengo.Node(output, size_in=D_out, size_out=D_out)
    nengo.Connection(post_ens, post)

    dummy_dec = np.zeros((n_neurons, D_out))
    #dummy_dec[:,0] = np.arange(n_neurons)
    #dummy_dec[:,-1] = -np.arange(n_neurons)

    c = nengo.Connection(pre, post_ens,
                         function=lambda x: np.zeros(D_out),
                         solver=nengo_spinnaker.utils.learning.FixedSolver(dummy_dec),
                         learning_rule_type=nengo.PES(learning_rate=learning_rate*decimal_scale))

    error = nengo.Node(lambda t, x: x, size_in=D_out, size_out=D_out)
    nengo.Connection(post, error, transform=1.0/decimal_scale)
    nengo.Connection(stim[0:D_out], error, transform=-1)
    nengo.Connection(stim, pre)
    nengo.Connection(error, c.learning_rule)

    stim_d = nengo.Node(stim_f, size_in=D_in, size_out=0)
    nengo.Connection(stim, stim_d)



if spinn:
    sim = nengo_spinnaker.Simulator(model)
    pre_dec = nengo_spinnaker.utils.learning.get_learnt_decoders(sim, pre)
    #pre_dec = pre_dec.flatten().reshape((D_out, n_neurons)).T

    sim.async_run_forever()
    start = timeit.default_timer()
    while timeit.default_timer() - start < T:
        sim.async_update()
    sim.close()
else:
    sim = nengo.Simulator(model)
    pre_dec = sim.signals[sim.model.sig[c]['weights']].T
    sim.run(T)

if spinn:
    dec = nengo_spinnaker.utils.learning.get_learnt_decoders(sim, pre)
    #dec = dec.flatten().reshape((D_out, n_neurons)).T
else:
    dec = sim.signals[sim.model.sig[c]['weights']].T

print(pre_dec)
print(dec)
output_data = np.array(output_data)
stim_data = np.array(stim_data)

model2 = nengo.Network()
with model2:
    pre2 = nengo.Ensemble(n_neurons, D_in, seed=seed,
                         encoders=sim.model.params[pre].encoders,
                         bias=sim.model.params[pre].bias,
                         gain=sim.model.params[pre].gain,
                         )

sim2 = nengo.Simulator(model2)
_, a = nengo.utils.ensemble.tuning_curves(pre2, sim2, inputs=stim_data)
out2 = np.dot(a, dec) / decimal_scale

if test2:
    model3 = nengo.Network()
    with model3:
        phi = (np.sqrt(5)-1)/2
        rate = 2*np.pi
        stim = nengo.Node(lambda t: [np.sin(rate*t), np.sin(rate*t/phi),
                                     np.sin(rate*t/(phi*2)), np.sin(rate*t/(phi*3))])

        pre = nengo.Ensemble(n_neurons, D_in, seed=seed)
        post = nengo.Node(output3, size_in=D_out, size_out=D_out)

        nengo.Connection(pre, post,
                         function=lambda x: np.zeros(D_out),
                         solver=nengo_spinnaker.utils.learning.FixedSolver(dec)
                         )
        nengo.Connection(stim, pre)

    if spinn:
        sim = nengo_spinnaker.Simulator(model3)

        sim.async_run_forever()
        start = timeit.default_timer()
        while timeit.default_timer() - start < T:
            sim.async_update()
        sim.close()
    else:
        sim = nengo.Simulator(model3)
        sim.run(T)
        sim.close()

output3_data = np.array(output3_data)


print('steps=', len(times))
print('dt=', float(T)/len(times))

import pylab
pylab.plot(times, output_data[:,0], color='b', label='output')
pylab.plot(times, stim_data[:,0], color='g', label='stim')
pylab.plot(times, out2[:,0], color='r', label='output2')
if test2:
    pylab.plot(times3, output3_data[:,0], color='k', label='output3')
pylab.legend(loc='best')

pylab.show()
