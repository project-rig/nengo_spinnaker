import nengo
import nengo_spinnaker
import nengo_spinnaker.utils.learning

model = nengo.Network()
with model:
    pre = nengo.Ensemble(100, 1, seed=1)
    post = nengo.Ensemble(50, 1)

    c = nengo.Connection(pre, post,
                         function=lambda x: 0,
                         learning_rule_type=nengo.PES())

    target = nengo.Node(0.5)

    error = nengo.Ensemble(n_neurons=50, dimensions=1)

    nengo.Connection(target, error, transform=-1)
    nengo.Connection(post, error, transform=1)

    nengo.Connection(error, c.learning_rule)

sim = nengo_spinnaker.Simulator(model)


d_start = nengo_spinnaker.utils.learning.get_learnt_decoders(sim, pre)

sim.async_run_forever()
import timeit
import time
start = timeit.default_timer()
while timeit.default_timer() - start < 10.0:
    sim.async_update()
    time.sleep(0.001)

print d_start
d_end = nengo_spinnaker.utils.learning.get_learnt_decoders(sim, pre)
print d_end

sim.close()


model2 = nengo.Network()
with model2:
    pre = nengo.Ensemble(100, 1, seed=1)
    post = nengo.Ensemble(50, 1)

    nengo.Connection(pre, post,
                     solver=nengo_spinnaker.utils.learning.FixedSolver(d_end))
    p = nengo.Probe(post, synapse=0.03)

sim2 = nengo_spinnaker.Simulator(model2)
sim2.run(1)

print('re-used')
print sim2.data[p][-10:]

