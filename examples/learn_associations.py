import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import nengo
import nengo_spinnaker

num_items = 5

d_key = 2
d_value = 4

spinnaker = True
record_encoders = True

rng = np.random.RandomState(seed=7)
keys = nengo.dists.UniformHypersphere(surface=True).sample(num_items, d_key, rng=rng)
values = nengo.dists.UniformHypersphere(surface=False).sample(num_items, d_value, rng=rng)

intercept = (np.dot(keys, keys.T) - np.eye(num_items)).flatten().max()
print("Intercept: %s" % intercept)

def cycle_array(x, period, dt=0.001):
    """Cycles through the elements"""
    i_every = int(round(period/dt))
    if i_every != period/dt:
        raise ValueError("dt (%s) does not divide period (%s)" % (dt, period))
    def f(t):
        i = int(round((t - dt)/dt))  # t starts at dt
        return x[(i/i_every)%len(x)]
    return f

# Model constants
n_neurons = 200
dt = 0.001
period = 0.3
T = period*num_items*2

# Model network
model = nengo.Network()
with model:

    # Create the inputs/outputs
    stim_keys = nengo.Node(output=cycle_array(keys, period, dt))
    stim_values = nengo.Node(output=cycle_array(values, period, dt))
    learning = nengo.Node(output=lambda t: -int(t>=T/2))
    recall = nengo.Node(size_in=d_value)

    # Create the memory
    memory = nengo.Ensemble(n_neurons, d_key, intercepts=[intercept]*n_neurons)

    # Learn the encoders/keys
    voja = nengo.Voja(post_tau=None, learning_rate=5e-2)
    conn_in = nengo.Connection(stim_keys, memory, synapse=None,
                               learning_rule_type=voja)
    nengo.Connection(learning, conn_in.learning_rule, synapse=None)

    # Learn the decoders/values, initialized to a null function
    conn_out = nengo.Connection(memory, recall, learning_rule_type=nengo.PES(1e-3),
                                function=lambda x: np.zeros(d_value))

    # Create the error population
    error = nengo.Ensemble(n_neurons, d_value)
    nengo.Connection(learning, error.neurons, transform=[[10.0]]*n_neurons,
                     synapse=None)

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(stim_values, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    # Setup probes
    p_keys = nengo.Probe(stim_keys, synapse=None)
    p_values = nengo.Probe(stim_values, synapse=None)
    p_learning = nengo.Probe(learning, synapse=None)
    p_error = nengo.Probe(error, synapse=0.005)
    p_recall = nengo.Probe(recall, synapse=None)

    if record_encoders:
        p_encoders = nengo.Probe(conn_in.learning_rule, 'scaled_encoders')

if spinnaker:
    sim = nengo_spinnaker.Simulator(model)
else:
    sim = nengo.Simulator(model)

sim.run(T)
t = sim.trange()

figure, axes = plt.subplots(4, sharex=True)

axes[0].set_title("Keys")
axes[0].plot(t, sim.data[p_keys])
axes[0].set_ylim((-1, 1))

axes[1].set_title("Values")
axes[1].plot(t, sim.data[p_values])
axes[1].set_ylim((-1, 1))

axes[2].set_title("Learning")
axes[2].plot(t, sim.data[p_learning])
axes[2].set_ylim((-1.2, 0.2))

axes[3].set_title("Value error")

train = t<=T/2
axes[3].plot(t[train], sim.data[p_error][train])

test = ~train
axes[3].plot(t[test], sim.data[p_recall][test] - sim.data[p_values][test])


if record_encoders:
    # Calculate encoder scale
    scale = (sim.model.params[memory].gain / memory.radius)[:, np.newaxis]

    # Create figure to show encoder animation
    figure, axis = plt.subplots()
    axis.set_xlim(-1.5, 1.5)
    axis.set_ylim(-1.5, 2)
    axis.set_aspect("equal")

    # Plot empty encoder and keys scatter
    scatter = axis.scatter([],[], label="Encoders")
    axis.scatter(keys[:, 0], keys[:, 1], c="red", s=150, alpha=0.6, label="Keys")

    # Generate legend
    axis.legend()

    def initfig():
        scatter.set_offsets([[]])
        return (scatter,)

    def updatefig(frame, encoders):
        xy = encoders[frame].copy() / scale
        scatter.set_offsets(xy)

        return (scatter,)

    # Play animation
    ani = animation.FuncAnimation(figure, updatefig, init_func=initfig, frames=range(int(T / (2 * dt))),
                                  fargs=(sim.data[p_encoders],), interval=5,
                                  blit=True, repeat=False)

plt.show()