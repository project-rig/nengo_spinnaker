import nengo
import numpy as np

from nengo_spinnaker import Simulator


def test_realtime():
    values1 = []
    values2 = []

    model = nengo.Network()
    with model:
        def f1(t):
            if len(values1) > 0:
                assert t > values1[-1]
            values1.append(t)
            return np.sin(t)

        def f2(t, x):
            if len(values2) > 0:
                assert t > values2[-1]
            values2.append(t)
            return -x

        node1 = nengo.Node(f1, size_out=1)
        node2 = nengo.Node(f2, size_in=1, size_out=1)
        nengo.Connection(node1, node2, synapse=None)
        ens = nengo.Ensemble(n_neurons=50, dimensions=1)
        nengo.Connection(node2, ens, synapse=0.01)

    sim = Simulator(model)
    with sim:
        sim.run(5)

    assert len(values1) == len(values2)
    assert 4.9 < values1[-1] < 5.1
    assert 4.9 < values2[-1] < 5.1


if __name__ == "__main__":
    test_realtime()
