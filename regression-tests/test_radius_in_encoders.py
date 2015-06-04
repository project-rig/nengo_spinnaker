import logging
import nengo
import nengo_spinnaker


def test_radius_is_included_in_encoders():
    radius=1000.0

    with nengo.Network() as model:
        stim = nengo.Node(0.5)
        ens1 = nengo.Ensemble(100, 1)
        ens2 = nengo.Ensemble(100, 1, radius=radius)
        ens3 = nengo.Ensemble(100, 1)

        nengo.Connection(stim, ens1)
        nengo.Connection(ens1, ens2, transform=radius)
        nengo.Connection(ens2, ens3, transform=1/radius)

        p1 = nengo.Probe(ens1, synapse=0.05)
        p2 = nengo.Probe(ens2, synapse=0.05)
        p3 = nengo.Probe(ens3, synapse=0.05)

    sim = nengo_spinnaker.Simulator(model)
    with sim:
        sim.run(2.0)

    assert .4*radius <= sim.data[p2][-1] <= .6*radius
    assert .4 <= sim.data[p3][-1] <= .6


if __name__ == "__main__":
    test_radius_is_included_in_encoders()
