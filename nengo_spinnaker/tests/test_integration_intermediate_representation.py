import nengo
import numpy as np

from nengo_spinnaker import ensemble as ns_ens
from nengo_spinnaker import intermediate_representation as ir


def test_ensemble():
    """A functional test that looks at constant value Nodes, probes, global
    inhibition connections.
    """
    with nengo.Network() as net:
        a = nengo.Node(output=0.5)
        b = nengo.Ensemble(100, 1)
        c = nengo.Ensemble(100, 5)
        d = nengo.Ensemble(100, 4)

        p_spikes = nengo.Probe(b.neurons)
        p_value = nengo.Probe(d)

        conn_ab = nengo.Connection(a, b)  # Optimised out

        # Global inhibition connections
        conn_bc = nengo.Connection(b, c.neurons, transform=[[-1]]*c.n_neurons)

        conn_cd = nengo.Connection(c[:4], d)  # Normal

    # Build the intermediate representation
    irn = ir.IntermediateRepresentation.from_objs_conns_probes(
        net.all_objects, net.connections, net.probes)

    # Check that b, c, and d are intermediate ensembles
    assert isinstance(irn.object_map[b], ns_ens.IntermediateEnsemble)
    assert irn.object_map[b].local_probes == [p_spikes]
    assert irn.object_map[b].direct_input == 0.5

    assert isinstance(irn.object_map[c], ns_ens.IntermediateEnsemble)
    assert irn.object_map[c].local_probes == list()
    assert np.all(irn.object_map[c].direct_input == np.zeros(5))

    assert isinstance(irn.object_map[d], ns_ens.IntermediateEnsemble)
    assert irn.object_map[d].local_probes == list()
    assert np.all(irn.object_map[d].direct_input == np.zeros(4))

    # Check that conn a->b was optimised out
    assert conn_ab not in irn.connection_map

    # Check that conn b->c was identified as global inhibition
    assert (irn.connection_map[conn_bc].sinks[0].port is
            ir.InputPort.global_inhibition)

    # Check that conn c->d was left as normal
    assert (irn.connection_map[conn_cd].sinks[0].port is
            ir.InputPort.standard)

    # The probe on d should be in the object map
    assert p_value in irn.object_map

    # There should be a connection d->p_value
    conn = irn.extra_connections[0]
    assert conn.source.object is irn.object_map[d]
    assert conn.source.port is ir.OutputPort.standard
    assert conn.sinks[0].object is irn.object_map[p_value]
    assert conn.sinks[0].port is ir.InputPort.standard


def test_spike_transmission():
    """Test that neuron-neuron connections result in single nets with sane
    parameters.
    """
    with nengo.Network() as net:
        a = nengo.Ensemble(300, 1)
        b = nengo.Ensemble(100, 1)
        c = nengo.Ensemble(500, 1)

        # Should become a single net!
        conn_ab = nengo.Connection(a.neurons[:100], b.neurons)
        conn_ac = nengo.Connection(a.neurons, c.neurons[100:400])

    # Build the intermediate representation
    irn = ir.IntermediateRepresentation.from_objs_conns_probes(
        [a, b, c], [conn_ab, conn_ac], [])

    # There should only be 1 Net with two sinks
    assert irn.connection_map[conn_ab] is irn.connection_map[conn_ac]
    net = irn.connection_map[conn_ab]
    assert net.source.object is irn.object_map[a]
    assert net.source.port is ir.OutputPort.neurons

    for sink in net.sinks:
        assert sink.object in [irn.object_map[b], irn.object_map[c]]
        assert sink.port is ir.InputPort.neurons

    assert net.weight == a.n_neurons
