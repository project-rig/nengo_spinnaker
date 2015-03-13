import nengo
from nengo.utils.builder import objs_and_connections

from nengo_spinnaker import builder


def test_separate_networks():
    """Test the correct partitioning of a Network into two networks comprising
    elements to be simulated on SpiNNaker and elements to simulate on the host
    PC.
    """
    net = nengo.Network()
    with net:
        a = nengo.Node(lambda t: t, size_out=1, size_in=0)  # Source Node
        b = nengo.Ensemble(233, 3)
        c = nengo.Ensemble(100, 3)
        d = nengo.Node(lambda t, x : [0, 0, 0], size_in=3)  # SpiNNaker sink
        e = nengo.Node(lambda t, x : None, size_in=3)  # Sink Node

        a.label = "A"
        b.label = "B"
        c.label = "C"
        d.label = "D"
        e.label = "E"

        c1 = nengo.Connection(a, b[0])  # Node -> Ensemble
        c2 = nengo.Connection(b, c)  # Ensemble -> Ensemble
        c3 = nengo.Connection(c, d)  # Ensemble -> Node
        c4 = nengo.Connection(d, e)  # Node -> Node

    # Test separate networks
    spinn_net, host_net = builder.separate_networks(*objs_and_connections(net))

    # The SpiNNaker network should contain a, b, c and d
    (spinn_objs, spinn_conns) = spinn_net
    assert set(spinn_objs) == {a, b, c, d}
    assert len(spinn_conns) == 3

    for conn in spinn_conns:
        if conn.pre_obj is a:
            assert conn.post_obj is b
        elif conn.pre_obj is b:
            assert conn.post_obj is c
        else:
            assert conn.pre_obj is c and conn.post_obj is d

    # The host network should contain a, c and d and their related input/output
    # Nodes.
    (host_objs, host_conns) = host_net
    assert len(host_objs) == 3 + 2  # a, c, d, e, a->Out & In->c
    assert len(host_conns) == 1 + 2  # a -> a->Out, In->c -> c, c -> d

    for obj in host_objs:
        if isinstance(obj, builder.PCToBoardNode):
            assert obj.node is a
        elif isinstance(obj, builder.PCFromBoardNode):
            assert obj.node is d
        else:
            assert isinstance(obj, nengo.Node)
            assert obj in [a, c, d, e]

    for conn in host_conns:
        if conn.pre_obj is a:
            assert conn.post_obj.node is a
        elif conn.post_obj is d:
            assert conn.pre_obj.node is d
        else:
            assert conn.pre_obj is d and conn.post_obj is e
