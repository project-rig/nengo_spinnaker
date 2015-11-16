import pytest

from nengo_spinnaker.utils.keyspaces import KeyspaceContainer
from nengo_spinnaker.netlist import Net, Vertex, utils


def test_get_nets_for_placement():
    """Test that Rig nets can be generated to be used during the placement."""
    # Create the vertices
    a = [object(), object()]
    b = [object(), object()]
    c = object()

    # Create the nets
    ab = Net(a, b, 1.0, None)
    bc = Net(b[0], c, 2.0, None)
    nets = [ab, bc]

    # Get the placement nets
    placement_nets = list(utils.get_nets_for_placement(nets))

    # Assert that these nets are reasonable
    assert len(placement_nets) == 3
    seen_sources = set()

    for net in placement_nets:
        # Assert that we've not seen the source before
        assert net.source not in seen_sources
        seen_sources.add(net.source)

        # Check the source matches the sink
        if net.sinks == b:
            assert net.source in set(a)
            assert net.weight == ab.weight
        else:
            assert net.source is b[0]
            assert net.sinks == [c]
            assert net.weight == bc.weight


def test_get_nets_for_routing():
    """Test that Rig nets can be generated to be used during the routing."""
    # Create the vertices
    a = [object(), object()]
    b = [object(), object()]
    c = object()

    # Create the nets
    ab = Net(a, b, 1.0, None)
    bc = Net(b, c, 2.0, None)
    nets = [ab, bc]

    # Create some placements:
    #  - A[0, 1] placed on the same chip
    #  - B[0, 1] placed on different chips
    placements = {a[0]: (0, 0), a[1]: (0, 0),
                  b[0]: (1, 0), b[1]: (0, 1), c: (1, 1)}

    # Get the routing nets
    routing_nets, extended_placements, derived_nets = \
        utils.get_nets_for_routing(nets, placements)

    # Check that the routing nets are sane
    assert len(routing_nets) == 3
    seen_b_placements = set()
    expected_b_placements = {placements[b[0]], placements[b[1]]}

    for net in routing_nets:
        if net.sinks == b:
            assert net.weight == ab.weight

            # Source should have been A, check the extended placement is
            # correct
            assert net.source not in placements
            assert extended_placements[net.source] == placements[a[0]]

            # Check that the net is correctly identified in the derived nets
            # mapping.
            assert derived_nets[ab][placements[a[0]]] is net
        else:
            assert net.sinks == [c]
            assert net.weight == bc.weight

            # Source should have been one of the Bs, check that the placement
            # is appropriate.
            assert net.source not in placements
            placement = extended_placements[net.source]
            assert placement in expected_b_placements
            assert placement not in seen_b_placements
            seen_b_placements.add(placement)

            # Check that the net is correctly identified in the derived nets
            # mapping.
            assert derived_nets[bc][placement] is net

    # Check that the original vertices are still present in the extended
    # placements.
    for v in [a[0], a[1], b[0], b[1], c]:
        assert extended_placements[v] == placements[v]


def test_identify_clusters():
    """Test the correct assignation of clusters."""
    # Create two vertices
    vertex_A = [Vertex() for _ in range(4)]
    vertex_B = [Vertex() for _ in range(2)]

    # Create placements such that vertex A and B fall on two chips and A[0] and
    # A[1] are on the same chip.
    placements = {
        vertex_A[0]: (0, 0), vertex_A[1]: (0, 0),
        vertex_A[2]: (0, 1), vertex_A[3]: (0, 1),
        vertex_B[0]: (0, 0), vertex_B[1]: (1, 0),
    }

    # Identify groups
    groups = [set(vertex_A), set(vertex_B)]

    # Identify clusters
    utils.identify_clusters(groups, placements)

    # Ensure that appropriate cluster indices are assigned to all vertices
    assert vertex_A[0].cluster == vertex_A[1].cluster  # 1 cluster of A
    assert vertex_A[2].cluster == vertex_A[3].cluster  # The other
    assert vertex_A[0].cluster != vertex_A[2].cluster  # Different IDs

    assert vertex_B[0].cluster != vertex_B[1].cluster  # Different IDs


def test_get_net_keyspaces():
    """Test the correct specification of keyspaces for nets."""
    # Create the vertices
    vertex_A = [Vertex() for _ in range(4)]
    vertex_B = Vertex()

    # Create placements such that vertex A and B fall on two chips and A[0] and
    # A[1] are on the same chip.
    placements = {
        vertex_A[0]: (0, 0), vertex_A[1]: (0, 0),
        vertex_A[2]: (0, 1), vertex_A[3]: (0, 1),
        vertex_B: (0, 0)
    }

    # Create a container for the keyspaces
    ksc = KeyspaceContainer()

    # Create the nets
    nets = [
        Net(vertex_A, vertex_B, 1.0, ksc["nengo"](object=0, connection=0)),
        Net(vertex_B, vertex_A, 2.0, ksc["nengo"](object=1, connection=0)),
        Net(vertex_A, vertex_A, 3.0, ksc["spam"]),
    ]

    # Identify groups
    groups = [set(vertex_A)]

    # Identify clusters
    utils.identify_clusters(groups, placements)
    assert vertex_B.cluster is None  # Not clustered

    # Get the routing nets
    _, _, derived_nets = utils.get_nets_for_routing(nets, placements)

    # Get the net keyspaces
    net_keyspaces = utils.get_net_keyspaces(placements, derived_nets)

    # Check the net keyspaces are correct
    # A -> B
    for xy, vertex in [((0, 0), vertex_A[0]), ((0, 1), vertex_A[2])]:
        net = derived_nets[nets[0]][xy]
        cluster = vertex.cluster
        assert net_keyspaces[net] == nets[0].keyspace(cluster=cluster)

    # B -> A
    net = derived_nets[nets[1]][(0, 0)]
    assert net_keyspaces[net] == nets[1].keyspace(cluster=0)

    # A -> A
    for xy in [(0, 0), (0, 1)]:
        net = derived_nets[nets[2]][xy]
        assert net_keyspaces[net] == nets[2].keyspace  # No change


def test_get_net_keyspaces_fails_for_inconsistent_cluster():
    """Test specification of keyspaces for nets fails in the case that
    inconsistent cluster IDs are assigned (this is unlikely to happen unless a
    Net somehow ends up having two different Nengo objects in its source
    list)."""
    # Create the vertices
    vertex_A = [Vertex() for _ in range(4)]
    vertex_B = Vertex()

    # Create placements such that vertex A and B fall on two chips and A[0] and
    # A[1] are on the same chip.
    placements = {
        vertex_A[0]: (0, 0), vertex_A[1]: (0, 0),
        vertex_A[2]: (0, 1), vertex_A[3]: (0, 1),
        vertex_B: (0, 0)
    }

    # Create a container for the keyspaces
    ksc = KeyspaceContainer()

    # Create the nets
    nets = [
        Net(vertex_A, vertex_B, 1.0, ksc["nengo"](object=0, connection=0)),
        Net(vertex_B, vertex_A, 2.0, ksc["nengo"](object=1, connection=0)),
        Net(vertex_A, vertex_A, 3.0, ksc["spam"]),
    ]

    # Identify groups
    groups = [set(vertex_A)]

    # Manually identify clusters (and do it such that it is inconsistent)
    vertex_A[0].cluster = 0
    vertex_A[1].cluster = 1
    vertex_A[2].cluster = 2
    vertex_A[3].cluster = 3

    # Get the routing nets
    _, _, derived_nets = utils.get_nets_for_routing(nets, placements)

    # Get the net keyspaces
    with pytest.raises(AssertionError):
        utils.get_net_keyspaces(placements, derived_nets)
