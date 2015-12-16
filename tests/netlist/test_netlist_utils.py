import pytest
from rig.place_and_route import Cores

from nengo_spinnaker.utils.keyspaces import KeyspaceContainer
from nengo_spinnaker.netlist import NMNet, Vertex, utils


def test_get_nets_for_placement():
    """Test that Rig nets can be generated to be used during the placement."""
    # Create the vertices
    a = object()
    b = object()
    c = object()
    d = object()
    e = object()

    # Create the nets
    ab_cd = NMNet([a, b], [c, d], 1.0, None)
    c_e = NMNet(c, e, 2.0, None)
    nets = [ab_cd, c_e]

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
        if net.sinks == [c, d]:
            assert net.source in set([a, b])
            assert net.weight == ab_cd.weight
        else:
            assert net.source is c
            assert net.sinks == [e]
            assert net.weight == c_e.weight

    assert seen_sources == set([a, b, c])


def test_get_nets_for_routing():
    """Test that Rig nets can be generated to be used during the routing."""
    # Create the vertices
    a = object()
    b = object()
    c = object()
    d = object()
    e = object()

    # Create the nets
    ab_cd = NMNet([a, b], [c, d], 1.0, None)
    cd_e = NMNet([c, d], e, 2.0, None)
    nets = [ab_cd, cd_e]

    # Create some placements:
    #  - a and b placed on the same chip
    #  - c and d placed on different chips
    placements = {a: (0, 0), b: (0, 0),
                  c: (1, 0), d: (0, 1), e: (1, 1)}

    # Create some resource requirements
    vertices_resources = {a: {Cores: 1},
                          b: {Cores: 2},
                          c: {Cores: 3},
                          d: {Cores: 4},
                          e: {Cores: 5}}

    # And create some sample allocations
    allocations = {a: {Cores: slice(1, 2)},
                   b: {Cores: slice(2, 4)},
                   c: {Cores: slice(1, 4)},
                   d: {Cores: slice(1, 5)},
                   e: {Cores: slice(1, 6)}}

    # Get the routing nets
    (routing_nets, extended_resources, extended_placements,
     extended_allocations, derived_nets) = utils.get_nets_for_routing(
        vertices_resources, nets, placements, allocations)

    # Check that the routing nets are sane
    assert len(routing_nets) == 3
    seen_cd_placements = set()
    expected_cd_placements = {placements[c], placements[d]}

    for net in routing_nets:
        # Check the sources is in the extended allocations and resources.
        assert extended_resources[net.source] == dict()
        assert extended_allocations[net.source] == dict()

        if net.sinks == [c, d]:
            assert net.weight == ab_cd.weight

            # Source should have been a and b, check the extended placement is
            # correct
            assert net.source not in placements
            assert extended_placements[net.source] == placements[a]
            assert extended_placements[net.source] == placements[b]

            # Check that the net is correctly identified in the derived nets
            # mapping.
            assert derived_nets[ab_cd][placements[a]] is net
        else:
            assert net.sinks == [e]
            assert net.weight == cd_e.weight

            # Source should have been one of c or d, check that the placement
            # is appropriate.
            assert net.source not in placements
            placement = extended_placements[net.source]
            assert placement in expected_cd_placements
            assert placement not in seen_cd_placements
            seen_cd_placements.add(placement)

            # Check that the net is correctly identified in the derived nets
            # mapping.
            assert derived_nets[cd_e][placement] is net

    assert seen_cd_placements == expected_cd_placements

    # Check that the original vertices are still present in the extended
    # placements.
    for v in [a, b, c, d, e]:
        assert extended_placements[v] == placements[v]
        assert extended_resources[v] == vertices_resources[v]
        assert extended_allocations[v] == allocations[v]


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
    resources = {v: {} for v in placements}
    allocations = {v: {} for v in placements}

    # Create a container for the keyspaces
    ksc = KeyspaceContainer()

    # Create the nets
    nets = [
        NMNet(vertex_A, vertex_B, 1.0, ksc["nengo"](connection_id=0)),
        NMNet(vertex_B, vertex_A, 2.0, ksc["nengo"](connection_id=1)),
        NMNet(vertex_A, vertex_A, 3.0, ksc["spam"]),
    ]

    # Identify groups
    groups = [set(vertex_A)]

    # Identify clusters
    utils.identify_clusters(groups, placements)
    assert vertex_B.cluster is None  # Not clustered

    # Get the routing nets
    _, _, _, _, derived_nets = utils.get_nets_for_routing(
        resources, nets, placements, allocations)

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
    resources = {v: {} for v in placements}
    allocations = {v: {} for v in placements}

    # Create a container for the keyspaces
    ksc = KeyspaceContainer()

    # Create the nets
    nets = [
        NMNet(vertex_A, vertex_B, 1.0, ksc["nengo"](connection_id=0)),
        NMNet(vertex_B, vertex_A, 2.0, ksc["nengo"](connection_id=1)),
        NMNet(vertex_A, vertex_A, 3.0, ksc["spam"]),
    ]

    # Identify groups
    groups = [set(vertex_A)]

    # Manually identify clusters (and do it such that it is inconsistent)
    vertex_A[0].cluster = 0
    vertex_A[1].cluster = 1
    vertex_A[2].cluster = 2
    vertex_A[3].cluster = 3

    # Get the routing nets
    _, _, _, _, derived_nets = utils.get_nets_for_routing(
        resources, nets, placements, allocations)

    # Get the net keyspaces
    with pytest.raises(AssertionError):
        utils.get_net_keyspaces(placements, derived_nets)
