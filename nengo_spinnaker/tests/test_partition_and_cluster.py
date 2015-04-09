import pytest

from nengo_spinnaker import netlist as nl
from nengo_spinnaker import partition_and_cluster as pac
from nengo_spinnaker.keyspaces import KeyspaceContainer


def test_identify_clusters():
    """Test the correct assignation of clusters, modification of vertex slices
    and net keyspaces.
    """
    ksc = KeyspaceContainer()
    ks = ksc["nengo"]
    other_ks = ksc["test"]

    # Begin by constructing a set of vertices and vertex slices, then some nets
    v1 = nl.Vertex()
    v2 = nl.Vertex()
    v3 = nl.Vertex()  # Not sliced!
    v1.n_atoms = 5
    v2.n_atoms = 5

    # Vertex slices
    v1s = [nl.VertexSlice(v1, slice(n, n+1)) for n in range(v1.n_atoms)]
    v2s = [nl.VertexSlice(v2, slice(n, n+1)) for n in range(v2.n_atoms)]

    # Nets: v1 -> v2 with default keyspace
    v12_nets = [nl.Net(a, v2s[:], 0, ks()) for a in v1s]

    # Nets: v2 -> b1 with OTHER keyspace
    v21_nets = [nl.Net(a, v1s[:], 0, other_ks()) for a in v2s]

    # Nets: v3 -> v1 with default keyspace
    v31_net = nl.Net(v3, v1s[:], 0, ks())

    # Construct placements such that there is mixing of v1 and v2, and 3
    # clusters of v1 and 2 clusters of v2.
    placements = {
        # (0,0) has: v1[0], v1[1], v2[0]
        v1s[0]: (0, 0),
        v1s[1]: (0, 0),
        v2s[0]: (0, 0),
        # (0, 1) has: v1[2], v1[3]
        v1s[2]: (0, 1),
        v1s[3]: (0, 1),
        # (1, 0) has: v2[1:]
        v2s[1]: (1, 0),
        v2s[2]: (1, 0),
        v2s[3]: (1, 0),
        v2s[4]: (1, 0),
        # (1, 1) has: v1[4] and v3
        v1s[4]: (1, 1),
        v3: (1, 1),
    }

    # Assign the clusters
    nets = v12_nets + v21_nets + [v31_net]
    pac.identify_clusters(placements, nets)

    # Assert that the cluster IDs are appropriate
    assert v1s[0].cluster == v1s[1].cluster
    assert v1s[2].cluster == v1s[3].cluster
    assert v1s[0].cluster != v1s[2].cluster != v1s[4].cluster
    assert max(v1s[0].cluster, v1s[2].cluster, v1s[4].cluster) == 2
    assert v2s[0].cluster != v2s[1].cluster
    assert v2s[1].cluster == v2s[2].cluster == v2s[3].cluster == v2s[4].cluster
    assert max(v2s[0].cluster, v2s[1].cluster) == 1

    # Assert that the cluster ID was applied to nets originating from v1 (with
    # the default keyspace) but not the other net.
    for (source, net) in zip(v1s, v12_nets):
        assert source.cluster == net.keyspace.nengo_cluster

    for net in v21_nets:
        with pytest.raises(AttributeError):
            net.keyspace.nengo_cluster

    assert v31_net.keyspace.nengo_cluster == 0
