"""Tools for partitioning large vertices and then dealing with clusters of the
resulting subvertices.
"""
import collections
from six import iteritems, itervalues

from .keyspaces import is_nengo_keyspace
from .netlist import VertexSlice


def identify_clusters(placed_vertices, nets):
    """Group vertex slices in clusters (based on chip assignation) and assign
    the cluster ID to nets which originate from these subvertices.

    A vertex may be partitioned into a number of vertex slices which are placed
    onto the cores of two SpiNNaker chips.  The vertex slices on these chips
    form two clusters; packets from these clusters need to be differentiated in
    order to route packets correctly.  For example::

       +--------+                      +--------+
       |        | ------- (a) ------>  |        |
       |   (A)  |                      |   (B)  |
       |        | <------ (b) -------  |        |
       +--------+                      +--------+

    Packets traversing `(a)` need to be differentiated from packets traversing
    `(b)`.  This can be done by including an additional field in the packet
    keys which indicates which chip the packet was sent from - in this case a
    single bit will suffice with packets from `(A)` using a key with the bit
    not set and packets from `(B)` setting the bit.

    This method will assign a unique ID to each cluster of vertex slices (e.g.,
    `(A)` and `(B)`) by assigning the index to the `cluster` attribute of each
    subvertex and will ensure the same cluster ID is present in the keyspace of
    all nets originating from the cluster (and using the standard Nengo
    keyspace).

    Parameters
    ----------
    placed_vertices : {vertex: (x, y), ...}
        Vertex placements (as produced by
        :py:func:`rig.place_and_route.place`).
    nets : [:py:class:`nengo_spinnaker.netlist.Net`, ...]
    """
    class Cluster(object):
        """Internal representation of a cluster."""
        def __init__(self):
            self.vertices = list()
            self.nets = list()

    # Build up the clusters of vertices and nets
    # vertices -> co-ordinate -> Cluster
    vertices_clusters = collections.defaultdict(
        lambda: collections.defaultdict(Cluster))

    for (vertex, coord) in iteritems(placed_vertices):
        # Add the vertex to the cluster dictionary iff it is a vertex slice
        if isinstance(vertex, VertexSlice):
            vertices_clusters[vertex.vertex][coord].vertices.append(vertex)

    for net in nets:
        # If the net uses the default keyspace
        if is_nengo_keyspace(net.keyspace):
            if isinstance(net.source, VertexSlice):
                # Then include it in the cluster if its originating vertex is a
                # vertex slice
                vertex = net.source.vertex
                coord = placed_vertices[net.source]
                vertices_clusters[vertex][coord].nets.append(net)
            else:
                # Otherwise it can have the cluster ID set to 0
                net.keyspace = net.keyspace(nengo_cluster=0)

    # Iterate through the clusters of subvertices
    for clusters in itervalues(vertices_clusters):
        for cluster_id, cluster in enumerate(itervalues(clusters)):
            # Assign the cluster ID to the vertex slices
            for vertex in cluster.vertices:
                vertex.cluster = cluster_id

            # Assign the cluster ID to the nets
            for net in cluster.nets:
                net.keyspace = net.keyspace(nengo_cluster=cluster_id)
