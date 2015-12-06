import collections
import rig.netlist
from six import iteritems, itervalues

from nengo_spinnaker.utils.keyspaces import is_nengo_keyspace


def get_nets_for_placement(nets):
    """Convert a list of N:M nets into a list of Rig nets suitable for
    performing placement.

    Parameters
    ----------
    nets : [:py:class:`~nengo_spinnaker.netlist.NMNet`, ...]
        N:M nets to convert

    Yields
    ------
    :py:class:`~rig.netlist.Net`
        1:M net suitable for use with Rig for placement purposes.
    """
    # For each source in each net create a new Rig net
    for net in nets:
        for source in net.sources:
            yield rig.netlist.Net(source, net.sinks, net.weight)


def get_nets_for_routing(resources, nets, placements, allocations):
    """Convert a list of N:M nets into a list of Rig nets suitable for
    performing routing.

    Parameters
    ----------
    resources : {vertex: {resource: requirement}, ...}
    nets : [:py:class:`~nengo_spinnaker.netlist.NMNet`, ...]
    placements : {vertex: (x, y), ...}
    allocations : {vertex: {resource: :py:class:`slice`}, ...}

    Returns
    -------
    [:py:class:`~rig.netlist.Net`, ...]
        1:M net suitable for use with Rig for routing purposes.
    {vertex: {resource: requirement}, ...}
        An extended copy of the resources dictionary which must be used when
        performing routing with the returned nets.
    {vertex: (x, y), ...}
        An extended copy of the placements dictionary which must be used when
        performing routing with the returned nets.
    {vertex: {resource: :py:class:`slice`}, ...}
        An extended copy of the allocations dictionary which must be used when
        performing routing with the returned nets.
    {:py:class:`~nengo_spinnaker.netlist.NMNet`:
            {(x, y): :py:class:`~rig.netlist.Net`, ...}, ...}
        Map from original nets to co-ordinates and the derived nets which
        originate from them.
    """
    routing_nets = list()  # Nets with which to perform routing
    extended_resources = dict(resources)  # New requirements will be added
    extended_placements = dict(placements)  # New placements will be added
    extended_allocations = dict(allocations)  # New allocations will be added
    derived_nets = collections.defaultdict(dict)  # {Net: {placement: rig.Net}}

    # For each Net build a set of all the co-ordinates from which the net now
    # originates.
    for net in nets:
        start_placements = set(placements[v] for v in net.sources)

        # For each of these co-ordinates create a new Rig Net with a new source
        # vertex placed at the given co-ordinate.
        for placement in start_placements:
            # Create a new source vertex and place it at the given placement
            vertex = object()
            extended_placements[vertex] = placement
            extended_resources[vertex] = dict()
            extended_allocations[vertex] = dict()

            # Create a new Rig Net using the new start vertex; add the new Net
            # to the dictionary of derived nets and the list of nets with which
            # to perform routing.
            new_net = rig.netlist.Net(vertex, net.sinks, net.weight)
            routing_nets.append(new_net)
            derived_nets[net][placement] = new_net

    return (routing_nets, extended_resources, extended_placements,
            extended_allocations, derived_nets)


def identify_clusters(groups, placements):
    """Group vertices in clusters (based on chip assignation).

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
    keys which indicates from which chip the packet was sent - in this case a
    single bit will suffice with packets from `(A)` using a key with the bit
    not set and packets from `(B)` setting the bit.

    This method will assign a unique ID to each cluster of vertices (e.g.,
    `(A)` and `(B)`) by storing the index in the `cluster` attribute of each
    subvertex. Later this ID can be used in the keyspace of all nets
    originating from the cluster.

    Parameters
    ----------
    groups : [{Vertex, ...}, ...]
        List of groups of vertices. This is used to identify objects which may
        be joined into a cluster.
    placements : {vertex: (x, y), ...}
    """
    # Perform clustering for each group of vertices in turn.
    for group in groups:
        # Build a map of placements to vertices, this will be used to identify
        # clusters.
        clusters = collections.defaultdict(list)  # Map co-ordinate to cluster

        # Add each vertex in the group to a cluster.
        for vertex in group:
            coord = placements[vertex]
            clusters[coord].append(vertex)

        # Assign a unique ID to each cluster in the group
        for cluster_id, cluster in enumerate(itervalues(clusters)):
            # Store the cluster ID in each vertex in the cluster
            for vertex in cluster:
                vertex.cluster = cluster_id


def get_net_keyspaces(placements, derived_nets):
    """Get a map from the nets used during routing to the keyspaces (NOT the
    keys and masks) that should be used when building routing tables.

    Cluster IDs will be applied to any nets which used the default Nengo
    keyspace.

    Parameters
    ----------
    placements : {vertex: (x, y), ...}
    derived_nets : {:py:class:`~nengo_spinnaker.netlist.NMNet`:
                    {(x, y): :py:class:`~rig.netlist.Net`, ...}, ...}
        Map from original nets to co-ordinates and the derived nets which
        originate from them as, returned by :py:func:`~.get_routing_nets`.

    Returns
    -------
    {net: keyspace, ...}
        A map from nets to :py:class:`~rig.bitfield.BitField`s that can later
        be used to generate routing tables.
    """
    net_keyspaces = dict()  # Map from derived nets to keyspaces

    for original_net, placement_nets in iteritems(derived_nets):
        for placement, net in iteritems(placement_nets):
            # If the keyspace is the default Nengo keyspace then add a cluster
            # ID, otherwise just store the keyspace as is.
            if is_nengo_keyspace(original_net.keyspace):
                # Get all the cluster IDs assigned to vertices with the given
                # placement (there should only be one cluster ID, if there are
                # more it would imply that multiple Nengo objects ended up in
                # the sources for a given Net and it is an error from which we
                # cannot recover).
                cluster_ids = set(vx.cluster for vx in original_net.sources
                                  if placements[vx] == placement)
                assert len(cluster_ids) == 1, "Inconsistent cluster IDs"
                cluster_id = next(iter(cluster_ids)) or 0  # Get the ID

                # Store the keyspace with the cluster ID attached
                net_keyspaces[net] = original_net.keyspace(cluster=cluster_id)
            else:
                # Store the keyspace as is
                net_keyspaces[net] = original_net.keyspace

    return net_keyspaces
