import logging
from collections import defaultdict, deque
from six import iteritems, iterkeys, itervalues

logger = logging.getLogger(__name__)


def allocate_signal_keyspaces(signal_routes, signal_id_constraints, keyspaces):
    # Filter signals and routes to be only those without a keyspace
    signal_routes = {signal: routes for signal, routes in
                     iteritems(signal_routes) if
                     signal.keyspace is None}

    # Get unique identifiers for each signal
    signal_ids = assign_mn_net_ids(signal_routes, signal_id_constraints)

    # Assign keyspaces to the signals
    for signal, i in iteritems(signal_ids):
        assert signal.keyspace is None
        signal.keyspace = keyspaces["nengo"](connection_id=i)

        # Expand the keyspace to fit the required indices
        signal.keyspace(index=signal.width - 1)

    if (signal_ids):
        logger.info("%u signals assigned %u IDs", len(signal_ids),
                    max(itervalues(signal_ids)) + 1)


def assign_mn_net_ids(nets_routes, prior_constraints=None):
    """Assign identifiers to multiple-source multicast nets such that
    equivalent identifiers are never assigned to pairs of multi-source nets
    which converge and subsequently diverge.

    Prior constraints can be provided so that information not included in the
    routing tree(s) can be taken into account (in the case of nengo_spinnaker
    this allows multi-source nets which target different filters in the same
    object to be given different identifiers).

    Parameters
    ----------
    nets_routes : {net: [RoutingTree, ...], ...}
        Dictionary mapping multi-source nets to the routing trees which
        describe them.
    prior_constraints : {net: {net, ...}, ...}
        Existing constraints to include within the net graph presented as an
        adjacency list.

    Returns
    -------
    {net: int}
        Mapping from each multiple source net to a valid identifier.
    """
    return colour_graph(
        build_mn_net_graph(nets_routes, prior_constraints)
    )


def build_mn_net_graph(nets_routes, prior_constraints=None):
    """Build a graph the nodes of which represent multicast nets and the edges
    of which represent constraints upon which nets may share keys.

    Parameters
    ----------
    nets_routes : {net: [RoutingTree, ...], ...}
        Dictionary mapping multi-source nets to the routing trees which
        describe them.
    prior_constraints : {net: {net, ...}, ...}
        Existing constraints to include within the net graph presented as an
        adjacency list.

    Returns
    -------
    {Net: {Net, ...}, ...}
        An adjacency list representation of a graph where the presence of an
        edge indicates that two multicast nets may not share a routing key.
    """
    # The different sets of routes from a chip indicate nets which cannot share
    # a routing key, this is indicated by creating an edge between those `nets'
    # in the net graph.
    net_graph = {net: set() for net in iterkeys(nets_routes)}

    # Construct a map from chips to unique sets of routes from that chip to
    # nets which take that route whilst simultaneously using this data
    # structure to fill in the above graph of which nets may not share a
    # routing key.
    chip_route_nets = defaultdict(lambda: defaultdict(deque))
    for net, trees in iteritems(nets_routes):
        for tree in trees:
            for _, chip, routes in tree.traverse():
                # Add this net to the set of nets who take this route at this
                # point.
                route = 0x0
                for r in routes:
                    route |= (1 << r)

                chip_route_nets[chip][route].append(net)

                # Add constraints to the net graph dependent on which nets take
                # different routes at this point.
                routes_from_chip = chip_route_nets[chip]
                for other_route, nets in iteritems(routes_from_chip):
                    if other_route != route:
                        for other_net in nets:
                            # This net cannot share an identifier with any of
                            # the nets who take a different route at this
                            # point.
                            if net != other_net:
                                net_graph[net].add(other_net)
                                net_graph[other_net].add(net)

    # Add any prior constraints into the net graph (doing so in such a way that
    # ensures that the prior constraints are undirected).
    if prior_constraints is not None:
        for u, vs in iteritems(prior_constraints):
            for v in vs:
                net_graph[u].add(v)
                net_graph[v].add(u)

    return net_graph


def assign_cluster_ids(operator_vertices, signal_routes, placements):
    """Assign identifiers to the clusters of vertices owned by each operator to
    the extent that multicast nets belonging to the same signal which originate
    at multiple chips can be differentiated if required.

    An operator may be partitioned into a number of vertices which are placed
    onto the cores of two SpiNNaker chips.  The vertices on these chips form
    two clusters; packets from these clusters need to be differentiated in
    order to be routed correctly.  For example:

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

    This method will assign an ID to each cluster of vertices (e.g., `(A)` and
    `(B)`) by storing the index in the `cluster` attribute of each vertex.
    Later this ID can be used in the keyspace of all nets originating from the
    cluster.
    """
    # Build a dictionary mapping each operator to the signals and routes for
    # which it is the source.
    operators_signal_routes = defaultdict(dict)
    for signal, routes in iteritems(signal_routes):
        operators_signal_routes[signal.source][signal] = routes

    # Assign identifiers to each of the clusters of vertices contained within
    # each operator.
    for operator, vertices in iteritems(operator_vertices):
        signal_routes = operators_signal_routes[operator]
        n_clusters = len({placements[vx] for vx in vertices})

        if len(signal_routes) == 0 or n_clusters == 1:
            # If the operator has no outgoing signals, or only one cluster,
            # then assign the same identifier to all of the vertices.
            for vertex in vertices:
                vertex.cluster = 0
        else:
            # Otherwise try to allocate as few cluster IDs as are required to
            # differentiate between multicast nets which take different routes
            # at the same router.
            #
            # Build a graph identifying which clusters may or may not share
            # identifiers.
            graph = build_cluster_graph(operators_signal_routes[operator])

            # Colour this graph to assign identifiers to the clusters
            cluster_ids = colour_graph(graph)

            # Assign these colours to the vertices.
            for vertex in vertices:
                placement = placements[vertex]
                vertex.cluster = cluster_ids[placement]


def build_cluster_graph(signal_routes):
    """Build a graph the nodes of which represent the chips on which the
    vertices representing a single operator have been placed and the edges of
    which represent constraints upon which of these chips may share routing
    identifiers for the purposes of this set of vertices.

    Parameters
    ----------
    signal_routes : {net: [RoutingTree, ...], ...}
        Dictionary mapping multi-source nets to the routing trees which
        describe them. The signals *MUST* all originate at the same vertex.

    Returns
    -------
    {(x, y): {(x, y), ...}, ...}
        An adjacency list representation of the graph described above.
    """
    cluster_graph = defaultdict(set)  # Adjacency list represent of the graph

    # Look at the multicast trees associated with each signal in turn.
    for trees in itervalues(signal_routes):
        # Build up a dictionary which maps each chip to a mapping of the routes
        # from this chip to the source cluster of the multicast nets which take
        # these routes. This will allow us to determine which clusters need to
        # be uniquely identified.
        chips_routes_clusters = defaultdict(lambda: defaultdict(set))

        # For every multicast tree associated with the signal we're currently
        # investigating:
        for tree in trees:
            source = tree.chip  # Get the origin of the multicast net
            cluster_graph[source]  # Ensure every cluster is in the graph

            # Traverse the multicast tree to build up the dictionary mapping
            # chips to routes and clusters.
            for _, chip, routes in tree.traverse():
                # Get the key for the routes taken
                route = 0x0
                for r in routes:
                    route |= (1 << r)

                # Add this cluster to the set of clusters whose net takes this
                # route at this point.
                chips_routes_clusters[chip][route].add(source)

                # Add constraints to the cluster graph dependent on which
                # multicast nets take different routes at this point.
                routes_from_chip = chips_routes_clusters[chip]
                for other_route, clusters in iteritems(routes_from_chip):
                    if other_route != route:  # We care about different routes
                        for cluster in clusters:
                            # This cluster cannot share an identifier with any
                            # of the clusters whose nets take a different route
                            # at this point.
                            cluster_graph[source].add(cluster)
                            cluster_graph[cluster].add(source)

    return cluster_graph


def colour_graph(graph):
    """Assign colours to each node in a graph such that connected nodes do not
    share a colour.

    Parameters
    ----------
    graph : {node: {node, ...}, ...}
        An adjacency list representation of a graph where the presence of an
        edge indicates that two nodes may not share a colour.

    Returns
    -------
    {node: int}
        Mapping from each node to an identifier (colour).
    """
    # This follows a heuristic of first assigning a colour to the node with the
    # highest degree and then progressing through other nodes in a
    # breadth-first search.
    colours = deque()  # List of sets which contain vertices
    unvisited = set(iterkeys(graph))  # Nodes which haven't been visited

    # While there are still unvisited nodes -- note that this may be true more
    # than once if there are disconnected cliques in the graph, e.g.:
    #
    #           (c)  (d)
    #            |   /
    #            |  /            (f) --- (g)
    #            | /               \     /
    #   (a) --- (b)                 \   /
    #             \                  (h)
    #              \
    #               \          (i)
    #               (e)
    #
    # Where a valid colouring would be:
    #   0: (b), (f), (i)
    #   1: (a), (c), (d), (e), (g)
    #   2: (h)
    #
    # Nodes might be visited in the order [(b) is always first]:
    #   (b), (a), (c), (d), (e) - new clique - (f), (g), (h) - again - (i)
    while unvisited:
        queue = deque()  # Queue of nodes to visit

        # Add the node with the greatest degree to the queue
        queue.append(max(unvisited, key=lambda vx: len(graph[vx])))

        # Perform a breadth-first search of the tree and colour nodes as we
        # touch them.
        while queue:
            node = queue.popleft()  # Get the next node to process

            if node in unvisited:
                # If the node is unvisited then mark it as visited
                unvisited.remove(node)

                # Colour the node, using the first legal colour or by creating
                # a new colour for the node.
                for group in colours:
                    if graph[node].isdisjoint(group):
                        group.add(node)
                        break
                else:
                    # Cannot colour this node with any of the existing colours,
                    # so create a new colour.
                    colours.append({node})

                # Add unvisited connected nodes to the queue
                for vx in graph[node]:
                    queue.append(vx)

    # Reverse the data format to result in {node: colour, ...}, for each group
    # of equivalently coloured nodes mark the colour on the node.
    colouring = dict()
    for i, group in enumerate(colours):
        for vx in group:
            colouring[vx] = i

    return colouring
