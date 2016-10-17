import logging
from collections import defaultdict, deque
import itertools
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
        signal.keyspace = keyspaces["nengo"](connection_id=i)

        # Expand the keyspace to fit the required indices
        signal.keyspace(index=signal.width - 1)

    if (signal_ids):
        logger.info("%u signals assigned %u IDs", len(signal_ids),
                    max(itervalues(signal_ids)))


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
    return colour_net_graph(
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
    # Construct a map from chips to unique sets of routes from that chip to
    # nets which take that route.
    chip_route_nets = defaultdict(lambda: defaultdict(deque))
    for net, trees in iteritems(nets_routes):
        for tree in trees:
            for _, chip, routes in tree.traverse():
                chip_route_nets[chip][frozenset(routes)].append(net)

    # The different sets of routes from a chip indicate nets which cannot share
    # a routing key, this is indicated by creating an edge between those `nets'
    # in the net graph.
    net_graph = {net: set() for net in iterkeys(nets_routes)}
    for route_nets in itervalues(chip_route_nets):
        for xs, ys in itertools.combinations(itervalues(route_nets), 2):
            for x, y in ((i, j) for j in ys for i in xs if i != j):
                # Add an edge iff. it would connect two *different* vertices
                net_graph[x].add(y)
                net_graph[y].add(x)

    # Add any prior constraints into the net graph (doing so in such a way that
    # ensures that the prior constraints are undirected).
    if prior_constraints is not None:
        for u, vs in iteritems(prior_constraints):
            for v in vs:
                net_graph[u].add(v)
                net_graph[v].add(u)

    return net_graph


def colour_net_graph(net_graph):
    """Assign colours to each net such that the colouring constraints are
    satisfied.

    Parameters
    ----------
    net_graph : {net: {net, ...}, ...}
        An adjacency list representation of a graph where the presence of an
        edge indicates that two multicast nets may not share a routing key.

    Returns
    -------
    {Net: int}
        Mapping from each net (key of the net graph) to an identifier.
    """
    # This follows a heuristic of first assigning a colour to the node with the
    # highest degree and then progressing through other nodes in a
    # breadth-first search.
    colours = deque()  # List of sets which contain vertices
    unvisited = set(iterkeys(net_graph))  # Nodes which haven't been visited

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
        queue.append(max(unvisited, key=lambda vx: len(net_graph[vx])))

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
                    if net_graph[node].isdisjoint(group):
                        group.add(node)
                        break
                else:
                    # Cannot colour this node with any of the existing colours,
                    # so create a new colour.
                    colours.append({node})

                # Add unvisited connected nodes to the queue
                for vx in net_graph[node]:
                    queue.append(vx)

    # Reverse the data format to result in {net: colour, ...}, for each group
    # of equivalently coloured nodes mark the colour on the node.
    colouring = dict()
    for i, group in enumerate(colours):
        for vx in group:
            colouring[vx] = i

    return colouring
