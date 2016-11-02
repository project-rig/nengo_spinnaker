import pytest
from rig.place_and_route.routing_tree import RoutingTree
from rig.routing_table import Routes
from six import iteritems

from nengo_spinnaker.netlist.key_allocation import (
    build_mn_net_graph, colour_graph, assign_mn_net_ids,
    build_cluster_graph
)


@pytest.mark.parametrize(
    "prior_constraints",
    [None, {},  # No constraints
     {'d': {'e'}, 'e': {'d'}},  # (e) may not share a colour with (d)
     ]
)
def test_build_mn_net_graph(prior_constraints):
    """Test the construction of the graph which describes which multi-source
    multicast nets may and may not share a key identifier.

    Six nets are constructed:

    (a):  NOTE MULTIPLE SOURCES (0, 0) and (0, 1)!

      (0, 1) --> (1, 1) [1]
                   ^
                   |
                   |
      (0, 0) --> (1, 0) [1, 2]


    (b):
      (0, 1) --> (1, 1) [1]


    (c):
      (0, 0) --> (1, 0) [1]


    (d):
                 (1, 1) [1, 2]

    (e): (0, 2) [1]


    (f)

      (0, 2) [1]
        ^
        |
        |
      (0, 1)

    The net graph should be of the form:

        (a) --- (c)
         | \
         |  (f)
         |     \
        (d) --- (b)

        (e)

    If prior constraints are provided then the graph should be of the form:

        (a) --- (c)
         | \
         |  (f)
         |     \
        (d) --- (b)
         |
         |
        (e)
    """
    # Construct the routing trees
    tree_a_11 = RoutingTree((1, 1), [(Routes.core(1), object())])
    tree_a_10 = RoutingTree((1, 0), [(Routes.core(1), object()),
                                     (Routes.core(2), object()),
                                     (Routes.north, tree_a_11)])
    tree_a0 = RoutingTree((0, 0), [(Routes.east, tree_a_10)])

    tree_b_11 = RoutingTree((1, 1), [(Routes.core(1), object())])
    tree_b = RoutingTree((0, 1), [(Routes.east, tree_b_11)])
    tree_a1 = tree_b  # The same

    tree_c_10 = RoutingTree((1, 0), [(Routes.core(1), object())])
    tree_c = RoutingTree((0, 0), [(Routes.east, tree_c_10)])

    tree_d = RoutingTree((1, 1), [(Routes.core(1), object()),
                                  (Routes.core(2), object())])

    tree_e = RoutingTree((0, 2), [(Routes.core(1), object())])

    tree_f = RoutingTree((0, 1), [(Routes.north, tree_e)])

    # Build the net graph
    routes = {
        'a': [tree_a0, tree_a1],
        'b': [tree_b],
        'c': [tree_c],
        'd': [tree_d],
        'e': [tree_e],
        'f': [tree_f],
    }

    if prior_constraints is not None:
        net_graph = build_mn_net_graph(routes, prior_constraints)
    else:
        net_graph = build_mn_net_graph(routes)

    # Assert that the net graph is as expected
    expected_graph = {
        'a': {'c', 'd', 'f'},
        'b': {'d', 'f'},
        'c': {'a'},
        'd': {'a', 'b'},
        'e': set(),
        'f': {'a', 'b'},
    }

    if prior_constraints is not None:
        for k, vs in iteritems(prior_constraints):
            expected_graph[k].update(vs)

    assert net_graph == expected_graph


def test_build_net_graph_complex():
    """In some cases multiple source nets might intersect and diverge with
    themselves (this occurs for recurrent connections on ensembles which have
    been partitioned across multiple chips) in this case we use a separate
    mechanism (cluster ID) to ensure that packets are routed differently so it
    is important that the graph ignores self-conflicting multiple source nets.

    Consider a multiple source net made up of the nets:

            (0, 0) --> (1, 0) [1]


        [1] (0, 0) <-- (1, 0)

    The packets traversing these nets must have different keys but this is
    achieved by assigning a separate field of the key to the one we are
    currently concerned with.

    Consequently the net graph resulting from the combination of the above
    multiple source net (a) with the other net (b):

            (0, 0) --> (1, 0) [2]

    Should be of the form:

            (a) -- (b)

    And not of the form (with a circular connection on (a)):

          /--\
          | (a) -- (b)
          \--/
    """
    # Construct the routes
    tree_a0_10 = RoutingTree((1, 0), [(Routes.core(1), object())])
    tree_a0 = RoutingTree((0, 0), [(Routes.east, tree_a0_10)])

    tree_a1_00 = RoutingTree((0, 0), [(Routes.core(1), object())])
    tree_a1 = RoutingTree((1, 0), [(Routes.west, tree_a1_00)])

    tree_b = tree_a0  # It's the same

    # Construct the net graph
    routes = {
        'a': [tree_a0, tree_a1],
        'b': [tree_b],
    }

    net_graph = build_mn_net_graph(routes)
    assert net_graph == {'a': {'b'}, 'b': {'a'}}


def test_colour_graph():
    """Test that the produced colouring is valid and that all nodes are assigned
    a colour.
    """
    graph = {
        'a': {'c', 'd'},
        'b': {'d'},
        'c': {'a'},
        'd': {'a', 'b'},
        'e': set(),
    }

    # Perform the colouring
    colours = colour_graph(graph)

    # Check the validity
    for net, others in iteritems(graph):
        for other in others:
            assert colours[net] != colours[other]


@pytest.mark.parametrize(
    "prior_constraints",
    [{},  # No constraints
     {'d': {'e'}, 'e': {'d'}},  # (e) may not share a colour with (d)
     {'d': {'e'}},  # Likewise, but improperly specified
     ]
)
def test_assign_net_ids(prior_constraints):
    """Test the entire process of assigning appropriate identifiers to
    multicast nets.

    Six nets are constructed:

    (a):
      (0, 1) --> (1, 1) [1]
                   ^
                   |
                   |
      (0, 0) --> (1, 0) [1, 2]


    (b):
      (0, 1) --> (1, 1) [1]


    (c):
      (0, 0) --> (1, 0) [1]


    (d):
                 (1, 1) [1, 2]

    (e): (0, 2) [1]


    (f)

      (0, 2) [1]
        ^
        |
        |
      (0, 1)

    The net graph should be of the form:

        (a) --- (c)
         | \
         |  (f)
         |     \
        (d) --- (b)

        (e)

    If prior constraints are provided then the graph should be of the form:

        (a) --- (c)
         | \
         |  (f)
         |     \
        (d) --- (b)
         |
         |
        (e)
    """
    # Construct the routing trees
    tree_a_11 = RoutingTree((1, 1), [(Routes.core(1), object())])
    tree_a_10 = RoutingTree((1, 0), [(Routes.core(1), object()),
                                     (Routes.core(2), object()),
                                     (Routes.north, tree_a_11)])
    tree_a0 = RoutingTree((0, 0), [(Routes.east, tree_a_10)])

    tree_b_11 = RoutingTree((1, 1), [(Routes.core(1), object())])
    tree_b = RoutingTree((0, 1), [(Routes.east, tree_b_11)])
    tree_a1 = tree_b  # The same

    tree_c_10 = RoutingTree((1, 0), [(Routes.core(1), object())])
    tree_c = RoutingTree((0, 0), [(Routes.east, tree_c_10)])

    tree_d = RoutingTree((1, 1), [(Routes.core(1), object()),
                                  (Routes.core(2), object())])

    tree_e = RoutingTree((0, 2), [(Routes.core(1), object())])

    tree_f = RoutingTree((0, 1), [(Routes.north, tree_e)])

    # Build the net graph
    routes = {
        'a': [tree_a0, tree_a1],
        'b': [tree_b],
        'c': [tree_c],
        'd': [tree_d],
        'e': [tree_e],
        'f': [tree_f],
    }
    # Nets are assigned identifiers
    net_ids = assign_mn_net_ids(routes, prior_constraints)

    # Check that the net IDs are appropriate
    net_graph = build_mn_net_graph(routes, prior_constraints)
    for net, others in iteritems(net_graph):
        for other in others:
            assert net_ids[net] != net_ids[other]


def test_build_cluster_graph_completely_connected():
    """Test the construction of a graph which indicates which of the clusters
    of the vertices of an operator may not share an identifier.

    In this case none of the clusters may share an ID as they have multiple
    crossing connections.
    """
    # Create the vertices
    a = object()
    b = object()
    c = object()

    # Create the routing trees for the entirely recurrent connections
    tree_a10 = RoutingTree((1, 0), [(Routes.core(1), c)])
    tree_a01 = RoutingTree((0, 1), [(Routes.core(1), b)])
    tree_a00 = RoutingTree((0, 0), [(Routes.north, tree_a01),
                                    (Routes.east, tree_a10),
                                    (Routes.core(1), a)])

    tree_b00 = RoutingTree((0, 0), [(Routes.east, tree_a10),
                                    (Routes.core(1), a)])
    tree_b01 = RoutingTree((0, 1), [(Routes.south, tree_b00),
                                    (Routes.core(1), b)])

    tree_c00 = RoutingTree((0, 0), [(Routes.north, tree_a01),
                                    (Routes.core(1), a)])
    tree_c10 = RoutingTree((1, 0), [(Routes.west, tree_c00),
                                    (Routes.core(1), c)])

    # Build the graph indicating which cluster placements cannot share
    # identifiers, in this case none of the clusters may share an ID
    graph = build_cluster_graph({object(): [tree_a00, tree_b01, tree_c10]})
    assert graph == {
        (0, 0): {(0, 1), (1, 0)},
        (0, 1): {(0, 0), (1, 0)},
        (1, 0): {(0, 0), (0, 1)},
    }


def test_build_cluster_graph_some_edges():
    """Test the construction of a graph which indicates which of the clusters
    of the vertices of an operator may not share an identifier.

    In this, slightly strange, case two of the clusters may not share an ID.
    """
    # Create the vertices
    a = object()
    b = object()
    c = object()

    # Create the routing trees
    tree_a10 = RoutingTree((1, 0), [(Routes.core(1), b)])
    tree_a00 = RoutingTree((0, 0), [(Routes.east, tree_a10)])

    tree_b00 = RoutingTree((0, 0), [(Routes.core(1), a)])
    tree_b10 = RoutingTree((1, 0), [(Routes.west, tree_b00)])

    tree_c01 = RoutingTree((0, 1), [(Routes.south, tree_b00)])

    # Build the graph
    graph = build_cluster_graph({object(): [tree_a00, tree_b10],
                                 object(): [tree_c01]})
    assert graph == {
        (0, 0): {(1, 0)},
        (1, 0): {(0, 0)},
        (0, 1): set(),  # Every cluster should be in the graph
    }
