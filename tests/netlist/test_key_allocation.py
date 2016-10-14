import pytest
from rig.place_and_route.routing_tree import RoutingTree
from rig.routing_table import Routes
from six import iteritems

from nengo_spinnaker.netlist.key_allocation import (
    build_mn_net_graph, colour_net_graph, assign_mn_net_ids)


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


def test_colour_net_graph():
    """Test that the produced colouring is valid and that all nets are assigned
    a colour.
    """
    net_graph = {
        'a': {'c', 'd'},
        'b': {'d'},
        'c': {'a'},
        'd': {'a', 'b'},
        'e': set(),
    }

    # Perform the colouring
    colours = colour_net_graph(net_graph)

    # Check the validity
    for net, others in iteritems(net_graph):
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
