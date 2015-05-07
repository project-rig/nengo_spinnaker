import mock
import pytest
from rig.bitfield import BitField
from rig import machine
from rig.machine import Cores, SDRAM

from nengo_spinnaker import netlist
from nengo_spinnaker.utils.itertools import flatten


def test_vertex():
    constraints = [mock.Mock()]
    resources = {machine.Cores: 1, machine.SDRAM: 8*1024}

    v = netlist.Vertex(application="test", constraints=constraints,
                       resources=resources)

    assert v.constraints == constraints
    assert v.constraints is not constraints
    assert v.resources == resources
    assert v.resources is not resources
    assert v.application == "test"


def test_vertex_slice():
    # No resources or constraints
    v = netlist.VertexSlice(slice(None))
    assert v.slice == slice(None)
    assert v.application is None
    assert v.constraints == list()
    assert v.resources == dict()

    # Provide resources and constraints and application
    resources = {machine.Cores: 1, machine.SDRAM: 8*1024}
    constraints = [mock.Mock()]

    v = netlist.VertexSlice(slice(0, 10), application="test",
                            constraints=constraints,
                            resources=resources)
    assert v.application == "test"
    assert v.constraints == constraints
    assert v.constraints is not constraints
    assert v.resources == resources
    assert v.resources is not resources


class TestNet(object):
    def test_single_sink(self):
        """Create a net with a single sink and source."""
        # Create source, sink and keyspace
        source = netlist.VertexSlice(slice(0, 5))
        sink = netlist.VertexSlice(slice(1, 6))
        weight = 3
        keyspace = BitField(length=32)

        # Create the Net, assert the values are stored
        net = netlist.Net(source, sink, weight, keyspace)

        assert net.source is source
        assert net.sinks == [sink]
        assert net.weight == weight
        assert net.keyspace is keyspace

    def test_multiple_sinks(self):
        """Create a net with a single sink and source."""
        # Create source, sink and keyspace
        source = netlist.VertexSlice(slice(0, 5))
        sinks = [netlist.VertexSlice(slice(1, 6)),
                 netlist.VertexSlice(slice(5, 8))]
        weight = 3
        keyspace = BitField(length=32)

        # Create the Net, assert the values are stored
        net = netlist.Net(source, sinks, weight, keyspace)

        assert net.source is source
        assert net.sinks == sinks
        assert net.weight == weight
        assert net.keyspace is keyspace

    @pytest.mark.parametrize("length", [16, 31, 33])
    def test_assert_keyspace_length(self, length):
        """Check that the keyspace is only accepted if it is of length 32
        bits.
        """
        source = netlist.VertexSlice(slice(0, 5))
        sink = netlist.VertexSlice(slice(1, 6))
        weight = 3
        keyspace = BitField(length=length)

        with pytest.raises(ValueError) as excinfo:
            netlist.Net(source, sink, weight, keyspace)

        err_string = str(excinfo.value)
        assert "keyspace" in err_string
        assert "32" in err_string
        assert "{}".format(length) in err_string


def test_place_and_route():
    """Test that Netlists can place and route themselves.

    Provide two vertices and net and ensure that the correct methods are called
    to place and allocate, fix the bitfield and then route the model.  Ensure
    that the returned values are stored.
    """
    # Create the objects to store in the Netlist
    v1 = netlist.Vertex("test_app", resources={Cores: 1, SDRAM: 400},
                        constraints=[mock.Mock(name="constraint1")])
    v2a = netlist.Vertex("test_app", resources={Cores: 2, SDRAM: 100},
                         constraints=[mock.Mock(name="constraint2"),
                                      mock.Mock(name="constraint3")])
    v2b = netlist.Vertex("test_app", resources={Cores: 2, SDRAM: 100},
                         constraints=[mock.Mock(name="constraint2"),
                                      mock.Mock(name="constraint3")])

    keyspace = mock.Mock(name="keyspace")
    keyspace.length = 32
    net = netlist.Net(v1, [v2a, v2b], 1, keyspace)
    groups = [(v2a, v2b)]

    keyspace_container = mock.Mock()

    # Create the Netlist
    nl = netlist.Netlist(
        nets=[net],
        vertices=[v1, v2a, v2b],
        keyspaces=keyspace_container,
        groups=groups,
        load_functions=list(),
        before_simulation_functions=list(),
        after_simulation_functions=list()
    )

    # Patch out all the methods that should be called
    with mock.patch("nengo_spinnaker.netlist.identify_clusters") as cluster:
        # Create a mock machine to place/route against
        machine = mock.Mock(name="machine")

        # Set up these methods to ensure that the call order is correct and
        # that they return appropriate objects
        placer_kwargs = {"spam": "foo"}

        def place_fn(resources, nets, machine, constraints, **kwargs):
            # Check that the arguments were correct
            assert resources == {v: v.resources for v in nl.vertices}
            assert nets == nl.nets
            assert constraints == list(flatten(  # pragma : no branch
                v.constraints for v in nl.vertices))
            assert kwargs == placer_kwargs

            # Return some placements
            return {v1: (0, 0), v2a: (0, 1), v2b: (1, 0)}

        place = mock.Mock(wraps=place_fn)

        allocater_kwargs = {"egg": "bar"}

        def allocate_fn(resources, nets, machine, constraints, placements,
                        **kwargs):
            assert place.call_count == 1

            # Check that the arguments were correct
            assert resources == {v: v.resources for v in nl.vertices}
            assert nets == nl.nets
            assert constraints == list(flatten(  # pragma : no branch
                v.constraints for v in nl.vertices))
            assert placements == {v1: (0, 0), v2a: (0, 1), v2b: (1, 0)}
            assert kwargs == allocater_kwargs

            # Return some allocations
            return {
                v1: {Cores: slice(0, 1)},
                v2a: {Cores: slice(9, 10)},
                v2b: {Cores: slice(4, 6)},
            }

        allocate = mock.Mock(wraps=allocate_fn)

        def cluster_fn(placed_vertices, nets, groups):
            assert allocate.call_count == 1

            # Check that the arguments are correct
            assert placed_vertices == {v1: (0, 0), v2a: (0, 1), v2b: (1, 0)}
            assert nets == nl.nets
            assert groups == nl.groups

        cluster.side_effect = cluster_fn

        def assign_fields_fn():
            assert cluster.call_count == 1

        keyspace_container.assign_fields.side_effect = assign_fields_fn

        routing_tree = mock.Mock(name="routing tree")

        router_kwargs = {"King": "of the Britons"}

        def route_fn(resources, nets, machine_, constraints, placements,
                     allocations, **kwargs):
            assert keyspace_container.assign_fields.call_count == 1

            # Check that the arguments are correct
            assert resources == {v: v.resources for v in nl.vertices}
            assert nets == nl.nets
            assert machine_ is machine
            assert constraints == list(flatten(  # pragma : no branch
                v.constraints for v in nl.vertices))
            assert placements == {v1: (0, 0), v2a: (0, 1), v2b: (1, 0)}
            assert allocations == {v1: {Cores: slice(0, 1)},
                                   v2a: {Cores: slice(9, 10)},
                                   v2b: {Cores: slice(4, 6)}, }
            assert kwargs == router_kwargs

            # Return some routes
            return {net: routing_tree}

        route = mock.Mock(wraps=route_fn)

        # Assert starting from a blank slate
        assert nl.placements == dict()
        assert nl.allocations == dict()
        assert nl.routes == dict()

        # Perform the place and route for the machine
        nl.place_and_route(
            machine,
            place=place,
            place_kwargs=placer_kwargs,
            allocate=allocate,
            allocate_kwargs=allocater_kwargs,
            route=route,
            route_kwargs=router_kwargs
        )

        # Assert methods were called once each
        assert place.call_count == 1
        assert allocate.call_count == 1
        assert cluster.call_count == 1
        assert keyspace_container.assign_fields.call_count == 1
        assert route.call_count == 1

        # Assert that the placements, etc. were recorded
        assert nl.placements == {v1: (0, 0), v2a: (0, 1), v2b: (1, 0)}
        assert nl.allocations == {v1: {Cores: slice(0, 1)},
                                  v2a: {Cores: slice(9, 10)},
                                  v2b: {Cores: slice(4, 6)}, }
        assert nl.routes == {net: routing_tree}


def test_load_application():
    """Test the steps involved in loading an application to a SpiNNaker
    machine.

     - Building and loading routing tables
     - Allocating SDRAM for vertices which require it
     - Calling application loading functions
     - Loading applications to the machine
    """
    # Mock controller which will be used to load the model
    controller = mock.Mock(name="controller")

    # Create the objects to store in the model
    v1 = netlist.Vertex("test_app", resources={Cores: 1, SDRAM: 400},
                        constraints=[mock.Mock(name="constraint1")])
    v2a = netlist.Vertex("test_app2", resources={Cores: 2, SDRAM: 100},
                         constraints=[mock.Mock(name="constraint2"),
                                      mock.Mock(name="constraint3")])
    v2b = netlist.Vertex("test_app2", resources={Cores: 2, SDRAM: 100},
                         constraints=[mock.Mock(name="constraint2"),
                                      mock.Mock(name="constraint3")])

    keyspace_container = mock.Mock(name="keyspace container")
    keyspace_container.routing_tag = "bob"

    keyspace = mock.Mock(name="keyspace")
    keyspace.length = 32

    def get_mask(tag):
        assert tag == keyspace_container.routing_tag
        return 0xffff0000

    def get_value(tag):
        assert tag == keyspace_container.routing_tag
        return 0x33330000

    keyspace.get_mask.side_effect = get_mask
    keyspace.get_value.side_effect = get_value

    net = netlist.Net(v1, [v2a, v2b], 1, keyspace)

    groups = [(v2a, v2b)]

    routing_tree = mock.Mock(name="routing tree")

    # Create the model
    model = netlist.Netlist(
        nets=[net],
        vertices=[v1, v2a, v2b],
        keyspaces=keyspace_container,
        groups=groups,
        load_functions=list(),
        before_simulation_functions=list(),
        after_simulation_functions=list()
    )
    model.placements = {v1: (0, 0), v2a: (0, 1), v2b: (1, 0)}
    model.allocations = {
        v1: {Cores: slice(0, 1), SDRAM: slice(0, 400)},
        v2a: {Cores: slice(9, 11), SDRAM: slice(300, 400)},
        v2b: {Cores: slice(9, 11), SDRAM: slice(300, 400)},
    }
    model.routes = {net: routing_tree}

    # Mock routing table
    routing_table = mock.Mock(name="routing table")

    # Patch out all the methods that should be called
    with \
            mock.patch("nengo_spinnaker.netlist.build_routing_tables") as \
            build_routing_tables, \
            mock.patch("nengo_spinnaker.netlist.build_application_map") as \
            build_application_map, \
            mock.patch("nengo_spinnaker.netlist.sdram_alloc_for_vertices") as \
            sdram_alloc:

        # Create replacement methods for patched methods
        def build_routing_tables_fn(routes, net_keys, **kwargs):
            # Assert that the arguments were correct
            assert routes == model.routes
            assert net_keys == {net: (0x33330000, 0xffff0000)}

            # Assert that the keyspace was only called once
            assert keyspace.get_mask.call_count == 1
            assert keyspace.get_value.call_count == 1

            # Return a routing table
            return routing_table

        build_routing_tables.side_effect = build_routing_tables_fn

        def build_application_map_fn(vertices_applications, placements,
                                     allocations):
            # Assert that the arguments are correct
            assert vertices_applications == {v: v.application for v in
                                             model.vertices}
            assert placements == model.placements
            assert allocations == model.allocations

            return {
                "test_app": {(0, 0): set([0])},
                "test_app2": {(0, 1): set([9, 10]),
                              (1, 0): set([9, 10]),
                              }
            }

        build_application_map.side_effect = build_application_map_fn

        sdram_allocs = {
            v1: mock.Mock(),
            v2a: mock.Mock(),
            v2b: mock.Mock(),
        }

        def sdram_alloc_fn(cn, placements, allocations):
            assert cn is controller
            assert placements == model.placements
            assert allocations == model.allocations

            return sdram_allocs

        sdram_alloc.side_effect = sdram_alloc_fn

        # Create some load functions, add them to the model
        def load_fn(model_, controller_):
            # Assert that memory is alloced before calling load functions
            assert sdram_alloc.call_count == 1
            assert model_.vertices_memory == sdram_allocs

            # Assert that the arguments are correct
            assert model_ is model
            assert controller_ is controller

        load_a = mock.Mock(wraps=load_fn)
        load_b = mock.Mock(wraps=load_fn)
        model.load_functions = [load_a, load_b]

        # Perform the loading
        model.load_application(controller)

        # Assert methods were called
        assert build_routing_tables.call_count == 1
        assert build_application_map.call_count == 1
        assert sdram_alloc.call_count == 1

        assert load_a.call_count == 1
        assert load_b.call_count == 1

        controller.load_routing_tables.assert_called_once_with(routing_table)
        controller.load_application.assert_called_once_with({
            "test_app": {(0, 0): set([0])},
            "test_app2": {(0, 1): set([9, 10]),
                          (1, 0): set([9, 10]),
                          }
        })

        assert model.vertices_memory == sdram_allocs
