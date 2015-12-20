import mock
import pytest
from rig.bitfield import BitField
from rig.place_and_route import Cores, SDRAM
from rig.place_and_route.constraints import (ReserveResourceConstraint,
                                             LocationConstraint)

from nengo_spinnaker import netlist
from nengo_spinnaker.utils.itertools import flatten


def test_load_application():
    """Test the steps involved in loading an application onto a SpiNNaker
    machine.

     - Building and loading routing tables
     - Allocating SDRAM for vertices which require it
     - Calling application loading functions
     - Loading applications to the machine
    """
    # Mock controller which will be used to load the model
    controller = mock.Mock(name="controller")

    # Create the objects to store in the model
    v1 = netlist.Vertex("test_app", resources={Cores: 1, SDRAM: 400})
    v2a = netlist.Vertex("test_app2", resources={Cores: 2, SDRAM: 100})
    v2b = netlist.Vertex("test_app2", resources={Cores: 2, SDRAM: 100})

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

    # net = netlist.NMNet(v1, [v2a, v2b], 1, keyspace)
    net = object()

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
    model.net_keyspaces = {net: keyspace}
    model.routes = {net: routing_tree}

    # Mock routing table
    routing_table = mock.Mock(name="routing table")

    # Patch out all the methods that should be called
    with \
            mock.patch("nengo_spinnaker.netlist.netlist."
                       "build_routing_tables") as build_routing_tables, \
            mock.patch("nengo_spinnaker.netlist.netlist."
                       "build_application_map") as build_application_map, \
            mock.patch("nengo_spinnaker.netlist.netlist."
                       "sdram_alloc_for_vertices") as sdram_alloc:
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


def test_before_simulation():
    """Test that all methods are called when asked to prepare a simulation and
    that the simulation duration is written in correctly.
    """
    # Create some "before_simulation" functions
    before_a = mock.Mock()
    before_b = mock.Mock()

    # Create a vertex
    vertex = mock.Mock()

    # Create a netlist
    model = netlist.Netlist(
        nets=[],
        vertices=[vertex],
        keyspaces={},
        groups={},
        load_functions=[],
        before_simulation_functions=[before_a, before_b]
    )
    model.placements[vertex] = (1, 2)
    model.allocations[vertex] = {Cores: slice(5, 7)}

    # Call the before_simulation_functions
    simulator = mock.Mock(name="Simulator")
    model.before_simulation(simulator, 100)

    before_a.assert_called_once_with(model, simulator, 100)
    before_b.assert_called_once_with(model, simulator, 100)

    # Check we wrote in the run time
    simulator.controller.write_vcpu_struct_field.assert_called_once_with(
        "user1", 100, 1, 2, 5
    )


def test_after_simulation():
    """Test that all methods are called when asked to finish a simulation."""
    # Create some "before_simulation" functions
    after_a = mock.Mock()
    after_b = mock.Mock()

    # Create a netlist
    model = netlist.Netlist(
        nets=[],
        vertices=[],
        keyspaces={},
        groups={},
        load_functions=[],
        after_simulation_functions=[after_a, after_b]
    )

    # Call the before_simulation_functions
    simulator = mock.Mock(name="Simulator")
    model.after_simulation(simulator, 100)

    after_a.assert_called_once_with(model, simulator, 100)
    after_b.assert_called_once_with(model, simulator, 100)
