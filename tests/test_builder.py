import mock
import nengo.builder
import pytest

import rig.bitfield
from rig.machine import Cores, SDRAM

from nengo_spinnaker.annotations import Annotations, AnnotatedNet, NetAddress
import nengo_spinnaker.builder
from nengo_spinnaker.builder import SpiNNakerModel
from nengo_spinnaker.netlist import Net, Vertex
from nengo_spinnaker.utils.itertools import flatten


class TestBuildObjectsAndNets(object):
    """Test that objects are correctly built when using
    SpiNNakerModel.from_annotations.
    """
    def test_no_builder(self):
        """Test a TypeError if no build method is known."""
        ann = mock.Mock(name="object_annotation")
        annotation = Annotations({}, {}, [ann], [])

        with mock.patch.object(SpiNNakerModel, "builders", {}), \
                pytest.raises(TypeError) as excinfo:
            SpiNNakerModel.from_annotations(None, annotation)

        assert ann.__class__.__name__ in str(excinfo.value)

    @pytest.mark.parametrize(
        "make_annotation",
        [(lambda obj, obj_annotation, connection, net:
            Annotations({obj: obj_annotation}, {connection: net}, [], [])),
         (lambda obj, obj_annotation, connection, net:
             Annotations({obj: obj_annotation}, {}, [], [net])),
         ]
    )
    def test_standard(self, make_annotation):
        """Test the standard building from annotations.
        """
        # Create the initial object, its parameters and its annotation
        class Obj(object):
            pass

        class ObjAnn(object):
            pass

        obj = Obj()
        obj_annotation = ObjAnn()

        # Create a model to hold the parameters
        model = nengo.builder.Model()

        # Create the builder and the return values for the builder
        obj_vertex = mock.Mock(name="object_vertex")
        obj_load = mock.Mock(name="object_load")
        obj_pre = mock.Mock(name="object_pre")
        obj_post = mock.Mock(name="object_post")

        obj_builder = mock.Mock(name="object_builder")
        obj_builder.return_value = (obj_vertex, obj_load, obj_pre, obj_post)

        # Create a Net
        net = AnnotatedNet(NetAddress(obj_annotation, None),
                           NetAddress(obj_annotation, None))
        net.keyspace = rig.bitfield.BitField(length=32)
        net.weight = 77

        # Create annotation
        annotation = make_annotation(obj, obj_annotation, None, net)

        # Call the builder, then ensure that the produced model is appropriate
        # and that the builder was called correctly.
        with mock.patch.object(SpiNNakerModel, "builders",
                               {ObjAnn: obj_builder}):
            smodel = SpiNNakerModel.from_annotations(model, annotation)

        # Check the builder was called correctly
        obj_builder.assert_called_once_with(obj, obj_annotation,
                                            model, annotation)

        # Check the return values were correctly used
        assert len(smodel.nets) == 1
        assert smodel.nets[0].source is obj_vertex
        assert smodel.nets[0].sinks == [obj_vertex]
        assert smodel.nets[0].weight == 77
        assert smodel.nets[0].keyspace is net.keyspace

        assert smodel.vertices == [obj_vertex]
        assert smodel.groups == dict()
        assert smodel.load_functions == [obj_load]
        assert smodel.before_simulation_functions == [obj_pre]
        assert smodel.after_simulation_functions == [obj_post]
        assert smodel.placements == dict()
        assert smodel.allocations == dict()
        assert smodel.vertices_memory == dict()
        assert smodel.routes == dict()

    @pytest.mark.parametrize(
        "make_annotation",
        [(lambda obj, obj_annotation, connection, net:
            Annotations({obj: obj_annotation}, {connection: net}, [], [])),
         (lambda obj, obj_annotation, connection, net:
            Annotations({obj: obj_annotation}, {}, [], [net])),
         ]
    )
    def test_multiple_vertices(self, make_annotation):
        """Test the standard building from annotations when multiple vertices
        are produced.
        """
        # Create the initial object, its parameters and its annotation
        class Obj(object):
            pass

        class ObjAnn(object):
            pass

        obj = Obj()
        obj_params = mock.Mock(name="obj_params")
        obj_annotation = ObjAnn()

        # Create a model to hold the parameters
        model = nengo.builder.Model()
        model.params[obj] = obj_params

        # Create the builder and the return values for the builder
        obj_vertex = [mock.Mock(name="object_vertex") for _ in range(10)]
        obj_load = mock.Mock(name="object_load")
        obj_pre = mock.Mock(name="object_pre")
        obj_post = mock.Mock(name="object_post")

        obj_builder = mock.Mock(name="object_builder")
        obj_builder.return_value = (obj_vertex, obj_load, obj_pre, obj_post)

        # Create a Net
        net = AnnotatedNet(NetAddress(obj_annotation, None),
                           NetAddress(obj_annotation, None))
        net.keyspace = rig.bitfield.BitField(length=32)
        net.weight = 7

        # Create annotation
        annotation = make_annotation(obj, obj_annotation, None, net)

        # Call the builder, then ensure that the produced model is appropriate
        # and that the builder was called correctly.
        with mock.patch.object(SpiNNakerModel, "builders",
                               {ObjAnn: obj_builder}):
            smodel = SpiNNakerModel.from_annotations(model, annotation)

        # Check the builder was called correctly
        obj_builder.assert_called_once_with(obj, obj_annotation,
                                            model, annotation)

        # Check the return values were correctly used
        assert smodel.vertices == obj_vertex
        assert smodel.groups == {v: 0 for v in obj_vertex}

        # Check the nets reflect the splitting of the vertices
        assert len(smodel.nets) == len(obj_vertex)
        for new_net in smodel.nets:
            assert new_net.source in obj_vertex
            assert new_net.sinks == obj_vertex
            assert new_net.keyspace is net.keyspace
            assert new_net.weight == net.weight


def test_place_and_route():
    """Test that models can place and route themselves.

    Provide two vertices and net and ensure that the correct methods are called
    to place and allocate, fix the bitfield and then route the model.  Ensure
    that the returned values are stored.
    """
    # Create the objects to store in the model
    v1 = Vertex("test_app", resources={Cores: 1, SDRAM: 400},
                constraints=[mock.Mock(name="constraint1")])
    v2a = Vertex("test_app", resources={Cores: 2, SDRAM: 100},
                 constraints=[mock.Mock(name="constraint2"),
                              mock.Mock(name="constraint3")])
    v2b = Vertex("test_app", resources={Cores: 2, SDRAM: 100},
                 constraints=[mock.Mock(name="constraint2"),
                              mock.Mock(name="constraint3")])

    keyspace = mock.Mock(name="keyspace")
    keyspace.length = 32
    net = Net(v1, [v2a, v2b], 1, keyspace)
    groups = {v2a: 1, v2b: 1}

    # Create the model
    model = SpiNNakerModel(
        nets=[net],
        vertices=[v1, v2a, v2b],
        groups=groups,
        load_functions=list(),
        before_simulation_functions=list(),
        after_simulation_functions=list()
    )

    # Patch out all the methods that should be called
    with \
            mock.patch.object(
                nengo_spinnaker.builder, "place") as place, \
            mock.patch.object(
                nengo_spinnaker.builder, "allocate") as allocate, \
            mock.patch.object(
                nengo_spinnaker.builder, "identify_clusters") as cluster, \
            mock.patch.object(
                nengo_spinnaker.builder.keyspaces, "assign_fields"
            ) as assign_fields, \
            mock.patch.object(
                nengo_spinnaker.builder, "route") as route:
        # Create a mock machine to place/route against
        machine = mock.Mock(name="machine")

        # Set up these methods to ensure that the call order is correct and
        # that they return appropriate objects
        def place_fn(resources, nets, machine, constraints):
            # Check that the arguments were correct
            assert resources == {v: v.resources for v in model.vertices}
            assert nets == model.nets
            assert constraints == list(flatten(
                v.constraints for v in model.vertices))

            # Return some placements
            return {v1: (0, 0), v2a: (0, 1), v2b: (1, 0)}

        place.side_effect = place_fn

        def allocate_fn(resources, nets, machine, constraints, placements):
            assert place.call_count == 1

            # Check that the arguments were correct
            assert resources == {v: v.resources for v in model.vertices}
            assert nets == model.nets
            assert constraints == list(flatten(
                v.constraints for v in model.vertices))
            assert placements == {v1: (0, 0), v2a: (0, 1), v2b: (1, 0)}

            # Return some allocations
            return {
                v1: {Cores: slice(0, 1)},
                v2a: {Cores: slice(9, 10)},
                v2b: {Cores: slice(4, 6)},
            }

        allocate.side_effect = allocate_fn

        def cluster_fn(placed_vertices, nets, groups):
            assert allocate.call_count == 1

            # Check that the arguments are correct
            assert placed_vertices == {v1: (0, 0), v2a: (0, 1), v2b: (1, 0)}
            assert nets == model.nets
            assert groups == model.groups

        cluster.side_effect = cluster_fn

        def assign_fields_fn():
            assert cluster.call_count == 1

        assign_fields.side_effect = assign_fields_fn

        routing_tree = mock.Mock(name="routing tree")

        def route_fn(resources, nets, machine_, constraints, placements,
                     allocations):
            assert assign_fields.call_count == 1

            # Check that the arguments are correct
            assert resources == {v: v.resources for v in model.vertices}
            assert nets == model.nets
            assert machine_ is machine
            assert constraints == list(flatten(
                v.constraints for v in model.vertices))
            assert placements == {v1: (0, 0), v2a: (0, 1), v2b: (1, 0)}
            assert allocations == {v1: {Cores: slice(0, 1)},
                                   v2a: {Cores: slice(9, 10)},
                                   v2b: {Cores: slice(4, 6)}, }

            # Return some routes
            return {net: routing_tree}

        route.side_effect = route_fn

        # Assert starting from a blank slate
        assert model.placements == dict()
        assert model.allocations == dict()
        assert model.routes == dict()

        # Perform the place and route for the machine
        model.place_and_route(machine)

        # Assert methods were called once each
        assert place.call_count == 1
        assert allocate.call_count == 1
        assert cluster.call_count == 1
        assert assign_fields.call_count == 1
        assert route.call_count == 1

        # Assert that the placements, etc. were recorded
        assert model.placements == {v1: (0, 0), v2a: (0, 1), v2b: (1, 0)}
        assert model.allocations == {v1: {Cores: slice(0, 1)},
                                     v2a: {Cores: slice(9, 10)},
                                     v2b: {Cores: slice(4, 6)}, }
        assert model.routes == {net: routing_tree}


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
    v1 = Vertex("test_app", resources={Cores: 1, SDRAM: 400},
                constraints=[mock.Mock(name="constraint1")])
    v2a = Vertex("test_app2", resources={Cores: 2, SDRAM: 100},
                 constraints=[mock.Mock(name="constraint2"),
                              mock.Mock(name="constraint3")])
    v2b = Vertex("test_app2", resources={Cores: 2, SDRAM: 100},
                 constraints=[mock.Mock(name="constraint2"),
                              mock.Mock(name="constraint3")])

    keyspace = mock.Mock(name="keyspace")
    keyspace.length = 32

    def get_mask(tag):
        assert tag == "routing"
        return 0xffff0000

    def get_value(tag):
        assert tag == "routing"
        return 0x33330000

    keyspace.get_mask.side_effect = get_mask
    keyspace.get_value.side_effect = get_value

    net = Net(v1, [v2a, v2b], 1, keyspace)

    groups = {v2a: 1, v2b: 1}

    routing_tree = mock.Mock(name="routing tree")

    # Create the model
    model = SpiNNakerModel(
        nets=[net],
        vertices=[v1, v2a, v2b],
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
            mock.patch.object(
                nengo_spinnaker.builder, "build_routing_tables") as \
            build_routing_tables, \
            mock.patch.object(
                nengo_spinnaker.builder, "build_application_map") as \
            build_application_map, \
            mock.patch.object(
                nengo_spinnaker.builder, "sdram_alloc_for_vertices") as \
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
