import mock
import nengo.builder
import pytest
import rig.bitfield

from nengo_spinnaker.annotations import Annotations, AnnotatedNet, NetAddress
from nengo_spinnaker.builder import SpiNNakerModel


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
        "make_annotation, exp_params",
        [(lambda obj, obj_annotation, connection, net:
            Annotations({obj: obj_annotation},
                        {connection: net}, [], []), True),
         (lambda obj, obj_annotation, connection, net:
            Annotations({obj: obj_annotation}, {}, [], [net]), True),
         (lambda obj, obj_annotation, connection, net:
            Annotations({}, {}, [obj_annotation], [net]), False),
         ]
    )
    def test_standard(self, make_annotation, exp_params):
        """Test the standard building from annotations.
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
            model = SpiNNakerModel.from_annotations(model, annotation)

        # Check the builder was called correctly
        if exp_params:
            obj_builder.assert_called_once_with(
                obj, obj_params, obj_annotation
            )
        else:
            obj_builder.assert_called_once_with(
                None, None, obj_annotation
            )

        # Check the return values were correctly used
        assert len(model.nets) == 1
        assert model.nets[0].source is obj_vertex
        assert model.nets[0].sinks == [obj_vertex]
        assert model.nets[0].weight == 77
        assert model.nets[0].keyspace is net.keyspace

        assert model.vertices == [obj_vertex]
        assert model.groups == dict()
        assert model.load_functions == [obj_load]
        assert model.before_simulation_functions == [obj_pre]
        assert model.after_simulation_functions == [obj_post]
        assert model.placements == dict()
        assert model.allocations == dict()
        assert model.application_memory == dict()
        assert model.routes == dict()

    @pytest.mark.parametrize(
        "make_annotation, exp_params",
        [(lambda obj, obj_annotation, connection, net:
            Annotations({obj: obj_annotation},
                        {connection: net}, [], []), True),
         (lambda obj, obj_annotation, connection, net:
            Annotations({obj: obj_annotation}, {}, [], [net]), True),
         (lambda obj, obj_annotation, connection, net:
            Annotations({}, {}, [obj_annotation], [net]), False),
         ]
    )
    def test_multiple_vertices(self, make_annotation, exp_params):
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
            model = SpiNNakerModel.from_annotations(model, annotation)

        # Check the builder was called correctly
        if exp_params:
            obj_builder.assert_called_once_with(
                obj, obj_params, obj_annotation
            )
        else:
            obj_builder.assert_called_once_with(
                None, None, obj_annotation
            )

        # Check the return values were correctly used
        assert model.vertices == obj_vertex
        assert model.groups == {v: 0 for v in obj_vertex}

        # Check the nets reflect the splitting of the vertices
        assert len(model.nets) == len(obj_vertex)
        for new_net in model.nets:
            assert new_net.source in obj_vertex
            assert new_net.sinks == obj_vertex
            assert new_net.keyspace is net.keyspace
            assert new_net.weight == net.weight
