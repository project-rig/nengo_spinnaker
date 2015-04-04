import mock
import pytest
from rig.bitfield import BitField

from nengo_spinnaker import intermediate_representation as ir
from nengo_spinnaker import model as nm
from nengo_spinnaker import netlist as nl


class TestModel(object):
    def test_build_standard(self):
        """Test that build methods are called, regardless of how they are
        specified and that an appropriate netlist is returned.
        """
        class OriginalObject(object):
            pass

        class IntermediateObject(object):
            pass

        class ExtraObject(object):
            pass

        # Construct the intermediate representation
        a = OriginalObject()
        a_intermediate = IntermediateObject()
        b = ExtraObject()
        source_a = nl.NetAddress(a_intermediate, nl.OutputPort.standard)
        sink_b = nl.NetAddress(b, nl.InputPort.standard)
        keyspace = mock.Mock(name="keyspace", spec=BitField)
        keyspace.length = 32
        c = ir.IntermediateNet(1234, source_a, sink_b, keyspace, False, 323)

        irn = ir.IntermediateRepresentation({a: a_intermediate}, {}, [b], [c])

        # Construct the builders
        class Vertex(object):
            pass

        a_vertex = Vertex()
        a_pre_load = mock.Mock()
        a_pre_sim = mock.Mock()
        a_post_sim = mock.Mock()
        a_builder = mock.Mock(
            name="a builder", spec_set=[],
            return_value=(a_vertex, a_pre_load, a_pre_sim, a_post_sim))

        b_vertex = Vertex()
        b_pre_load = mock.Mock()
        b_pre_sim = None  # SHOULD NOT END UP IN THE LIST!
        b_post_sim = mock.Mock()
        b_builder = mock.Mock(
            name="b builder", spec_set=[],
            return_value=(b_vertex, b_pre_load, b_pre_sim, b_post_sim))

        # Run the builder
        with mock.patch.object(nm.Model, "builders",
                               {IntermediateObject: a_builder}):
            model = nm.Model.from_intermediate_representation(
                irn, {ExtraObject: b_builder})

        # Assert we built correctly
        # Vertices
        a_builder.assert_called_once_with(a_intermediate, a, irn)
        b_builder.assert_called_once_with(b, None, irn)
        assert {a_vertex, b_vertex} == set(model.vertices)
        assert model.vertex_map[a_intermediate] is a_vertex
        assert model.vertex_map[b] is b_vertex

        # Nets
        nets = list(model.nets)
        assert len(nets) == 1
        net = model.net_map[c][0]
        assert net.source == a_vertex
        assert net.sinks == [b_vertex]
        assert net.keyspace is c.keyspace
        assert net.weight == c.weight

        # Callbacks
        assert set(model.preload_callables) == {a_pre_load, b_pre_load}
        assert set(model.presim_callables) == {a_pre_sim}
        assert set(model.postsim_callables) == {a_post_sim, b_post_sim}

    def test_build_fails_for_missing_builder(self):
        """Test that an exception is raised if there is no build method for a
        given type of object.
        """
        # Construct the intermediate representation
        b = mock.Mock()
        irn = ir.IntermediateRepresentation({}, {}, [b], [])

        # Run the builder, this should fail because there is no builder for
        # `b`.
        with mock.patch.object(nm.Model, "builders", {}), \
                pytest.raises(TypeError) as excinfo:
            nm.Model.from_intermediate_representation(irn)

        assert b.__class__.__name__ in str(excinfo.value)

    def test_build_multiple_vertices_source(self):
        """Test that multiple vertices are supported (explicitly when the
        original object would have been the source of a vertex.
        """
        class OriginalObject(object):
            pass

        class IntermediateObject(object):
            pass

        class ExtraObject(object):
            pass

        # Construct the intermediate representation
        a = OriginalObject()
        a_intermediate = IntermediateObject()
        b = ExtraObject()
        source_a = nl.NetAddress(a_intermediate, nl.OutputPort.standard)
        sink_b = nl.NetAddress(b, nl.InputPort.standard)
        keyspace = mock.Mock(name="keyspace", spec=BitField)
        keyspace.length = 32
        c = ir.IntermediateNet(1234, source_a, sink_b, keyspace, False)

        irn = ir.IntermediateRepresentation({a: a_intermediate}, {}, [b], [c])

        # Construct the builders
        class Vertex(object):
            pass

        a_vertices = [Vertex(), Vertex()]
        a_builder = mock.Mock(
            name="a builder", spec_set=[],
            return_value=(a_vertices, None, None, None))

        b_vertex = Vertex()
        b_builder = mock.Mock(
            name="b builder", spec_set=[],
            return_value=(b_vertex, None, None, None))

        # Run the builder
        with mock.patch.object(nm.Model, "builders",
                               {IntermediateObject: a_builder}):
            model = nm.Model.from_intermediate_representation(
                irn, {ExtraObject: b_builder})

        # Assert we built correctly
        # Vertices
        a_builder.assert_called_once_with(a_intermediate, a, irn)
        b_builder.assert_called_once_with(b, None, irn)
        assert {a_vertices[0], a_vertices[1], b_vertex} == set(model.vertices)

        # Nets
        assert len(list(model.nets)) == 2

        for net in model.nets:
            assert isinstance(net, nl.Net)
            assert net.source in a_vertices
            assert net.sinks == [b_vertex]
            assert net.keyspace is c.keyspace

    def test_build_multiple_vertices_sinks(self):
        """Test that multiple vertices are supported (explicitly when the
        original object would have been the source of a vertex.
        """
        class OriginalObject(object):
            pass

        class IntermediateObject(object):
            pass

        class ExtraObject(object):
            pass

        # Construct the intermediate representation
        a = OriginalObject()
        a_intermediate = IntermediateObject()
        b = ExtraObject()
        source_a = nl.NetAddress(a_intermediate, nl.OutputPort.standard)
        sink_b = nl.NetAddress(b, nl.InputPort.standard)
        keyspace = mock.Mock(name="keyspace", spec=BitField)
        keyspace.length = 32
        c = ir.IntermediateNet(1234, source_a, sink_b, keyspace, False)

        irn = ir.IntermediateRepresentation({a: a_intermediate}, {}, [b], [c])

        # Construct the builders
        class Vertex(object):
            pass

        a_vertex = Vertex()
        a_builder = mock.Mock(
            name="a builder", spec_set=[],
            return_value=(a_vertex, None, None, None))

        b_vertices = [Vertex(), Vertex()]
        b_builder = mock.Mock(
            name="b builder", spec_set=[],
            return_value=(b_vertices, None, None, None))

        # Run the builder
        with mock.patch.object(nm.Model, "builders",
                               {IntermediateObject: a_builder}):
            model = nm.Model.from_intermediate_representation(
                irn, {ExtraObject: b_builder})

        # Assert we built correctly
        # Vertices
        a_builder.assert_called_once_with(a_intermediate, a, irn)
        b_builder.assert_called_once_with(b, None, irn)
        assert {a_vertex, b_vertices[0], b_vertices[1]} == set(model.vertices)

        # Nets
        nets = list(model.nets)
        assert len(nets) == 1
        net = nets[0]
        assert isinstance(net, nl.Net)
        assert net.source == a_vertex
        assert net.sinks == b_vertices
        assert net.keyspace is c.keyspace
