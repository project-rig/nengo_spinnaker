import mock
import nengo
from nengo.utils.builder import objs_and_connections, full_transform
import numpy as np
import pytest

from nengo_spinnaker import netlist as nl
from nengo_spinnaker import intermediate_representation as ir


class TestGetIntermediateObject(object):
    """Test building intermediate objects."""
    def test_object_has_no_seed(self):
        # Construct an object, a builder and a mapping which indicates which
        # builder should be used for the object class.
        obj = mock.Mock(spec_set=[], name="object")

        obj_builder = mock.Mock(spec_set=[], name="builder")
        obj_builder.return_value = mock.Mock(spec_set=[], name="intermediate")

        builders = {obj.__class__: obj_builder}

        # Get the built object, assert that the builder is called appropriately
        assert (ir._get_intermediate_object(builders, obj) ==
                obj_builder.return_value)
        obj_builder.call_count == 1
        assert obj_builder.call_args[0][0] is obj
        assert obj_builder.call_args[0][1] is not None

    def test_object_has_seed(self):
        # Construct an object, a builder and a mapping which indicates which
        # builder should be used for the object class.
        obj = mock.Mock(spec_set=["seed"], name="object")

        # floats should NEVER be provided unless they're provided by the
        # object.
        obj.seed = 343.0

        obj_builder = mock.Mock(spec_set=[], name="builder")
        obj_builder.return_value = mock.Mock(spec_set=[], name="intermediate")

        builders = {obj.__class__: obj_builder}

        # Get the built object, assert that the builder is called appropriately
        assert (ir._get_intermediate_object(builders, obj) ==
                obj_builder.return_value)
        obj_builder.call_count == 1
        assert obj_builder.call_args[0][0] is obj
        assert obj_builder.call_args[0][1] == obj.seed

    def test_no_builder(self):
        with pytest.raises(TypeError) as excinfo:
            ir._get_intermediate_object({}, mock.Mock())
        assert "Mock" in str(excinfo.value)


class TestGetIntermediateEndPoint(object):
    """Test getting sinks and sources for intermediate representations of
    connections.
    """
    def test_invalid_call(self):
        with pytest.raises(ValueError):
            ir._get_intermediate_endpoint(0, {}, None, None)

    def test_get_intermediate_end_point_source(self):
        # Construct a mock endpoint, a mock connection, a mock getter and a
        # mock IR.
        end_obj = mock.Mock(spec_set=[], name="end object")
        conn = mock.Mock(spec_set=["pre_obj"], name="connection")
        conn.pre_obj = end_obj

        irn = mock.Mock(spec_set=[], name="irn")

        endpoint_getter = mock.Mock(spec_set=[], name="getter")
        endpoint_getter.return_value = mock.Mock(spec_set=[], name="endpoint")

        endpoint_getters = {end_obj.__class__: endpoint_getter}

        # Make the call
        retval = ir._get_intermediate_endpoint(
            ir._EndpointType.source, endpoint_getters, conn, irn)

        assert endpoint_getter.call_count == 1
        assert endpoint_getter.call_args[0] == (conn, irn)
        assert retval == endpoint_getter.return_value

    def test_get_intermediate_end_point_source_no_builder(self):
        # Construct a mock endpoint, a mock connection, a mock getter and a
        # mock IR.
        end_obj = mock.Mock(spec_set=[], name="end object")
        conn = mock.Mock(spec_set=["pre_obj"], name="connection")
        conn.pre_obj = end_obj

        irn = mock.Mock(spec_set=[], name="irn")

        endpoint_getter = mock.Mock(spec_set=[], name="getter")
        endpoint_getter.return_value = mock.Mock(spec_set=[], name="endpoint")

        # Make the call
        with pytest.raises(TypeError) as excinfo:
            ir._get_intermediate_endpoint(ir._EndpointType.source,
                                          {}, conn, irn)
        assert "Mock" in str(excinfo.value)

    def test_get_intermediate_end_point_sink(self):
        # Construct a mock endpoint, a mock connection, a mock getter and a
        # mock IR.
        end_obj = mock.Mock(spec_set=[], name="end object")
        conn = mock.Mock(spec_set=["post_obj"], name="connection")
        conn.post_obj = end_obj

        irn = mock.Mock(spec_set=[], name="irn")

        endpoint_getter = mock.Mock(spec_set=[], name="getter")
        endpoint_getter.return_value = mock.Mock(spec_set=[], name="endpoint")

        endpoint_getters = {end_obj.__class__: endpoint_getter}

        # Make the call
        retval = ir._get_intermediate_endpoint(
            ir._EndpointType.sink, endpoint_getters, conn, irn)

        assert endpoint_getter.call_count == 1
        assert endpoint_getter.call_args[0] == (conn, irn)
        assert retval == endpoint_getter.return_value

    def test_get_intermediate_end_point_sink_no_builder(self):
        # Construct a mock endpoint, a mock connection, a mock getter and a
        # mock IR.
        end_obj = mock.Mock(spec_set=[], name="end object")
        conn = mock.Mock(spec_set=["post_obj"], name="connection")
        conn.post_obj = end_obj

        irn = mock.Mock(spec_set=[], name="irn")

        endpoint_getter = mock.Mock(spec_set=[], name="getter")
        endpoint_getter.return_value = mock.Mock(spec_set=[], name="endpoint")

        # Make the call
        with pytest.raises(TypeError) as excinfo:
            ir._get_intermediate_endpoint(ir._EndpointType.sink, {}, conn, irn)
        assert "Mock" in str(excinfo.value)
        assert "sink" in str(excinfo.value)


class TestGetIntermediateNet(object):
    """Tests the construction of intermediate nets from connections.
    """
    class ObjTypeA(object):
        pass

    class ObjTypeB(object):
        pass

    class FauxConnection(object):
        def __init__(self, pre, post):
            self.pre_obj = pre
            self.post_obj = post

    def test_standard_no_seed(self):
        """The simple case where all we have to do is get the seed, the source,
        the sink and combine any extra objects and connections.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        irn = ir.IntermediateRepresentation({}, {}, [], [])

        source_getters = {
            a.__class__: lambda x, y: (x.pre_obj, None, None, None)}
        sink_getters = {
            b.__class__: lambda x, y: (x.post_obj, None, None, None)}

        # Build the connection
        ic, extra_objs, extra_conns = ir._get_intermediate_net(
            source_getters, sink_getters, c, irn)

        assert ic.source is a
        assert ic.sink is b
        assert ic.seed is not None
        assert ic.keyspace is None
        assert (extra_objs, extra_conns) == ([], [])

    def test_standard_with_seed(self):
        """The simple case where all we have to do is get the seed, the source,
        the sink and combine any extra objects and connections.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)
        c.seed = 303.0

        irn = ir.IntermediateRepresentation({}, {}, [], [])

        source_getters = {
            a.__class__: lambda x, y: (x.pre_obj, None, None, None)}
        sink_getters = {
            b.__class__: lambda x, y: (x.post_obj, None, None, None)}

        # Build the connection
        ic, extra_objs, extra_conns = ir._get_intermediate_net(
            source_getters, sink_getters, c, irn)

        # Assert the seed made it
        assert ic.seed == c.seed

    def test_pre_supplied_keyspace(self):
        """Test that when only the source provides a keyspace it is applied to
        the net.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        a_ks = mock.Mock(spec_set=[], name="ks_a")

        irn = ir.IntermediateRepresentation({}, {}, [], [])

        source_getters = {
            a.__class__: lambda x, y: (x.pre_obj, a_ks, None, None)}
        sink_getters = {
            b.__class__: lambda x, y: (x.post_obj, None, None, None)}

        # Build the connection
        ic, extra_objs, extra_conns = ir._get_intermediate_net(
            source_getters, sink_getters, c, irn)

        # Assert the keyspace made it
        assert ic.keyspace is a_ks

    def test_post_supplied_keyspace(self):
        """Test that when only the sink provides a keyspace it is applied to
        the net.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        b_ks = mock.Mock(spec_set=[], name="ks_b")

        irn = ir.IntermediateRepresentation({}, {}, [], [])

        source_getters = {
            a.__class__: lambda x, y: (x.pre_obj, None, None, None)}
        sink_getters = {
            b.__class__: lambda x, y: (x.post_obj, b_ks, None, None)}

        # Build the connection
        ic, extra_objs, extra_conns = ir._get_intermediate_net(
            source_getters, sink_getters, c, irn)

        # Assert the keyspace made it
        assert ic.keyspace is b_ks

    def test_supplied_keyspace_collision(self):
        """Test that when keyspaces are provided by BOTH the source and the
        sink return keyspaces an error is raised.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        a_ks = mock.Mock(spec_set=[], name="ks_a")
        b_ks = mock.Mock(spec_set=[], name="ks_b")

        irn = ir.IntermediateRepresentation({}, {}, [], [])

        source_getters = {
            a.__class__: lambda x, y: (x.pre_obj, a_ks, None, None)}
        sink_getters = {
            b.__class__: lambda x, y: (x.post_obj, b_ks, None, None)}

        # Build the connection
        with pytest.raises(NotImplementedError) as excinfo:
            ic, extra_objs, extra_conns = ir._get_intermediate_net(
                source_getters, sink_getters, c, irn)
        assert "keyspace" in str(excinfo.value)

    @pytest.mark.parametrize(
        "source_getters, sink_getters",
        [({ObjTypeA: lambda x, y: (None, ) * 4},
          {ObjTypeB: lambda x, y: (x.post_obj, ) + (None, ) * 3}),
         ({ObjTypeA: lambda x, y: (x.post_obj, ) + (None, ) * 3},
          {ObjTypeB: lambda x, y: (None, ) * 4}),
         ]
    )
    def test_connection_rejected(self, source_getters, sink_getters):
        """Test that no connection is inserted if it is rejected by the
        pre-object or the post-object.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        irn = ir.IntermediateRepresentation({}, {}, [], [])

        # Build the connection
        ic, extra_objs, extra_conns = ir._get_intermediate_net(
            source_getters, sink_getters, c, irn)
        assert ic is None


class TestIntermediateObject(object):
    def test_init(self):
        """Test that the __init__ sets the seed."""
        seed = 32345
        obj = mock.Mock(spec_set=[])

        # Create the intermediate object, ensure that it is sensible
        o = ir.IntermediateObject(obj, seed)
        assert o.seed == seed


def test_get_source_standard():
    """Test that get_source_standard just does a look up in the object map
    dictionary and uses OutputPort.standard.
    """
    with nengo.Network():
        a = nengo.Ensemble(300, 4)
        b = nengo.Ensemble(300, 2)
        c = nengo.Connection(a[:2], b)

    obj_map = {
        a: mock.Mock(name="ir_a", spec_set=[]),
        b: mock.Mock(name="ir_b", spec_set=[]),
    }

    irn = ir.IntermediateRepresentation(obj_map, {}, [], [])
    assert (
        ir.get_source_standard(c, irn) ==
        (nl.NetAddress(obj_map[a], nl.OutputPort.standard), None, None, None)
    )


def test_get_sink_standard():
    """Test that get_sink_standard just does a look up in the object map
    dictionary and uses InputPort.standard.
    """
    with nengo.Network():
        a = nengo.Ensemble(300, 4)
        b = nengo.Ensemble(300, 2)
        c = nengo.Connection(a[:2], b)

    obj_map = {
        a: mock.Mock(name="ir_a", spec_set=[]),
        b: mock.Mock(name="ir_b", spec_set=[]),
    }

    irn = ir.IntermediateRepresentation(obj_map, {}, [], [])
    assert (
        ir.get_sink_standard(c, irn) ==
        (nl.NetAddress(obj_map[b], nl.InputPort.standard), None, None, None)
    )


class TestIntermediateEnsemble(object):
    @pytest.mark.parametrize(
        "size_in",
        [1, 4, 9]
    )
    def test_init(self, size_in):
        """Test that the init correctly initialise the direct input and the
        list of local probes.
        """
        seed = 34567
        with nengo.Network():
            a = nengo.Ensemble(100, size_in)

        # Create the intermediate representation
        o = ir.IntermediateEnsemble(a, seed)
        assert o.seed == seed
        assert np.all(o.direct_input == np.zeros(size_in))
        assert o.local_probes == list()


class TestGetEnsembleSink(object):
    def test_get_sink_standard(self):
        """Test that get_sink_standard just does a look up in the object map
        dictionary and uses InputPort.standard.
        """
        with nengo.Network():
            a = nengo.Ensemble(300, 4)
            b = nengo.Ensemble(300, 2)
            c = nengo.Connection(a[:2], b)

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: mock.Mock(name="ir_b", spec_set=[]),
        }

        irn = ir.IntermediateRepresentation(obj_map, {}, [], [])
        assert (
            ir.get_ensemble_sink(c, irn) ==
            (nl.NetAddress(obj_map[b], nl.InputPort.standard), None,
             None, None)
        )

    def test_get_sink_constant_node(self):
        """Test that if the "pre" object is a constant valued Node that None is
        returned and that the IntermediateEnsemble is modified.
        """
        with nengo.Network():
            a = nengo.Node([2.0, -0.25])
            b = nengo.Ensemble(300, 2)

            c = nengo.Connection(a[0], b[1], transform=5,
                                 function=lambda x: x+2)
            d = nengo.Connection(a[0], b[0])

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: ir.IntermediateEnsemble(b, None)
        }

        # We don't return a sink (None means "no connection required")
        irn = ir.IntermediateRepresentation(obj_map, {}, [], [])
        assert ir.get_ensemble_sink(c, irn) == (None, None, None, None)

        # But the Node values are added into the intermediate representation
        # for the ensemble with the connection transform and function applied.
        assert np.all(obj_map[b].direct_input ==
                      np.dot(full_transform(c, slice_pre=False),
                             c.function(a.output[c.pre_slice])))

        # For the next connection assert that we again don't add a connection
        # and that the direct input is increased.
        assert ir.get_ensemble_sink(d, irn) == (None, None, None, None)
        assert np.all(obj_map[b].direct_input ==
                      np.dot(full_transform(c, slice_pre=False),
                             c.function(a.output[c.pre_slice])) +
                      np.dot(full_transform(d), a.output))


class TestGetNeuronsSink(object):
    def test_neurons_to_neurons(self):
        """Test that get_neurons_sink correctly returns the Ensemble as the
        object and InputPort.neurons as the port.
        """
        with nengo.Network():
            a = nengo.Ensemble(300, 4)
            b = nengo.Ensemble(300, 2)
            c = nengo.Connection(a.neurons, b.neurons)

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: mock.Mock(name="ir_b", spec_set=[]),
        }

        irn = ir.IntermediateRepresentation(obj_map, {}, [], [])
        assert (
            ir.get_neurons_sink(c, irn) ==
            (nl.NetAddress(obj_map[b], nl.InputPort.neurons), None,
             None, None)
        )

    @pytest.mark.parametrize(
        "a",  # a is the originator of a connection into some neurons
        [nengo.Node(lambda t: t**2, size_in=0, size_out=1,
                    add_to_container=False),
         nengo.Ensemble(100, 1, add_to_container=False),
         ]
    )
    def test_global_inhibition(self, a):
        """Test that get_neurons_sink correctly returns the target as the
        Ensemble and the port as global_inhibition.
        """
        b = nengo.Ensemble(100, 2, add_to_container=False)
        c = nengo.Connection(a, b.neurons, transform=[[1]]*b.n_neurons,
                             add_to_container=False)

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: mock.Mock(name="ir_b", spec_set=[]),
        }

        irn = ir.IntermediateRepresentation(obj_map, {}, [], [])
        assert (
            ir.get_neurons_sink(c, irn) ==
            (nl.NetAddress(obj_map[b], nl.InputPort.global_inhibition), None,
             None, None)
        )

    def test_other(self):
        with nengo.Network():
            a = nengo.Ensemble(300, 1)
            b = nengo.Ensemble(300, 2)
            c = nengo.Connection(a, b.neurons, function=lambda x: [x]*300)

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: mock.Mock(name="ir_b", spec_set=[]),
        }

        irn = ir.IntermediateRepresentation(obj_map, {}, [], [])
        with pytest.raises(NotImplementedError):
            ir.get_neurons_sink(c, irn)


@pytest.mark.skipif(True, reason="functional")
class TestIntermediateRepresentation(object):
    """Test the generation of intermediate representations.

    **FUNCTIONAL TESTS**
    """
    def test_ensemble_to_ensemble_connections(self):
        """Test building an intermediate representation for a network
        containing only two simple Ensemble->Ensemble connections.
        """
        # Construct the network
        net = nengo.Network()
        with net:
            a = nengo.Ensemble(100, 4)
            a.seed = 20120804
            b = nengo.Ensemble(100, 3)

            c = nengo.Connection(a[2:4], b[1:3])
            c.seed = 19910727
            d = nengo.Connection(b[0], a[3])

        # Get objects and connections, then build the intermediate
        # representation
        objs, conns = objs_and_connections(net)
        intermediate = ir.IntermediateRepresentation.from_objs_conns_probes(
            objs, conns, net.probes)
        assert len(intermediate.extra_objects) == 0
        assert len(intermediate.extra_connections) == 0

        # Assert the intermediate representation contains intermediate
        # representations from the Ensembles and the connection and nothing
        # else.
        assert len(intermediate.object_map) == 2

        int_a = intermediate.object_map[a]
        assert int_a.seed == a.seed
        assert np.all(int_a.direct_input == np.zeros(a.size_in))
        assert int_a.local_probes == list()

        int_b = intermediate.object_map[b]
        assert int_b.seed is not None
        assert np.all(int_b.direct_input == np.zeros(b.size_in))
        assert int_b.local_probes == list()

        assert len(intermediate.connection_map) == 2
        int_ab = intermediate.connection_map[c]
        assert int_ab.seed == c.seed
        assert int_ab.source.object is int_a
        assert int_ab.source.port is ir.OutputPort.standard
        assert int_ab.sink.object is int_b
        assert int_ab.sink.port is ir.InputPort.standard

        int_ba = intermediate.connection_map[d]
        assert int_ba.seed is not None
        assert int_ba.source.object is int_b
        assert int_ba.source.port is ir.OutputPort.standard
        assert int_ba.sink.object is int_a
        assert int_ba.sink.port is ir.InputPort.standard

    def test_probe_spike_voltage(self):
        """Test that spike and voltage probes get added as "local probes" to
        intermediate ensembles.
        """
        net = nengo.Network()
        with net:
            a = nengo.Ensemble(100, 5)
            p0 = nengo.Probe(a.neurons)
            p1 = nengo.Probe(a.neurons, "voltage")

        # Get objects and connections, then build the intermediate
        # representation
        objs, conns = objs_and_connections(net)
        intermediate = ir.IntermediateRepresentation.from_objs_conns_probes(
            objs, conns, net.probes)

        # Only the ensemble should exist and it should have a probe
        assert len(intermediate.object_map) == 1
        assert set(intermediate.object_map[a].local_probes) == {p0, p1}

    def test_node_to_ensemble_connection(self):
        # Construct the network
        net = nengo.Network()
        with net:
            a = nengo.Node(lambda t: t**2)
            b = nengo.Ensemble(100, 3)
            c = nengo.Connection(a, b[2])

        # Get objects and connections, then build the intermediate
        # representation
        objs, conns = objs_and_connections(net)
        intermediate = ir.IntermediateRepresentation.from_objs_conns_probes(
            objs, conns, net.probes)

        # Check that the object map contains an Ensemble and a Node
        assert len(intermediate.object_map) == 2

        int_a = intermediate.object_map[a]
        assert int_a.seed is not None

        int_b = intermediate.object_map[b]
        assert np.all(int_b.direct_input == np.zeros(b.size_in))

        # Check that the connection made it
        assert len(intermediate.connection_map) == 1
        int_ab = intermediate.connection_map[c]
        assert int_ab.source.object is int_a
        assert int_ab.source.port is ir.OutputPort.standard
        assert int_ab.sink.object is int_b
        assert int_ab.sink.port is ir.InputPort.standard

    def test_global_inhibition_connection(self):
        # Construct the network
        net = nengo.Network()
        with net:
            a = nengo.Ensemble(100, 1)
            b = nengo.Ensemble(100, 3)
            c = nengo.Connection(a, b.neurons, transform=[[1.0]]*b.n_neurons)

        # Get objects and connections, then build the intermediate
        # representation
        objs, conns = objs_and_connections(net)
        intermediate = ir.IntermediateRepresentation.from_objs_conns_probes(
            objs, conns, net.probes)

        # Check that the object map contains an Ensemble and a Node
        assert len(intermediate.object_map) == 2

        int_a = intermediate.object_map[a]
        assert np.all(int_a.direct_input == np.zeros(a.size_in))

        int_b = intermediate.object_map[b]
        assert np.all(int_b.direct_input == np.zeros(b.size_in))

        # Check that the connection made it as a global inhibition connection
        assert len(intermediate.connection_map) == 1
        int_ab = intermediate.connection_map[c]
        assert int_ab.source.object is int_a
        assert int_ab.source.port is ir.OutputPort.standard
        assert int_ab.sink.object is int_b
        assert int_ab.sink.port is ir.InputPort.global_inhibition
