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

        a_extra_objs = [mock.Mock()]
        a_extra_conns = [mock.Mock()]
        b_extra_objs = [mock.Mock()]
        b_extra_conns = [mock.Mock()]

        irn = ir.IntermediateRepresentation({}, {}, [], [])

        source_getters = {
            a.__class__:
                lambda x, y: (x.pre_obj, {"extra_objects": a_extra_objs,
                                          "extra_connections": a_extra_conns})
        }
        sink_getters = {
            b.__class__:
                lambda x, y: (x.post_obj, {"extra_objects": b_extra_objs,
                                           "extra_connections": b_extra_conns})
        }

        # Build the connection
        ic, extra_objs, extra_conns = ir._get_intermediate_net(
            source_getters, sink_getters, c, irn)

        assert ic.source is a
        assert ic.sink is b
        assert ic.seed is not None
        assert ic.keyspace is None
        assert not ic.latching  # Receiving buffers should clear every timestep
        assert extra_objs == a_extra_objs + b_extra_objs
        assert extra_conns == a_extra_conns + b_extra_conns

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
            a.__class__: lambda x, y: (x.pre_obj, {})}
        sink_getters = {
            b.__class__: lambda x, y: (x.post_obj, {})}

        # Build the connection
        ic, extra_objs, extra_conns = ir._get_intermediate_net(
            source_getters, sink_getters, c, irn)

        # Assert the seed made it
        assert ic.seed == c.seed

    def test_unknown_keyword(self):
        """The simple case where all we have to do is get the seed, the source,
        the sink and combine any extra objects and connections.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)
        c.seed = 303.0

        irn = ir.IntermediateRepresentation({}, {}, [], [])

        source_getters = {
            a.__class__: lambda x, y: (x.pre_obj, {"spam": "eggs"})}
        sink_getters = {
            b.__class__: lambda x, y: (x.post_obj, {})}

        # Build the connection
        with pytest.raises(NotImplementedError) as excinfo:
            ir._get_intermediate_net(source_getters, sink_getters, c, irn)
        assert "spam" in str(excinfo.value)

        source_getters = {
            a.__class__: lambda x, y: (x.pre_obj, {})}
        sink_getters = {
            b.__class__: lambda x, y: (x.post_obj, {"eggs": "spam"})}

        # Build the connection
        with pytest.raises(NotImplementedError) as excinfo:
            ir._get_intermediate_net(source_getters, sink_getters, c, irn)
        assert "eggs" in str(excinfo.value)

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
            a.__class__: lambda x, y: (x.pre_obj, dict(keyspace=a_ks))}
        sink_getters = {
            b.__class__: lambda x, y: (x.post_obj, {})}

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
            a.__class__: lambda x, y: (x.pre_obj, {})}
        sink_getters = {
            b.__class__: lambda x, y: (x.post_obj, dict(keyspace=b_ks))}

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
            a.__class__: lambda x, y: (x.pre_obj, dict(keyspace=a_ks))}
        sink_getters = {
            b.__class__: lambda x, y: (x.post_obj, dict(keyspace=b_ks))}

        # Build the connection
        with pytest.raises(NotImplementedError) as excinfo:
            ic, extra_objs, extra_conns = ir._get_intermediate_net(
                source_getters, sink_getters, c, irn)
        assert "keyspace" in str(excinfo.value)

    @pytest.mark.parametrize(
        "source_getters, sink_getters",
        [({ObjTypeA: lambda x, y: (None, {})},
          {ObjTypeB: lambda x, y: (x.post_obj, {})}),
         ({ObjTypeA: lambda x, y: (x.pre_obj, {})},
          {ObjTypeB: lambda x, y: (None, {})}),
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

    @pytest.mark.parametrize(
        "source_getters, sink_getters",
        [({ObjTypeA: lambda x, y: (x.pre_obj, {"latching": True})},
          {ObjTypeB: lambda x, y: (x.post_obj, {})}),
         ({ObjTypeA: lambda x, y: (x.pre_obj, {})},
          {ObjTypeB: lambda x, y: (x.post_obj, {"latching": True})}),
         ]
    )
    def test_requires_latching_net(self, source_getters, sink_getters):
        """Test that the net is marked as latching if the source or sink
        requires it.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        irn = ir.IntermediateRepresentation({}, {}, [], [])

        # Build the connection
        ic, extra_objs, extra_conns = ir._get_intermediate_net(
            source_getters, sink_getters, c, irn)
        assert ic.latching


class TestGetIntermediateProbe(object):
    """Test that getting intermediate probes calls a method associated with the
    target of the probe.
    """
    def test_get_intermediate_probe_no_seed(self):
        class Obj(object):
            pass

        # Create the probe
        probe = mock.Mock(spec_set=["target"], name="Probe")
        probe.target = Obj()

        # Create the probe getter
        probe_getter = mock.Mock(spec_set=[], name="probe getter")
        probe_getter.return_value = mock.Mock(spec_set=[], name="retval")
        probe_getters = {Obj: probe_getter}

        # Create a mock IRN
        irn = mock.Mock(spec_set=[], name="irn")

        # Check that call works as expected
        assert (ir._get_intermediate_probe(probe_getters, probe, irn) is
                probe_getter.return_value)
        assert probe_getter.call_count == 1
        assert probe_getter.call_args[0][0] is probe
        assert probe_getter.call_args[0][1] is not None
        assert probe_getter.call_args[0][2] is irn

    def test_get_intermediate_probe_with_seed(self):
        class Obj(object):
            pass

        # Create the probe
        probe = mock.Mock(spec_set=["target", "seed"], name="Probe")
        probe.target = Obj()
        probe.seed = 303.0

        # Create the probe getter
        probe_getter = mock.Mock(spec_set=[], name="probe getter")
        probe_getter.return_value = mock.Mock(spec_set=[], name="retval")
        probe_getters = {Obj: probe_getter}

        # Create a mock IRN
        irn = mock.Mock(spec_set=[], name="irn")

        # Check that call works as expected
        assert (ir._get_intermediate_probe(probe_getters, probe, irn) is
                probe_getter.return_value)
        assert probe_getter.call_count == 1
        assert probe_getter.call_args[0][0] is probe
        assert probe_getter.call_args[0][1] == probe.seed
        assert probe_getter.call_args[0][2] is irn

    def test_get_intermediate_probe_fails(self):
        class Obj(object):
            pass

        # Create the probe
        probe = mock.Mock(spec_set=["target"], name="Probe")
        probe.target = Obj()
        probe_getters = {}

        irn = mock.Mock(spec_set=[], name="irn")

        # Check that call fails with a TypeError
        with pytest.raises(TypeError) as excinfo:
            ir._get_intermediate_probe(probe_getters, probe, irn)
        assert "Obj" in str(excinfo.value)


class TestIntermediateObject(object):
    def test_init(self):
        """Test that the __init__ sets the seed."""
        seed = 32345
        obj = mock.Mock(spec_set=[])

        # Create the intermediate object, ensure that it is sensible
        o = ir.IntermediateObject(obj, seed, [1, 2, 3])
        assert o.seed == seed
        assert o.constraints == [1, 2, 3]


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
        (nl.NetAddress(obj_map[a], nl.OutputPort.standard), {})
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
        (nl.NetAddress(obj_map[b], nl.InputPort.standard), {})
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
        assert o.constraints == list()
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
            (nl.NetAddress(obj_map[b], nl.InputPort.standard), {})
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
        assert ir.get_ensemble_sink(c, irn) == (None, {})

        # But the Node values are added into the intermediate representation
        # for the ensemble with the connection transform and function applied.
        assert np.all(obj_map[b].direct_input ==
                      np.dot(full_transform(c, slice_pre=False),
                             c.function(a.output[c.pre_slice])))

        # For the next connection assert that we again don't add a connection
        # and that the direct input is increased.
        assert ir.get_ensemble_sink(d, irn) == (None, {})
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
            (nl.NetAddress(obj_map[b], nl.InputPort.neurons), {})
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
            (nl.NetAddress(obj_map[b], nl.InputPort.global_inhibition), {})
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


def test_get_output_probe():
    """Test building probes for Ensemble or Node-type objects."""
    with nengo.Network():
        a = nengo.Node(lambda t: t**2, size_out=1, size_in=0)
        p = nengo.Probe(a)

    # Get the IR for the Node
    ir_a = ir.IntermediateObject(a, 1101)

    # Building the probe should return an IntermediateObject for the probe and
    # a new Net from the Node to the Probe.
    new_obj, new_objs, new_conns = ir.get_output_probe(
            p, 1159, ir.IntermediateRepresentation({a: ir_a}, {}, (), ()))

    assert new_obj.seed == 1159
    assert new_obj.constraints == list()

    assert new_objs == list()

    assert len(new_conns) == 1
    new_conn = new_conns[0]
    assert new_conn.source == nl.NetAddress(ir_a, nl.OutputPort.standard)
    assert new_conn.sink == nl.NetAddress(new_obj, nl.InputPort.standard)
    assert new_conn.keyspace is None
    assert not new_conn.latching


class TestGetEnsembleProbe(object):
    def test_get_output_probe(self):
        """Test building probes for Ensemble or Node-type objects."""
        with nengo.Network():
            a = nengo.Ensemble(300, 4)
            p = nengo.Probe(a)

        # Get the IR for the Node
        ir_a = ir.IntermediateEnsemble(a, 1101)

        # Building the probe should return an IntermediateObject for the probe
        # and a new Net from the Ensemble to the Probe.
        new_obj, new_objs, new_conns = ir.get_ensemble_probe(
                p, 1159, ir.IntermediateRepresentation({a: ir_a}, {}, (), ()))

        assert new_obj.seed == 1159
        assert new_obj.constraints == list()

        assert new_objs == list()

        assert len(new_conns) == 1
        new_conn = new_conns[0]
        assert new_conn.source == nl.NetAddress(ir_a, nl.OutputPort.standard)
        assert new_conn.sink == nl.NetAddress(new_obj, nl.InputPort.standard)
        assert new_conn.keyspace is None
        assert not new_conn.latching

    def test_get_input_probe(self):
        with nengo.Network():
            a = nengo.Ensemble(300, 4)
            p = nengo.Probe(a, attr="input")

        # Get the IR for the Node
        ir_a = ir.IntermediateEnsemble(a, 1101)

        with pytest.raises(NotImplementedError):
            ir.get_ensemble_probe(
                p, 1159, ir.IntermediateRepresentation({a: ir_a}, {}, (), ()))


def test_get_neurons_probe():
    """Test building probes for Neuron-type objects."""
    with nengo.Network():
        a = nengo.Ensemble(300, 2)
        p = nengo.Probe(a.neurons)

    # Get the IR for the ensemble
    ir_a = ir.IntermediateEnsemble(a, 1105)
    assert ir_a.local_probes == list()

    # Building the probe should just add it to the intermediate representation
    # for `a`'s list of local probes.
    assert (
        ir.get_neurons_probe(
            p, 3345, ir.IntermediateRepresentation({a: ir_a}, {}, (), ())) ==
        (None, [], [])
    )
    assert ir_a.local_probes == [p]


class TestIntermediateRepresentation(object):
    def test_get_nets_starting_at_ending_at(self):
        """Test retrieving nets which begin or end at a given intermediate
        object.
        """
        class Obj(object):
            pass

        # Create objects and their intermediate representations
        a = Obj()
        b = Obj()
        ir_a = ir.IntermediateObject(a, 1)
        ir_b = ir.IntermediateObject(b, 2)

        # Create some nets, some with and some without matching connections
        conn_ab1 = mock.Mock(spec_set=[], name="A->B")
        net_ab1 = ir.IntermediateNet(
            3, nl.NetAddress(ir_a, nl.OutputPort.standard),
            nl.NetAddress(ir_b, nl.InputPort.standard), None, False
        )

        net_ab2 = ir.IntermediateNet(
            3, nl.NetAddress(ir_a, nl.OutputPort.neurons),
            nl.NetAddress(ir_b, nl.InputPort.standard), None, False
        )

        conn_ba1 = mock.Mock(spec_set=[], name="B->A")
        net_ba1 = ir.IntermediateNet(
            3, nl.NetAddress(ir_b, nl.OutputPort.standard),
            nl.NetAddress(ir_a, nl.InputPort.standard), None, False
        )

        # Construct the intermediate representation
        irn = ir.IntermediateRepresentation(
            {a: ir_a, b: ir_b}, {conn_ab1: net_ab1, conn_ba1: net_ba1},
            [], [net_ab2]
        )

        # Retrieve the nets starting at a
        net_ax = irn.get_nets_starting_at(ir_a)
        assert net_ax[nl.OutputPort.standard] == {net_ab1: conn_ab1}
        assert net_ax[nl.OutputPort.neurons] == {net_ab2: None}

        # Retrieve the nets starting at b
        net_bx = irn.get_nets_starting_at(ir_b)
        assert net_bx[nl.OutputPort.standard] == {net_ba1: conn_ba1}
        assert net_bx[nl.OutputPort.neurons] == {}

        # Retrieve the nets ending at a
        net_xa = irn.get_nets_ending_at(ir_a)
        assert net_xa[nl.InputPort.standard] == {net_ba1: conn_ba1}

        # Retrieve the nets ending at b
        net_xb = irn.get_nets_ending_at(ir_b) 
        assert net_xb[nl.InputPort.standard] == {net_ab1: conn_ab1,
                                                 net_ab2: None}


# @pytest.mark.skipif(True, reason="functional")
class TestIntermediateRepresentationFunctional(object):
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

        # The Ensemble and each probe should exist in the object map (but the
        # probes should map to None).
        assert set(intermediate.object_map.keys()) == {a, p0, p1}
        assert set(intermediate.object_map[a].local_probes) == {p0, p1}
        assert intermediate.object_map[p0] is None
        assert intermediate.object_map[p1] is None

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
