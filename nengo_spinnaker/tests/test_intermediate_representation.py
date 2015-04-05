import mock
import nengo
import pytest

from nengo_spinnaker import intermediate_representation as ir


class TestSinkOrSourceSpecification(object):
    def test_default_args(self):
        sink = mock.Mock(name="sink", spec_set=[ir.NetAddress])
        ss = ir.soss(sink)

        assert ss.target is sink
        assert ss.extra_objects == list()
        assert ss.extra_nets == list()
        assert ss.keyspace is None
        assert ss.latching is False
        assert ss.weight is None

    def test_non_default_args(self):
        sink = mock.Mock(name="sink")
        extra_obj = mock.Mock(name="extra object")
        extra_net = mock.Mock(name="extra net")
        keyspace = mock.Mock(name="keyspace")
        latching = True
        weight = 5

        ss = ir.soss(sink, extra_objects=[extra_obj], extra_nets=[extra_net],
                     keyspace=keyspace, latching=latching, weight=weight)
        
        assert ss.target is sink
        assert ss.extra_objects == [extra_obj]
        assert ss.extra_nets == [extra_net]
        assert ss.keyspace is keyspace
        assert ss.latching is latching
        assert ss.weight == weight


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
        with mock.patch.object(ir.IntermediateRepresentation,
                               "object_builders", builders):
            irn = ir.IntermediateRepresentation.from_objs_conns_probes(
                [obj], [], [])

        assert (irn.object_map[obj] == obj_builder.return_value)
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
        with mock.patch.object(ir.IntermediateRepresentation,
                               "object_builders", builders):
            irn = ir.IntermediateRepresentation.from_objs_conns_probes(
                [obj], [], [])

        assert (irn.object_map[obj] == obj_builder.return_value)
        obj_builder.call_count == 1
        assert obj_builder.call_args[0][0] is obj
        assert obj_builder.call_args[0][1] == obj.seed

    def test_no_builder(self):
        obj = mock.Mock()

        with mock.patch.object(ir.IntermediateRepresentation,
                               "object_builders", {}), \
                pytest.raises(TypeError) as excinfo:
            ir._get_intermediate_object({}, mock.Mock())

        assert obj.__class__.__name__ in str(excinfo.value)

    def test_object_extra_builder(self):
        # Construct an object, a builder and a mapping which indicates which
        # builder should be used for the object class.
        obj = mock.Mock(spec_set=[], name="object")

        obj_builder = mock.Mock(spec_set=[], name="builder")
        obj_builder.return_value = mock.Mock(spec_set=[], name="intermediate")

        builders = {obj.__class__: obj_builder}

        # Get the built object, assert that the builder is called appropriately
        with mock.patch.object(ir.IntermediateRepresentation,
                               "object_builders", {}):
            ir.IntermediateRepresentation.from_objs_conns_probes(
                [obj], [], [], extra_object_builders=builders
            )

        obj_builder.call_count == 1

    def test_object_is_removed(self):
        obj = mock.Mock(spec_set=[], name="object")

        obj_builder = mock.Mock(spec_set=[], name="builder")
        obj_builder.return_value = None  # Remove the item!

        builders = {obj.__class__: obj_builder}

        # Get the built object, assert that the builder is called appropriately
        with mock.patch.object(ir.IntermediateRepresentation,
                               "object_builders", builders):
            irn = ir.IntermediateRepresentation.from_objs_conns_probes(
                [obj], [], [])

        assert irn.object_map == {}


class TestGetIntermediateNet(object):
    """Tests the construction of intermediate nets from connections.
    """
    class ObjTypeA(object):
        pass

    class ObjTypeB(object):
        pass

    class FauxConnection(object):
        def __init__(self, pre, post, size_out=12):
            self.pre_obj = pre
            self.post_obj = post
            self.size_out = size_out

    def test_standard_no_seed(self):
        """The simple case where all we have to do is get the seed, the source,
        the sink and combine any extra objects and connections.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b, 33)

        a_extra_objs = [mock.Mock()]
        a_extra_conns = [mock.Mock()]
        b_extra_objs = [mock.Mock()]
        b_extra_conns = [mock.Mock()]

        source_getters = {
            a.__class__:
                lambda x, y: ir.soss(x.pre_obj, extra_objects=a_extra_objs,
                                     extra_nets=a_extra_conns)
        }
        sink_getters = {
            b.__class__:
                lambda x, y: ir.soss(x.post_obj, extra_objects=b_extra_objs,
                                     extra_nets=b_extra_conns)
        }

        # Build the connection
        with mock.patch.object(ir.IntermediateRepresentation,
                               "source_getters", source_getters), \
                mock.patch.object(ir.IntermediateRepresentation,
                                  "sink_getters", sink_getters):
            irn = ir.IntermediateRepresentation.from_objs_conns_probes(
                [], [c], [])

        ic = irn.connection_map[c]
        assert ic.source is a
        assert ic.sink is b
        assert ic.seed is not None
        assert ic.keyspace is None
        assert ic.weight == c.size_out
        assert not ic.latching  # Receiving buffers should clear every timestep
        assert irn.extra_objects == a_extra_objs + b_extra_objs
        assert irn.extra_connections == a_extra_conns + b_extra_conns

    def test_standard_with_seed(self):
        """The simple case where all we have to do is get the seed, the source,
        the sink and combine any extra objects and connections.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)
        c.seed = 303.0

        source_getters = {
            a.__class__: lambda x, y: ir.soss(x.pre_obj)}
        sink_getters = {
            b.__class__: lambda x, y: ir.soss(x.post_obj)}

        # Build the connection
        with mock.patch.object(ir.IntermediateRepresentation,
                               "source_getters", source_getters), \
                mock.patch.object(ir.IntermediateRepresentation,
                                  "sink_getters", sink_getters):
            irn = ir.IntermediateRepresentation.from_objs_conns_probes(
                [], [c], [])

        # Assert the seed made it
        assert irn.connection_map[c].seed == c.seed

    def test_pre_supplied_keyspace(self):
        """Test that when only the source provides a keyspace it is applied to
        the net.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        a_ks = mock.Mock(spec_set=[], name="ks_a")

        source_getters = {
            a.__class__: lambda x, y: ir.soss(x.pre_obj, keyspace=a_ks)}
        sink_getters = {
            b.__class__: lambda x, y: ir.soss(x.post_obj)}

        # Build the connection
        with mock.patch.object(ir.IntermediateRepresentation,
                               "source_getters", source_getters), \
                mock.patch.object(ir.IntermediateRepresentation,
                                  "sink_getters", sink_getters):
            irn = ir.IntermediateRepresentation.from_objs_conns_probes(
                [], [c], [])

        # Assert the keyspace made it
        assert irn.connection_map[c].keyspace is a_ks

    def test_post_supplied_keyspace(self):
        """Test that when only the sink provides a keyspace it is applied to
        the net.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        b_ks = mock.Mock(spec_set=[], name="ks_b")

        source_getters = {
            a.__class__: lambda x, y: ir.soss(x.pre_obj)}
        sink_getters = {
            b.__class__: lambda x, y: ir.soss(x.post_obj, keyspace=b_ks)}

        # Build the connection
        with mock.patch.object(ir.IntermediateRepresentation,
                               "source_getters", source_getters), \
                mock.patch.object(ir.IntermediateRepresentation,
                                  "sink_getters", sink_getters):
            irn = ir.IntermediateRepresentation.from_objs_conns_probes(
                [], [c], [])

        # Assert the keyspace made it
        assert irn.connection_map[c].keyspace is b_ks

    def test_supplied_keyspace_collision(self):
        """Test that when keyspaces are provided by BOTH the source and the
        sink return keyspaces an error is raised.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        a_ks = mock.Mock(spec_set=[], name="ks_a")
        b_ks = mock.Mock(spec_set=[], name="ks_b")

        source_getters = {
            a.__class__: lambda x, y: ir.soss(x.pre_obj, keyspace=a_ks)}
        sink_getters = {
            b.__class__: lambda x, y: ir.soss(x.post_obj, keyspace=b_ks)}

        # Build the connection
        with mock.patch.object(ir.IntermediateRepresentation,
                               "source_getters", source_getters), \
                mock.patch.object(ir.IntermediateRepresentation,
                                  "sink_getters", sink_getters), \
                pytest.raises(NotImplementedError) as excinfo:
            ir.IntermediateRepresentation.from_objs_conns_probes([], [c], [])

        assert "keyspace" in str(excinfo.value)

    @pytest.mark.parametrize(
        "source_getters, sink_getters",
        [({ObjTypeA: lambda x, y: ir.soss(None)},
          {ObjTypeB: lambda x, y: ir.soss(x.post_obj)}),
         ({ObjTypeA: lambda x, y: ir.soss(x.pre_obj)},
          {ObjTypeB: lambda x, y: ir.soss(None)}),
         ({ObjTypeA: lambda x, y: None},
          {ObjTypeB: lambda x, y: ir.soss(x.post_obj)}),
         ({ObjTypeA: lambda x, y: ir.soss(x.pre_obj)},
          {ObjTypeB: lambda x, y: None}),
         ]
    )
    def test_connection_rejected(self, source_getters, sink_getters):
        """Test that no connection is inserted if it is rejected by the
        pre-object or the post-object.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        with mock.patch.object(ir.IntermediateRepresentation,
                               "source_getters", source_getters), \
                mock.patch.object(ir.IntermediateRepresentation,
                                  "sink_getters", sink_getters):
            irn = ir.IntermediateRepresentation.from_objs_conns_probes(
                [], [c], [])

        assert c not in irn.connection_map

    @pytest.mark.parametrize(
        "source_getters, sink_getters",
        [({ObjTypeA: lambda x, y: ir.soss(x.pre_obj, latching=True)},
          {ObjTypeB: lambda x, y: ir.soss(x.post_obj)}),
         ({ObjTypeA: lambda x, y: ir.soss(x.pre_obj)},
          {ObjTypeB: lambda x, y: ir.soss(x.post_obj, latching=True)}),
         ]
    )
    def test_requires_latching_net(self, source_getters, sink_getters):
        """Test that the net is marked as latching if the source or sink
        requires it.
        """
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        with mock.patch.object(ir.IntermediateRepresentation,
                               "source_getters", source_getters), \
                mock.patch.object(ir.IntermediateRepresentation,
                                  "sink_getters", sink_getters):
                irn = ir.IntermediateRepresentation.from_objs_conns_probes(
                    [], [c], [])

        assert len(irn.connection_map) == 1
        assert irn.connection_map[c].latching

    @pytest.mark.parametrize(
        "source_getters, sink_getters, reason, fail_type",
        [({ObjTypeA: lambda x, y: ir.soss(x.pre_obj)}, {}, "sink", ObjTypeB),
         ({}, {ObjTypeB: lambda x, y: ir.soss(x.post_obj)},
          "source", ObjTypeA)]
    )
    def test_missing_builder(self, source_getters, sink_getters, reason,
                             fail_type):
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        with mock.patch.object(ir.IntermediateRepresentation,
                               "source_getters", source_getters), \
                mock.patch.object(ir.IntermediateRepresentation,
                                  "sink_getters", sink_getters), \
                pytest.raises(TypeError) as excinfo:
            ir.IntermediateRepresentation.from_objs_conns_probes([], [c], [])

        assert reason in str(excinfo.value)
        assert fail_type.__name__ in str(excinfo.value)

    def test_updated_builders(self):
        a = self.ObjTypeA()
        b = self.ObjTypeB()
        c = self.FauxConnection(a, b)

        a_extra_objs = [mock.Mock()]
        a_extra_conns = [mock.Mock()]
        b_extra_objs = [mock.Mock()]
        b_extra_conns = [mock.Mock()]

        source_getters = {
            a.__class__:
                lambda x, y: ir.soss(x.pre_obj, extra_objects=a_extra_objs,
                                     extra_nets=a_extra_conns)
        }
        sink_getters = {
            b.__class__:
                lambda x, y: ir.soss(x.post_obj, extra_objects=b_extra_objs,
                                     extra_nets=b_extra_conns)
        }

        # Build the connection
        with mock.patch.object(ir.IntermediateRepresentation,
                               "source_getters", {}), \
                mock.patch.object(ir.IntermediateRepresentation,
                                  "sink_getters", {}):
            ir.IntermediateRepresentation.from_objs_conns_probes(
                [], [c], [],
                extra_source_getters=source_getters,
                extra_sink_getters=sink_getters
            )


class TestGetIntermediateProbe(object):
    """Test that getting intermediate probes calls a method associated with the
    target of the probe.
    """
    class Obj(object):
        pass

    def test_get_intermediate_probe_no_seed(self):
        # Create the probe
        probe = mock.Mock(spec_set=["target"], name="Probe")
        probe.target = self.Obj()

        # Create the probe getter
        probe_getter = mock.Mock(spec_set=[], name="probe getter")
        probe_getter.return_value = (
            mock.Mock(spec_set=[], name="probe"),
            [mock.Mock(name="o")],
            [mock.Mock(name="c")]
        )
        probe_getters = {self.Obj: probe_getter}

        # Check that call works as expected
        with mock.patch.object(ir.IntermediateRepresentation,
                               "probe_builders", probe_getters):
            irn = ir.IntermediateRepresentation.from_objs_conns_probes(
                [], [], [probe])

        assert (irn.object_map[probe] == probe_getter.return_value[0])
        assert irn.extra_objects == probe_getter.return_value[1]
        assert irn.extra_connections == probe_getter.return_value[2]

        assert probe_getter.call_count == 1
        assert probe_getter.call_args[0][0] is probe
        assert probe_getter.call_args[0][1] is not None

    def test_get_intermediate_probe_with_seed(self):
        # Create the probe
        probe = mock.Mock(spec_set=["target", "seed"], name="Probe")
        probe.target = self.Obj()
        probe.seed = 303.0

        # Create the probe getter
        probe_getter = mock.Mock(spec_set=[], name="probe getter")
        probe_getter.return_value = (
            mock.Mock(spec_set=[], name="probe"),
            [mock.Mock(name="o")],
            [mock.Mock(name="c")]
        )
        probe_getters = {self.Obj: probe_getter}

        # Check that call works as expected
        with mock.patch.object(ir.IntermediateRepresentation,
                               "probe_builders", probe_getters):
            irn = ir.IntermediateRepresentation.from_objs_conns_probes(
                [], [], [probe])

        assert (irn.object_map[probe] == probe_getter.return_value[0])
        assert probe_getter.call_count == 1
        assert probe_getter.call_args[0][0] is probe
        assert probe_getter.call_args[0][1] == probe.seed

    def test_get_intermediate_probe_extra_builder(self):
        # Create the probe
        probe = mock.Mock(spec_set=["target"], name="Probe")
        probe.target = self.Obj()

        # Create the probe getter
        probe_getter = mock.Mock(spec_set=[], name="probe getter")
        probe_getter.return_value = (
            mock.Mock(spec_set=[], name="probe"),
            [mock.Mock(name="o")],
            [mock.Mock(name="c")]
        )
        probe_getters = {self.Obj: probe_getter}

        # Check that call works as expected
        with mock.patch.object(ir.IntermediateRepresentation,
                               "probe_builders", {}):
            ir.IntermediateRepresentation.from_objs_conns_probes(
                [], [], [probe], extra_probe_builders=probe_getters)

        assert probe_getter.call_count == 1

    def test_get_intermediate_probe_fails(self):
        # Create the probe
        probe = mock.Mock(spec_set=["target"], name="Probe")
        probe.target = self.Obj()

        # Check that call fails with a TypeError
        with mock.patch.object(ir.IntermediateRepresentation,
                               "probe_builders", {}), \
                pytest.raises(TypeError) as excinfo:
            ir.IntermediateRepresentation.from_objs_conns_probes(
                [], [], [probe])
        assert probe.target.__class__.__name__ in str(excinfo.value)


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
        ir.soss(ir.NetAddress(obj_map[a], ir.OutputPort.standard))
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
        ir.soss(ir.NetAddress(obj_map[b], ir.InputPort.standard))
    )


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
    assert new_conn.source == ir.NetAddress(ir_a, ir.OutputPort.standard)
    assert new_conn.sink == ir.NetAddress(new_obj, ir.InputPort.standard)
    assert new_conn.keyspace is None
    assert not new_conn.latching


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
            3, ir.NetAddress(ir_a, ir.OutputPort.standard),
            ir.NetAddress(ir_b, ir.InputPort.standard), None, False
        )

        net_ab2 = ir.IntermediateNet(
            3, ir.NetAddress(ir_a, ir.OutputPort.neurons),
            ir.NetAddress(ir_b, ir.InputPort.standard), None, False
        )

        conn_ba1 = mock.Mock(spec_set=[], name="B->A")
        net_ba1 = ir.IntermediateNet(
            3, ir.NetAddress(ir_b, ir.OutputPort.standard),
            ir.NetAddress(ir_a, ir.InputPort.standard), None, False
        )

        # Construct the intermediate representation
        irn = ir.IntermediateRepresentation(
            {a: ir_a, b: ir_b}, {conn_ab1: net_ab1, conn_ba1: net_ba1},
            [], [net_ab2]
        )

        # Retrieve the nets starting at a
        net_ax = irn.get_nets_starting_at(ir_a)
        assert net_ax[ir.OutputPort.standard] == {net_ab1: [conn_ab1]}
        assert net_ax[ir.OutputPort.neurons] == {net_ab2: []}

        # Retrieve the nets starting at b
        net_bx = irn.get_nets_starting_at(ir_b)
        assert net_bx[ir.OutputPort.standard] == {net_ba1: [conn_ba1]}
        assert net_bx[ir.OutputPort.neurons] == {}

        # Retrieve the nets ending at a
        net_xa = irn.get_nets_ending_at(ir_a)
        assert net_xa[ir.InputPort.standard] == {net_ba1: [conn_ba1]}

        # Retrieve the nets ending at b
        net_xb = irn.get_nets_ending_at(ir_b)
        assert net_xb[ir.InputPort.standard] == {net_ab1: [conn_ab1],
                                                 net_ab2: []}
