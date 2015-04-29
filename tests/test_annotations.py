import mock
import nengo
import pytest
from rig.bitfield import BitField

from nengo_spinnaker.annotations import Annotations, AnnotatedNet, soss
from nengo_spinnaker import annotations as ir
from nengo_spinnaker.keyspaces import KeyspaceContainer


class TestAnnotationsFromModel(object):
    """Test construction of annotations from a Nengo Model.
    """
    class TestBuildObject(object):
        @pytest.mark.parametrize("machine_timestep", [100, 1000])
        def test_build_object_from_registered_builder(self, machine_timestep):
            """Test that objects are correctly identified and built from the
            dictionary which can be registered against.
            """
            obj_builder = mock.Mock(name="builder")
            annotation = mock.Mock(name="annotation")
            obj_builder.return_value = annotation

            obj = mock.Mock(name="object", spec_set=nengo.base.NengoObject)
            built_obj = mock.Mock(name="built-object", spec_set=[])

            # Construct the fake model
            model = mock.Mock(name="model", spec_set=["params"])
            model.params = {obj: built_obj}

            # Call the annotation build method
            obj_builders = {nengo.base.NengoObject: obj_builder}
            with mock.patch.object(Annotations, "object_builders",
                                   obj_builders):
                o = Annotations.from_model(
                    model, machine_timestep=machine_timestep
                )

            # The builder should have been called once with the object and its
            # built equivalent.
            obj_builder.assert_called_once_with(obj, built_obj)
            assert o.objects[obj] is annotation

            # The timestep should have been saved
            assert o.machine_timestep == machine_timestep

        def test_build_object_from_extra_builder(self):
            """Test that objects are correctly identified and built from a
            supplied dictionary.
            """
            obj_builder = mock.Mock(name="builder")
            annotation = mock.Mock(name="annotation")
            obj_builder.return_value = annotation

            obj = mock.Mock(name="object", spec_set=nengo.base.NengoObject)
            built_obj = mock.Mock(name="built-object", spec_set=[])

            # Construct the fake model
            model = mock.Mock(name="model", spec_set=["params"])
            model.params = {obj: built_obj}

            # Call the annotation build method
            obj_builders = {nengo.base.NengoObject: obj_builder}
            with mock.patch.object(Annotations, "object_builders", {}):
                o = Annotations.from_model(model,
                                           extra_object_builders=obj_builders)

            # The builder should have been called once with the object and its
            # built equivalent.
            obj_builder.assert_called_once_with(obj, built_obj)
            assert o.objects[obj] is annotation

        def test_build_object_fails_with_no_builder(self):
            """Test that if there is no builder an error is raised.
            """
            obj = mock.Mock(name="object", spec_set=nengo.base.NengoObject)
            built_obj = mock.Mock(name="built-object", spec_set=[])

            # Construct the fake model
            model = mock.Mock(name="model", spec_set=["params"])
            model.params = {obj: built_obj}

            # Call the annotation build method
            with mock.patch.object(Annotations, "object_builders", {}), \
                    pytest.raises(TypeError) as excinfo:
                Annotations.from_model(model)

            assert obj.__class__.__name__ in str(excinfo.value)

    class TestGetAnnotationProbe(object):
        """Tests the calling of probe annotaters."""
        def test_from_registered_dicts(self):
            """Test that the probe builder is correctly called."""
            # Create the mock objects
            obj = mock.Mock(name="obj")  # Object the probe is probing
            probe = mock.Mock(name="probe", spec_set=nengo.Probe)
            probe.target = obj

            # Builder for the probe
            annotated_probe = mock.Mock(name="probe_annotation")
            extra_objs = [mock.Mock() for i in range(3)]
            extra_conns = [mock.Mock() for i in range(3)]
            builder = mock.Mock(name="builder")
            builder.return_value = (annotated_probe, extra_objs, extra_conns)

            # Create the annotations and ensure that the builder is called with
            # the correct parameters and that the annotations and extra objects
            # are retained.
            model = mock.Mock(spec_set=["params"])
            model.params = {probe: None}

            probe_builders = {type(obj): builder}
            with mock.patch.object(Annotations, "probe_builders",
                                   probe_builders):
                ann = Annotations.from_model(model)

            assert builder.call_count == 1
            args = builder.call_args[0]
            assert args[0] is probe
            assert isinstance(args[1], Annotations)

            assert ann.objects[probe] is annotated_probe
            assert ann.extra_objects == extra_objs
            assert ann.extra_connections == extra_conns

        def test_from_extra_dicts(self):
            """Test that the probe builder is correctly called."""
            # Create the mock objects
            obj = mock.Mock(name="obj")  # Object the probe is probing
            probe = mock.Mock(name="probe", spec_set=nengo.Probe)
            probe.target = obj

            # Builder for the probe
            annotated_probe = mock.Mock(name="probe_annotation")
            extra_objs = [mock.Mock() for i in range(3)]
            extra_conns = [mock.Mock() for i in range(3)]
            builder = mock.Mock(name="builder")
            builder.return_value = (annotated_probe, extra_objs, extra_conns)

            # Create the annotations and ensure that the builder is called with
            # the correct parameters and that the annotations and extra objects
            # are retained.
            model = mock.Mock(spec_set=["params"])
            model.params = {probe: None}

            probe_builders = {type(obj): builder}
            with mock.patch.object(Annotations, "probe_builders", {}):
                Annotations.from_model(
                    model, extra_probe_builders=probe_builders
                )

            assert builder.call_count == 1

        def test_no_builder(self):
            """Test that the probe builder is correctly called."""
            # Create the mock objects
            obj = mock.Mock(name="obj")  # Object the probe is probing
            probe = mock.Mock(name="probe", spec_set=nengo.Probe)
            probe.target = obj

            # Create the annotations and ensure that the builder is called with
            # the correct parameters and that the annotations and extra objects
            # are retained.
            model = mock.Mock(spec_set=["params"])
            model.params = {probe: None}

            with mock.patch.object(Annotations, "probe_builders", {}),\
                    pytest.raises(TypeError) as excinfo:
                Annotations.from_model(model)

            assert obj.__class__.__name__ in str(excinfo.value)

    class TestGetAnnotationNet(object):
        """Tests the construction of intermediate nets from connections.
        """
        class ObjTypeA(object):
            pass

        class ObjTypeB(object):
            pass

        def FauxConnection(self, pre, post, size_out=12):
            conn = mock.Mock(spec_set=nengo.Connection)
            conn.pre_obj = pre
            conn.post_obj = post
            conn.size_out = size_out
            return conn

        def test_standard(self):
            """The simple case where all we have to do is get the source and
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
                a.__class__: lambda x, y: soss(
                    x.pre_obj, extra_objects=a_extra_objs,
                    extra_nets=a_extra_conns
                )
            }
            sink_getters = {
                b.__class__: lambda x, y: soss(
                    x.post_obj, extra_objects=b_extra_objs,
                    extra_nets=b_extra_conns
                )
            }

            # Build the connection
            model = mock.Mock(spec_set=["params"])
            model.params = {c: None}

            with mock.patch.object(Annotations, "source_getters",
                                   source_getters),\
                    mock.patch.object(Annotations, "sink_getters",
                                      sink_getters):
                ann = Annotations.from_model(model)

            ic = ann.connections[c]
            assert ic.source is a
            assert b in ic.sinks
            assert ic.keyspace is None
            assert ic.weight == c.size_out
            assert not ic.latching  # Rx buffers should clear every timestep
            assert ann.extra_objects == a_extra_objs + b_extra_objs
            assert ann.extra_connections == a_extra_conns + b_extra_conns

        def test_pre_supplied_keyspace(self):
            """Test that when only the source provides a keyspace it is applied
            to the net.
            """
            a = self.ObjTypeA()
            b = self.ObjTypeB()
            c = self.FauxConnection(a, b)

            a_ks = mock.Mock(spec_set=[], name="ks_a")

            source_getters = {
                a.__class__: lambda x, y: soss(x.pre_obj, keyspace=a_ks)}
            sink_getters = {
                b.__class__: lambda x, y: soss(x.post_obj)}

            # Build the connection
            model = mock.Mock(spec_set=["params"])
            model.params = {c: None}

            with mock.patch.object(Annotations, "source_getters",
                                   source_getters),\
                    mock.patch.object(Annotations, "sink_getters",
                                      sink_getters):
                irn = Annotations.from_model(model)

            # Assert the keyspace made it
            assert irn.connections[c].keyspace is a_ks

        def test_post_supplied_keyspace(self):
            """Test that when only the sink provides a keyspace it is applied
            to the net.
            """
            a = self.ObjTypeA()
            b = self.ObjTypeB()
            c = self.FauxConnection(a, b)

            b_ks = mock.Mock(spec_set=[], name="ks_b")

            source_getters = {
                a.__class__: lambda x, y: soss(x.pre_obj)}
            sink_getters = {
                b.__class__: lambda x, y: soss(x.post_obj, keyspace=b_ks)}

            # Build the connection
            model = mock.Mock(spec_set=["params"])
            model.params = {c: None}

            with mock.patch.object(Annotations, "source_getters",
                                   source_getters),\
                    mock.patch.object(Annotations, "sink_getters",
                                      sink_getters):
                irn = Annotations.from_model(model)

            # Assert the keyspace made it
            assert irn.connections[c].keyspace is b_ks

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
                a.__class__: lambda x, y: soss(x.pre_obj, keyspace=a_ks)}
            sink_getters = {
                b.__class__: lambda x, y: soss(x.post_obj, keyspace=b_ks)}

            # Build the connection
            model = mock.Mock(spec_set=["params"])
            model.params = {c: None}

            with mock.patch.object(Annotations, "source_getters",
                                   source_getters),\
                    mock.patch.object(Annotations, "sink_getters",
                                      sink_getters),\
                    pytest.raises(NotImplementedError) as excinfo:
                Annotations.from_model(model)

            assert "keyspace" in str(excinfo.value)

        @pytest.mark.parametrize(
            "source_getters, sink_getters",
            [({ObjTypeA: lambda x, y: soss(None)},
              {ObjTypeB: lambda x, y: soss(x.post_obj)}),
             ({ObjTypeA: lambda x, y: soss(x.pre_obj)},
              {ObjTypeB: lambda x, y: soss(None)}),
             ({ObjTypeA: lambda x, y: None},
              {ObjTypeB: lambda x, y: soss(x.post_obj)}),
             ({ObjTypeA: lambda x, y: soss(x.pre_obj)},
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

            # Build the connection
            model = mock.Mock(spec_set=["params"])
            model.params = {c: None}

            with mock.patch.object(Annotations, "source_getters",
                                   source_getters),\
                    mock.patch.object(Annotations, "sink_getters",
                                      sink_getters):
                irn = Annotations.from_model(model)

            assert irn.connections[c] is None

        @pytest.mark.parametrize(
            "source_getters, sink_getters",
            [({ObjTypeA: lambda x, y: soss(x.pre_obj, latching=True)},
              {ObjTypeB: lambda x, y: soss(x.post_obj)}),
             ({ObjTypeA: lambda x, y: soss(x.pre_obj)},
              {ObjTypeB: lambda x, y: soss(x.post_obj, latching=True)}),
             ]
        )
        def test_requires_latching_net(self, source_getters, sink_getters):
            """Test that the net is marked as latching if the source or sink
            requires it.
            """
            a = self.ObjTypeA()
            b = self.ObjTypeB()
            c = self.FauxConnection(a, b)

            # Build the connection
            model = mock.Mock(spec_set=["params"])
            model.params = {c: None}

            with mock.patch.object(Annotations, "source_getters",
                                   source_getters),\
                    mock.patch.object(Annotations, "sink_getters",
                                      sink_getters):
                irn = Annotations.from_model(model)

            assert irn.connections[c].latching

        @pytest.mark.parametrize(
            "source_getters, sink_getters, expected_weight",
            [({ObjTypeA: lambda x, y: soss(x.pre_obj, weight=5)},
              {ObjTypeB: lambda x, y: soss(x.post_obj)}, 5),
             ({ObjTypeA: lambda x, y: soss(x.pre_obj, weight=3)},
              {ObjTypeB: lambda x, y: soss(x.post_obj, weight=9)}, 9),
             ]
        )
        def test_get_weight(self, source_getters, sink_getters,
                            expected_weight):
            """Test that the weight is correctly returned.
            """
            a = self.ObjTypeA()
            b = self.ObjTypeB()
            c = self.FauxConnection(a, b, 7)

            # Build the connection
            model = mock.Mock(spec_set=["params"])
            model.params = {c: None}

            with mock.patch.object(Annotations, "source_getters",
                                   source_getters),\
                    mock.patch.object(Annotations, "sink_getters",
                                      sink_getters):
                irn = Annotations.from_model(model)

            assert irn.connections[c].weight == expected_weight

        @pytest.mark.parametrize(
            "source_getters, sink_getters, reason, fail_type",
            [({ObjTypeA: lambda x, y: soss(x.pre_obj)}, {}, "sink", ObjTypeB),
             ({}, {ObjTypeB: lambda x, y: soss(x.post_obj)},
              "source", ObjTypeA)]
        )
        def test_missing_builder(self, source_getters, sink_getters, reason,
                                 fail_type):
            a = self.ObjTypeA()
            b = self.ObjTypeB()
            c = self.FauxConnection(a, b)

            # Build the connection
            model = mock.Mock(spec_set=["params"])
            model.params = {c: None}

            with mock.patch.object(Annotations, "source_getters",
                                   source_getters),\
                    mock.patch.object(Annotations, "sink_getters",
                                      sink_getters),\
                    pytest.raises(TypeError) as excinfo:
                Annotations.from_model(model)

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
                a.__class__: lambda x, y: soss(
                    x.pre_obj, extra_objects=a_extra_objs,
                    extra_nets=a_extra_conns
                )
            }
            sink_getters = {
                b.__class__: lambda x, y: soss(
                    x.post_obj, extra_objects=b_extra_objs,
                    extra_nets=b_extra_conns
                )
            }

            # Build the connection
            model = mock.Mock(spec_set=["params"])
            model.params = {c: None}

            with mock.patch.object(Annotations, "source_getters", {}),\
                    mock.patch.object(Annotations, "sink_getters", {}):
                irn = Annotations.from_model(
                    model, extra_source_getters=source_getters,
                    extra_sink_getters=sink_getters
                )

            assert c in irn.connections

        def test_use_existing_net(self):
            """Test that if the source getter returns a Net instead of a soss
            then we use the existing net and don't ask for a new sink.
            """
            a = self.ObjTypeA()
            b = self.ObjTypeB()
            c = self.FauxConnection(a, b)

            d = AnnotatedNet(None, [None])  # Existing net

            source_getters = {a.__class__: lambda x, y: d}
            sink_getters = {b.__class__: lambda x, y: soss(x.post_obj)}

            # Build the connection
            model = mock.Mock(spec_set=["params"])
            model.params = {c: None}

            with mock.patch.object(Annotations, "source_getters",
                                   source_getters),\
                    mock.patch.object(Annotations, "sink_getters",
                                      sink_getters):
                irn = Annotations.from_model(model)

            # Assert the existing net was used and modified
            assert irn.connections[c] is d
            assert d.sinks == [None, b]


def test_get_source_standard():
    """Test that get_source_standard just does a look up in the object map
    dictionary and uses OutputPort.standard.
    """
    with nengo.Network():
        a = nengo.Ensemble(300, 4)
        b = nengo.Ensemble(300, 2)
        c = nengo.Connection(a[:2], b)

    objects = {
        a: mock.Mock(name="ir_a", spec_set=[]),
        b: mock.Mock(name="ir_b", spec_set=[]),
    }

    irn = Annotations(objects, {}, [], [])
    assert (
        ir.get_source_standard(c, irn) ==
        ir.soss(ir.NetAddress(objects[a], ir.OutputPort.standard))
    )


def test_get_sink_standard():
    """Test that get_sink_standard just does a look up in the object map
    dictionary and uses InputPort.standard.
    """
    with nengo.Network():
        a = nengo.Ensemble(300, 4)
        b = nengo.Ensemble(300, 2)
        c = nengo.Connection(a[:2], b)

    objects = {
        a: mock.Mock(name="ir_a", spec_set=[]),
        b: mock.Mock(name="ir_b", spec_set=[]),
    }

    irn = Annotations(objects, {}, [], [])
    assert (
        ir.get_sink_standard(c, irn) ==
        ir.soss(ir.NetAddress(objects[b], ir.InputPort.standard))
    )


def test_get_probe_standard():
    """Test that the default probe type is sane."""
    probe = mock.Mock(name="probe", spec_set=[])
    ann = mock.Mock(name="annotations", spec_set=[])

    (ann_p, extra_objs, extra_conns) = ir.get_probe(probe, ann)
    assert isinstance(ann_p, ir.ObjectAnnotation)
    assert extra_objs == extra_conns == list()


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
        ir_a = ir.ObjectAnnotation(a)
        ir_b = ir.ObjectAnnotation(b)

        # Create some nets, some with and some without matching connections
        conn_ab1 = mock.Mock(spec_set=[], name="A->B")
        net_ab1 = AnnotatedNet(
            ir.NetAddress(ir_a, ir.OutputPort.standard),
            ir.NetAddress(ir_b, ir.InputPort.standard), None, False
        )

        net_ab2 = AnnotatedNet(
            ir.NetAddress(ir_a, ir.OutputPort.neurons),
            ir.NetAddress(ir_b, ir.InputPort.standard), None, False
        )

        conn_ba1 = mock.Mock(spec_set=[], name="B->A")
        net_ba1 = AnnotatedNet(
            ir.NetAddress(ir_b, ir.OutputPort.standard),
            ir.NetAddress(ir_a, ir.InputPort.standard), None, False
        )

        # Construct the intermediate representation
        irn = Annotations(
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

    def test_apply_default_keyspace(self):
        """Test applying the default keyspace to nets without a defined
        keyspace.
        """
        # Get the default keyspace we want to use
        ksc = KeyspaceContainer()
        default_keyspace = ksc["nengo"]

        # Now create an intermediate Net with a mixture of objects with and
        # without keyspaces.
        mock_ks1 = mock.Mock(name="keyspace")

        class Obj(object):
            pass

        # Create objects and their intermediate representations
        a = Obj()
        b = Obj()
        c = Obj()
        ir_a = ir.ObjectAnnotation(a)
        ir_b = ir.ObjectAnnotation(b)
        ir_c = ir.ObjectAnnotation(c)

        # Create some nets, some with and some without matching connections
        conn_ab1 = mock.Mock(spec_set=[], name="A->B")
        net_ab1 = AnnotatedNet(
            ir.NetAddress(ir_a, ir.OutputPort.standard),
            ir.NetAddress(ir_b, ir.InputPort.standard), mock_ks1, False
        )

        net_ab2 = AnnotatedNet(
            ir.NetAddress(ir_a, ir.OutputPort.neurons),
            ir.NetAddress(ir_b, ir.InputPort.standard), None, False
        )

        net_ab3 = AnnotatedNet(
            ir.NetAddress(ir_a, ir.OutputPort.standard),
            ir.NetAddress(ir_b, ir.InputPort.standard), None, False
        )

        conn_ba1 = mock.Mock(spec_set=[], name="B->A")
        net_ba1 = AnnotatedNet(
            ir.NetAddress(ir_b, ir.OutputPort.standard),
            ir.NetAddress(ir_a, ir.InputPort.standard), None, False
        )

        net_cb1 = AnnotatedNet(
            ir.NetAddress(ir_c, ir.OutputPort.standard),
            ir.NetAddress(ir_b, ir.InputPort.standard), mock_ks1, False
        )

        # Construct the intermediate representation
        irn = Annotations(
            {a: ir_a, b: ir_b}, {conn_ab1: net_ab1, conn_ba1: net_ba1},
            [ir_c], [net_ab2, net_ab3, net_cb1]
        )

        # Apply the default keyspace
        irn.apply_default_keyspace(default_keyspace)

        # Check that keyspaces have been applied
        assert net_ab1.keyspace is mock_ks1
        assert net_cb1.keyspace is mock_ks1
        assert isinstance(net_ab2.keyspace, BitField)
        assert isinstance(net_ab3.keyspace, BitField)
        assert isinstance(net_ba1.keyspace, BitField)
        assert net_ab2.keyspace.object == net_ab3.keyspace.object
        assert (net_ab2.keyspace.connection !=
                net_ab3.keyspace.connection)

        assert net_ba1.keyspace.object != net_ab2.keyspace.object
        assert net_ba1.keyspace.connection == 0
