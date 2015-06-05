import mock
from mock import patch
import nengo
from nengo.cache import NoDecoderCache
import numpy as np
import pytest
from six import iteritems

from nengo_spinnaker.builder.builder import (
    Model, Signal, spec, _make_signal, ObjectPort, OutputPort, InputPort,
    netlistspec
)
from nengo_spinnaker.netlist import Vertex, VertexSlice


class TestSignal(object):
    @pytest.mark.parametrize("latching, weight", [(True, 5), (False, 1)])
    def test_single_sink(self, latching, weight):
        """Test that creating a signal with a single sink still expands to a
        list.
        """
        source = mock.Mock()
        sink = mock.Mock()
        keyspace = mock.Mock()

        # Create the signal
        signal = Signal(source, sink, keyspace, weight, latching)

        # Ensure the parameters are sane
        assert signal.source is source
        assert signal.sinks == [sink]
        assert signal.keyspace is keyspace
        assert signal.weight == weight
        assert signal.latching is latching

    def test_multiple_sink(self):
        """Test that creating a signal with a multiple sinks works."""
        source = mock.Mock()
        sinks = [mock.Mock() for _ in range(3)]
        keyspace = mock.Mock()

        # Create the signal
        signal = Signal(source, sinks, keyspace)

        # Ensure the parameters are sane
        assert signal.source is source
        assert signal.sinks == sinks
        assert signal.keyspace is keyspace
        assert signal.weight == 0
        assert not signal.latching


def test_model_init():
    """Test initialising a model, should be completely empty."""
    model = Model()
    assert model.dt == 0.001
    assert model.machine_timestep == 1000

    assert model.params == dict()
    assert model.seeds == dict()

    assert dict(model.object_operators) == dict()
    assert dict(model.connections_signals) == dict()
    assert model.extra_operators == list()
    assert model.extra_signals == list()

    assert isinstance(model.decoder_cache, NoDecoderCache)
    assert len(model.keyspaces) == 1


def test_model_init_with_keyspaces():
    """Test initialising a model, should be completely empty."""
    keyspaces = mock.Mock()
    model = Model(keyspaces=keyspaces)

    assert model.keyspaces is keyspaces


class TestBuild(object):
    @pytest.mark.parametrize("use_make_object", [False, True])
    def test_builder_dictionaries_are_combined(self, use_make_object):
        """Test that the builder and extra_builders dictionaries are combined
        and that a mrolookupdict is used.
        """
        class A(object):
            seed = 101

        class B(object):
            pass

        builders = {A: mock.Mock()}
        extra_builders = {B: mock.Mock()}

        a = A()
        b = B()

        network = mock.Mock()
        network.seed = None
        network.connections = []
        network.ensembles = [a]
        network.nodes = [b]
        network.networks = []
        network.probes = []
        network.config = mock.Mock(name="config")

        if not use_make_object:
            # Patch the default builders
            with patch.object(Model, "builders", new=builders):
                # Create a model and build the mock network
                model = Model()
                model.build(network, extra_builders=extra_builders)

            # Assert that the config was stored in the model
            assert model.config is network.config
        else:
            # Create the model
            model = Model()
            model.rng = np.random

            # When using `make_object` directly `_builders` should be defined
            # and used.
            model._builders.update(builders)
            model._builders.update(extra_builders)

            # Build the objects
            model.make_object(a)
            model.make_object(b)

        # Assert that seeds were supplied
        assert model.seeds[a] == a.seed
        assert model.seeds[b] is not None

        # Assert the builders got called
        builders[A].assert_called_once_with(model, a)
        extra_builders[B].assert_called_once_with(model, b)

    def test_builds_hierarchy(self):
        """Test that networks are built hierarchically.
        """
        class A(object):
            seed = 101

        class B(object):
            pass

        builders = {A: mock.Mock()}
        extra_builders = {B: mock.Mock()}

        a = A()
        b = B()

        network_a = mock.Mock()
        network_a.seed = None
        network_a.connections = []
        network_a.ensembles = [a]
        network_a.nodes = []
        network_a.networks = []
        network_a.probes = []
        network_a.config = mock.Mock(name="config")

        network_b = mock.Mock()
        network_b.seed = None
        network_b.connections = []
        network_b.ensembles = []
        network_b.nodes = [b]
        network_b.networks = []
        network_b.probes = []
        network_b.config = mock.Mock(name="config")

        network = mock.Mock()
        network.seed = None
        network.connections = []
        network.ensembles = []
        network.nodes = []
        network.networks = [network_a, network_b]
        network.probes = []
        network.config = mock.Mock(name="config")

        # Patch the default builders
        with patch.object(Model, "builders", new=builders):
            # Create a model and build the mock network
            model = Model()
            model.build(network, extra_builders=extra_builders)

        # Assert that the config was stored in the model
        assert model.config is network.config

        # Assert that seeds were supplied
        assert model.seeds[a] == a.seed
        assert model.seeds[b] is not None

        # Assert the builders got called
        builders[A].assert_called_once_with(model, a)
        extra_builders[B].assert_called_once_with(model, b)


class TestMakeConnection(object):
    @pytest.mark.parametrize("use_registered_dicts", [True, False])
    @pytest.mark.parametrize("seed", [None, 456])
    def test_make_connections(self, use_registered_dicts, seed):
        """Test that building connections adds a new signal to the model."""
        # TODO Test that the connection is fully built
        model = Model()

        class A(object):
            pass

        class B(object):
            pass

        a = A()
        b = B()

        # Create a connection from a to b
        connection = mock.Mock()
        connection.seed = seed
        connection.pre_obj = a
        connection.post_obj = b

        # Create getter methods
        source = ObjectPort(mock.Mock(), None)
        sink = ObjectPort(mock.Mock(), None)

        def source_getter_fn(m, c):
            assert m is model
            assert c is connection

            return spec(source)

        source_getter = mock.Mock(wraps=source_getter_fn)

        def sink_getter_fn(m, c):
            assert m is model
            assert c is connection

            return spec(sink)

        sink_getter = mock.Mock(wraps=sink_getter_fn)

        # Create a method to build the connection
        built_connection = mock.Mock(name="built connection")

        def connection_builder_fn(m, c):
            assert m is model
            assert c is connection

            return built_connection

        connection_builder_a = mock.Mock(wraps=connection_builder_fn)
        connection_builder_b = mock.Mock(wraps=connection_builder_fn)

        # Create a mock network
        network = mock.Mock()
        network.seed = None
        network.connections = [connection]
        network.ensembles = []
        network.nodes = []
        network.networks = []
        network.probes = []

        if use_registered_dicts:
            # Patch the getters, add a null builder
            with patch.object(model, "source_getters", {A: source_getter}), \
                    patch.object(model, "sink_getters", {B: sink_getter}), \
                    patch.object(model, "connection_parameter_builders",
                                 {A: connection_builder_a,
                                  B: connection_builder_b}):
                # Build the network
                model.build(network)
        else:
            model.build(network,
                        extra_source_getters={A: source_getter},
                        extra_sink_getters={B: sink_getter},
                        extra_connection_parameter_builders={
                            A: connection_builder_a,
                            B: connection_builder_b,
                        })

        # Check that seeds were provided
        if seed is not None:
            assert model.seeds[connection] == seed
        else:
            assert model.seeds[connection] is not None

        # Assert the getters were called
        assert source_getter.call_count == 1
        assert sink_getter.call_count == 1

        # Assert that the connection parameter builder was called
        assert connection_builder_a.call_count == 1
        assert connection_builder_b.call_count == 0

        # Assert that the parameters were saved
        assert model.params[connection] is built_connection

        # Assert that the signal exists
        signal = model.connections_signals[connection]
        assert signal.source is source
        assert signal.sinks == [sink]

    @pytest.mark.parametrize(
        "source_getter, sink_getter",
        [(lambda m, c: None, lambda m, c: spec(None)),
         (lambda m, c: spec(None), lambda m, c: None),
         ]
    )
    def test_make_connection_no_signal(self, source_getter, sink_getter):
        """Test that building connections adds a new signal to the model."""
        model = Model()

        class A(object):
            pass

        # Create a connection from a to b
        connection = mock.Mock()
        connection.pre_obj = A()
        connection.post_obj = A()

        # Create a mock network
        network = mock.Mock()
        network.seed = None
        network.connections = [connection]
        network.ensembles = []
        network.nodes = []
        network.networks = []
        network.probes = []

        # Patch the getters, add a null builder
        with patch.object(model, "source_getters", {A: source_getter}), \
                patch.object(model, "sink_getters", {A: sink_getter}), \
                patch.object(model, "connection_parameter_builders",
                             {A: mock.Mock()}):
            # Build the network
            model.build(network)

        # Assert that no signal exists
        assert connection not in model.connections_signals


class TestBuildProbe(object):
    """Test the building of probes."""
    @pytest.mark.parametrize("use_arguments", [False, True])
    @pytest.mark.parametrize("with_slice", [False, True])
    def test_standard(self, use_arguments, with_slice):
        # Create test network
        with nengo.Network() as network:
            a = nengo.Ensemble(100, 2)

            if not with_slice:
                p_a = nengo.Probe(a)
            else:
                p_a = nengo.Probe(a[0])

            p_n = nengo.Probe(a.neurons)

        # Create a model
        model = Model()

        # Dummy neurons builder
        ens_build = mock.Mock(name="ensemble builder")

        # Define two different probe build functions
        def build_ens_probe_fn(model, probe):
            assert ens_build.called
            assert model is model
            assert probe is p_a

        build_ens_probe = mock.Mock(wraps=build_ens_probe_fn)

        def build_neurons_probe_fn(model, probe):
            assert ens_build.called
            assert model is model
            assert probe is p_n

        build_neurons_probe = mock.Mock(wraps=build_neurons_probe_fn)

        # Build the model
        probe_builders = {nengo.Ensemble: build_ens_probe,
                          nengo.ensemble.Neurons: build_neurons_probe}
        with patch.object(model, "builders", new={nengo.Ensemble: ens_build}):
            if not use_arguments:
                with patch.object(model, "probe_builders", new=probe_builders):
                    model.build(network)
            else:
                with patch.object(model, "probe_builders", new={}):
                    model.build(network, extra_probe_builders=probe_builders)

        # Assert the probe functions were built
        assert p_a in model.seeds
        assert p_n in model.seeds
        assert build_ens_probe.call_count == 1
        assert build_neurons_probe.call_count == 1

    def test_probe_building_disabled(self, recwarn):
        """Test that build methods are not called and that a warning is raised
        if probe building is disabled.
        """
        # Create test network
        with nengo.Network() as network:
            a = nengo.Ensemble(100, 2)
            p_a = nengo.Probe(a, label="Output")

        # Create a model
        model = Model()

        # Dummy neurons builder
        ens_build = mock.Mock(name="ensemble builder")

        # Define a probe build function
        build_ens_probe = mock.Mock()

        # Build the model
        probe_builders = {nengo.Ensemble: build_ens_probe}
        with patch.object(model, "builders", new={nengo.Ensemble: ens_build}),\
                patch.object(model, "probe_builders", new=probe_builders):
            model.build(network, build_probes=False)

        # Assert the probes were NOT built
        assert p_a not in model.seeds
        assert build_ens_probe.call_count == 0

        # And that a warning was raised
        w = recwarn.pop()
        assert "Probes" in str(w.message)
        assert "disabled" in str(w.message)


def test_get_object_and_connection_id():
    """Test retrieving an object and a connection ID."""
    obj_a = mock.Mock(name="object a")
    conn_a0 = mock.Mock(name="connection a[0]")
    conn_a1 = mock.Mock(name="connection a[0]")

    obj_b = mock.Mock(name="object b")
    conn_b0 = mock.Mock(name="connection b[0]")

    # Create an empty model
    model = Model()

    # The first connection from the first object should get (0, 0), no matter
    # how many times we ask
    assert (0, 0) == model._get_object_and_connection_id(obj_a, conn_a0)

    # The second connection from the first object should get (0, 1)
    assert (0, 1) == model._get_object_and_connection_id(obj_a, conn_a1)
    assert (0, 0) == model._get_object_and_connection_id(obj_a, conn_a0)

    # The first connection from the second object should get (1, 0)
    assert (1, 0) == model._get_object_and_connection_id(obj_b, conn_b0)
    assert (0, 1) == model._get_object_and_connection_id(obj_a, conn_a1)
    assert (0, 0) == model._get_object_and_connection_id(obj_a, conn_a0)


def test_spec():
    """Test specifying the source or sink of a signal."""
    # With minimal arguments
    s = spec(None)
    assert s.target is None
    assert s.keyspace is None
    assert not s.latching
    assert s.weight == 0

    # With all arguments
    target = mock.Mock(name="target")
    keyspace = mock.Mock(name="keyspace")
    weight = 5
    latching = True

    s = spec(target, keyspace=keyspace, weight=weight, latching=latching)
    assert s.target is target
    assert s.keyspace is keyspace
    assert s.weight == weight
    assert s.latching is latching


class TestMakeSignalFromSpecs(object):
    """Test constructing signals from spec objects."""
    def test_standard(self):
        """Test that mostly empty specs result in appropriate calls to build
        the keyspace and that the weight is grabbed from the connection.
        """
        # Create the model with it's default keyspace
        model = mock.Mock()
        model.keyspaces = {"nengo": mock.Mock()}
        exp_ks = model.keyspaces["nengo"].return_value = mock.Mock()
        model._get_object_and_connection_id.return_value = (1, 3)

        # Create the connection that we're building
        pre = mock.Mock("pre")
        post = mock.Mock("post")
        post.size_in = 5
        connection = mock.Mock(spec_set=nengo.Connection)
        connection.pre_obj = pre
        connection.post_obj = post

        # Create a spec for the source and a spec for the sink
        source_obj = mock.Mock()
        source_spec = spec(source_obj)
        sink_obj = mock.Mock()
        sink_spec = spec(sink_obj)

        # Get the Signal
        signal = _make_signal(model, connection, source_spec, sink_spec)
        assert signal.source is source_obj
        assert signal.sinks == [sink_obj]
        assert signal.keyspace is exp_ks
        assert signal.weight == post.size_in
        assert signal.latching is False

        # Check that the keyspace was called correctly
        model.keyspaces["nengo"].assert_called_once_with(object=1,
                                                         connection=3)

    @pytest.mark.parametrize(
        "make_source_spec, make_sink_spec",
        [(lambda obj: spec(obj, latching=True),
          lambda obj: spec(obj, latching=False)),
         (lambda obj: spec(obj, latching=False),
          lambda obj: spec(obj, latching=True)),
         (lambda obj: spec(obj, latching=True),
          lambda obj: spec(obj, latching=True)),
         ]
    )
    def test_latching(self, make_source_spec, make_sink_spec):
        """Test that latching commands are taken from the spec.
        """
        # Create the model with it's default keyspace
        model = Model()
        model.keyspaces = {"nengo": mock.Mock()}

        # Create the connection that we're building
        pre = mock.Mock("pre")
        post = mock.Mock("post")
        post.size_in = 5
        connection = mock.Mock(spec_set=nengo.Connection)
        connection.pre_obj = pre
        connection.post_obj = post

        # Create a spec for the source and a spec for the sink
        source_obj = mock.Mock()
        source_spec = make_source_spec(source_obj)
        sink_obj = mock.Mock()
        sink_spec = make_sink_spec(sink_obj)

        # Get the Signal
        signal = _make_signal(model, connection, source_spec, sink_spec)
        assert signal.latching is True

    @pytest.mark.parametrize(
        "source_weight, sink_weight, expected_weight",
        [(0, 0, 5),
         (10, 0, 10),
         (0, 10, 10),
         ]
    )
    def test_weights(self, source_weight, sink_weight, expected_weight):
        """Test that weights are taken from the spec.
        """
        # Create the model with it's default keyspace
        model = Model()

        # Create the connection that we're building
        pre = mock.Mock("pre")
        post = mock.Mock("post")
        post.size_in = 5
        connection = mock.Mock(spec_set=nengo.Connection)
        connection.pre_obj = pre
        connection.post_obj = post

        # Create a spec for the source and a spec for the sink
        source_obj = mock.Mock()
        source_spec = spec(source_obj, weight=source_weight)
        sink_obj = mock.Mock()
        sink_spec = spec(sink_obj, weight=sink_weight)

        # Get the Signal
        signal = _make_signal(model, connection, source_spec, sink_spec)
        assert signal.weight == expected_weight

    def test_keyspace_from_source(self):
        # Create the model with it's default keyspace
        model = Model()

        # Create the keyspace
        keyspace = mock.Mock(name="keyspace")

        # Create the connection that we're building
        pre = mock.Mock("pre")
        post = mock.Mock("post")
        post.size_in = 0
        connection = mock.Mock(spec_set=nengo.Connection)
        connection.pre_obj = pre
        connection.post_obj = post

        # Create a spec for the source and a spec for the sink
        source_obj = mock.Mock()
        source_spec = spec(source_obj, keyspace=keyspace)
        sink_obj = mock.Mock()
        sink_spec = spec(sink_obj)

        # Get the Signal
        signal = _make_signal(model, connection, source_spec, sink_spec)
        assert signal.keyspace is keyspace

    def test_keyspace_from_sink(self):
        # Create the model with it's default keyspace
        model = Model()

        # Create the keyspace
        keyspace = mock.Mock(name="keyspace")

        # Create the connection that we're building
        pre = mock.Mock("pre")
        post = mock.Mock("post")
        post.size_in = 0
        connection = mock.Mock(spec_set=nengo.Connection)
        connection.pre_obj = pre
        connection.post_obj = post

        # Create a spec for the source and a spec for the sink
        source_obj = mock.Mock()
        source_spec = spec(source_obj)
        sink_obj = mock.Mock()
        sink_spec = spec(sink_obj, keyspace=keyspace)

        # Get the Signal
        signal = _make_signal(model, connection, source_spec, sink_spec)
        assert signal.keyspace is keyspace

    def test_keyspace_collision(self):
        # Create the model with it's default keyspace
        model = Model()

        # Create the keyspace
        keyspace_a = mock.Mock(name="keyspace")
        keyspace_b = mock.Mock(name="keyspace")

        # Create the connection that we're building
        pre = mock.Mock("pre")
        post = mock.Mock("post")
        post.size_in = 0
        connection = mock.Mock(spec_set=nengo.Connection)
        connection.pre_obj = pre
        connection.post_obj = post

        # Create a spec for the source and a spec for the sink
        source_obj = mock.Mock()
        source_spec = spec(source_obj, keyspace=keyspace_a)
        sink_obj = mock.Mock()
        sink_spec = spec(sink_obj, keyspace=keyspace_b)

        with pytest.raises(NotImplementedError) as excinfo:
            _make_signal(model, connection, source_spec, sink_spec)
        assert "keyspace" in str(excinfo.value)


class TestGetSignalsAndConnections(object):
    """Test getting the signals and connections which either originate or
    terminate at a given object.
    """
    def test_get_signals_and_connections_starting_from(self):
        """Test getting the signals and connections which start from a given
        object.
        """
        # Create some objects and some connections
        obj_a = mock.Mock(name="object a")
        obj_b = mock.Mock(name="object b")

        conn_ab1 = mock.Mock()
        sig_ab1 = Signal(ObjectPort(obj_a, OutputPort.standard),
                         ObjectPort(obj_b, InputPort.standard),
                         None)
        conn_ab2 = mock.Mock()
        sig_ab2 = Signal(ObjectPort(obj_a, OutputPort.standard),
                         ObjectPort(obj_b, InputPort.standard),
                         None)

        sig_ab3 = Signal(ObjectPort(obj_a, OutputPort.standard),
                         ObjectPort(obj_b, InputPort.standard),
                         None)
        sig_ab4 = Signal(ObjectPort(obj_a, OutputPort.standard),
                         ObjectPort(obj_b, InputPort.standard),
                         None)

        conn_ba1 = mock.Mock()
        port_b1 = mock.Mock(name="port B1")
        sig_ba1 = Signal(ObjectPort(obj_b, port_b1),
                         ObjectPort(obj_a, InputPort.standard),
                         None)
        conn_ba2 = mock.Mock()
        conn_ba3 = mock.Mock()
        port_b2 = mock.Mock(name="port B2")
        sig_ba2 = Signal(ObjectPort(obj_b, port_b2),
                         ObjectPort(obj_a, InputPort.standard),
                         None)

        port_b3 = mock.Mock(name="port B3")
        sig_ba3 = Signal(ObjectPort(obj_b, port_b3),
                         ObjectPort(obj_a, InputPort.standard),
                         None)

        # Create a model holding all of these items
        model = Model()
        model.connections_signals = {
            conn_ab1: sig_ab1,
            conn_ab2: sig_ab2,
            conn_ba1: sig_ba1,
            conn_ba2: sig_ba2,
            conn_ba3: sig_ba2,
        }
        model.extra_signals = [sig_ab3, sig_ab4, sig_ba3]

        # Query it for connections starting from different objects
        assert model.get_signals_connections_from_object(obj_a) == {
            OutputPort.standard: {
                sig_ab1: [conn_ab1],
                sig_ab2: [conn_ab2],
                sig_ab3: [],
                sig_ab4: [],
            },
        }

        for port, sigs_conns in iteritems(
                model.get_signals_connections_from_object(obj_b)):
            if port is port_b1:
                assert sigs_conns == {
                    sig_ba1: [conn_ba1],
                }
            elif port is port_b2:
                for sig, conns in iteritems(sigs_conns):
                    assert sig is sig_ba2
                    for conn in conns:
                        assert conn is conn_ba2 or conn is conn_ba3
            elif port is port_b3:
                assert sigs_conns == {sig_ba3: []}
            else:
                assert False, "Unexpected signal"

    def test_get_signals_and_connections_terminating_at(self):
        """Test getting the signals and connections which end at a given
        object.
        """
        # Create some objects and some connections
        obj_a = mock.Mock(name="object a")
        obj_b = mock.Mock(name="object b")

        conn_ab1 = mock.Mock()
        port_b1 = mock.Mock(name="port B1")
        sig_ab1 = Signal(ObjectPort(obj_a, OutputPort.standard),
                         ObjectPort(obj_b, port_b1),
                         None)
        conn_ab2 = mock.Mock()
        port_b2 = mock.Mock(name="port B2")
        sig_ab2 = Signal(ObjectPort(obj_a, OutputPort.standard),
                         ObjectPort(obj_b, port_b2),
                         None)

        sig_ab3 = Signal(ObjectPort(obj_a, OutputPort.standard),
                         ObjectPort(obj_b, port_b2),
                         None)

        conn_ba1 = mock.Mock()
        sig_ba1 = Signal(ObjectPort(obj_b, OutputPort.standard),
                         ObjectPort(obj_a, InputPort.standard),
                         None)
        conn_ba2 = mock.Mock()
        conn_ba3 = mock.Mock()
        sig_ba2 = Signal(ObjectPort(obj_b, port_b2),
                         ObjectPort(obj_a, InputPort.standard),
                         None)

        # Create a model holding all of these items
        model = Model()
        model.connections_signals = {
            conn_ab1: sig_ab1,
            conn_ab2: sig_ab2,
            conn_ba1: sig_ba1,
            conn_ba2: sig_ba2,
            conn_ba3: sig_ba2,
        }
        model.extra_signals = [sig_ab3]

        # Query it for connections terminating at different objects
        for port, sigs_conns in iteritems(
                model.get_signals_connections_to_object(obj_a)):
            assert port is InputPort.standard

            for sig, conns in iteritems(sigs_conns):
                if sig is sig_ba1:
                    assert conns == [conn_ba1]
                elif sig is sig_ba2:
                    for conn in conns:
                        assert conn in [conn_ba2, conn_ba3]
                elif sig is sig_ba3:
                    assert len(conns) == 0
                else:
                    assert False, "Unexpected signal"

        assert model.get_signals_connections_to_object(obj_b) == {
            port_b1: {
                sig_ab1: [conn_ab1],
            },
            port_b2: {
                sig_ab2: [conn_ab2],
                sig_ab3: [],
            },
        }


class TestMakeNetlist(object):
    """Test production of netlists from operators and signals."""
    def test_single_vertices(self):
        """Test that operators which produce single vertices work correctly and
        that all functions and signals are correctly collected and included in
        the final netlist.
        """
        # Create the first operator
        vertex_a = mock.Mock(name="vertex A")
        load_fn_a = mock.Mock(name="load function A")
        pre_fn_a = mock.Mock(name="pre function A")
        post_fn_a = mock.Mock(name="post function A")

        object_a = mock.Mock(name="object A")
        operator_a = mock.Mock(name="operator A")
        operator_a.make_vertices.return_value = \
            netlistspec(vertex_a, load_fn_a, pre_fn_a, post_fn_a)

        # Create the second operator
        vertex_b = mock.Mock(name="vertex B")
        load_fn_b = mock.Mock(name="load function B")

        object_b = mock.Mock(name="object B")
        operator_b = mock.Mock(name="operator B")
        operator_b.make_vertices.return_value = \
            netlistspec(vertex_b, load_fn_b)

        # Create a signal between the operators
        keyspace = mock.Mock(name="keyspace")
        keyspace.length = 32
        signal_ab = Signal(ObjectPort(operator_a, None),
                           ObjectPort(operator_b, None),
                           keyspace=keyspace, weight=43)

        # Create the model, add the items and then generate the netlist
        model = Model()
        model.object_operators[object_a] = operator_a
        model.object_operators[object_b] = operator_b
        model.connections_signals[None] = signal_ab
        netlist = model.make_netlist()

        # Check that the make_vertices functions were called
        operator_a.make_vertices.assert_called_once_with(model)
        operator_b.make_vertices.assert_called_once_with(model)

        # Check that the netlist is as expected
        assert len(netlist.nets) == 1
        for net in netlist.nets:
            assert net.source is vertex_a
            assert net.sinks == [vertex_b]
            assert net.keyspace is keyspace
            assert net.weight == signal_ab.weight

        assert set(netlist.vertices) == set([vertex_a, vertex_b])
        assert netlist.keyspaces is model.keyspaces
        assert netlist.groups == list()
        assert set(netlist.load_functions) == set([load_fn_a, load_fn_b])
        assert netlist.before_simulation_functions == [pre_fn_a]
        assert netlist.after_simulation_functions == [post_fn_a]

    def test_extra_operators_and_signals(self):
        """Test the operators and signals in the extra_operators and
        extra_signals lists are included when building netlists.
        """
        # Create the first operator
        vertex_a = mock.Mock(name="vertex A")
        load_fn_a = mock.Mock(name="load function A")
        pre_fn_a = mock.Mock(name="pre function A")
        post_fn_a = mock.Mock(name="post function A")

        operator_a = mock.Mock(name="operator A")
        operator_a.make_vertices.return_value = \
            netlistspec(vertex_a, load_fn_a, pre_fn_a, post_fn_a)

        # Create the second operator
        vertex_b = mock.Mock(name="vertex B")
        load_fn_b = mock.Mock(name="load function B")

        operator_b = mock.Mock(name="operator B")
        operator_b.make_vertices.return_value = \
            netlistspec(vertex_b, load_fn_b)

        # Create a signal between the operators
        keyspace = mock.Mock(name="keyspace")
        keyspace.length = 32
        signal_ab = Signal(ObjectPort(operator_a, None),
                           ObjectPort(operator_b, None),
                           keyspace=keyspace, weight=43)

        # Create the model, add the items and then generate the netlist
        model = Model()
        model.extra_operators = [operator_a, operator_b]
        model.extra_signals = [signal_ab]
        netlist = model.make_netlist()

        # Check that the make_vertices functions were called
        operator_a.make_vertices.assert_called_once_with(model)
        operator_b.make_vertices.assert_called_once_with(model)

        # Check that the netlist is as expected
        assert len(netlist.nets) == 1
        for net in netlist.nets:
            assert net.source is vertex_a
            assert net.sinks == [vertex_b]
            assert net.keyspace is keyspace
            assert net.weight == signal_ab.weight

        assert set(netlist.vertices) == set([vertex_a, vertex_b])
        assert netlist.keyspaces is model.keyspaces
        assert netlist.groups == list()
        assert set(netlist.load_functions) == set([load_fn_a, load_fn_b])
        assert netlist.before_simulation_functions == [pre_fn_a]
        assert netlist.after_simulation_functions == [post_fn_a]

    def test_multiple_sink_vertices(self):
        # Create the first operator
        vertex_a = mock.Mock(name="vertex A")
        load_fn_a = mock.Mock(name="load function A")
        pre_fn_a = mock.Mock(name="pre function A")
        post_fn_a = mock.Mock(name="post function A")

        object_a = mock.Mock(name="object A")
        operator_a = mock.Mock(name="operator A")
        operator_a.make_vertices.return_value = \
            netlistspec(vertex_a, load_fn_a, pre_fn_a, post_fn_a)

        # Create the second operator
        vertex_b0 = mock.Mock(name="vertex B0")
        vertex_b1 = mock.Mock(name="vertex B1")
        load_fn_b = mock.Mock(name="load function B")

        object_b = mock.Mock(name="object B")
        operator_b = mock.Mock(name="operator B")
        operator_b.make_vertices.return_value = \
            netlistspec([vertex_b0, vertex_b1], load_fn_b)

        # Create a signal between the operators
        keyspace = mock.Mock(name="keyspace")
        keyspace.length = 32
        signal_ab = Signal(ObjectPort(operator_a, None),
                           ObjectPort(operator_b, None),
                           keyspace=keyspace, weight=3)

        # Create the model, add the items and then generate the netlist
        model = Model()
        model.object_operators[object_a] = operator_a
        model.object_operators[object_b] = operator_b
        model.connections_signals[None] = signal_ab
        netlist = model.make_netlist()

        # Check that the netlist is as expected
        assert set(netlist.vertices) == set([vertex_a, vertex_b0, vertex_b1])
        assert len(netlist.nets) == 1
        for net in netlist.nets:
            assert net.source is vertex_a
            assert net.sinks == [vertex_b0, vertex_b1]
            assert net.keyspace is keyspace
            assert net.weight == signal_ab.weight

        # Check that the groups are correct
        assert netlist.groups == [set([vertex_b0, vertex_b1])]

    def test_multiple_source_vertices(self):
        # Create the first operator
        vertex_a0 = VertexSlice(slice(0, 1))
        vertex_a1 = VertexSlice(slice(1, 2))
        load_fn_a = mock.Mock(name="load function A")
        pre_fn_a = mock.Mock(name="pre function A")
        post_fn_a = mock.Mock(name="post function A")

        object_a = mock.Mock(name="object A")
        operator_a = mock.Mock(name="operator A")
        operator_a.make_vertices.return_value = \
            netlistspec([vertex_a0, vertex_a1], load_fn_a, pre_fn_a, post_fn_a)

        # Create the second operator
        vertex_b = Vertex()
        load_fn_b = mock.Mock(name="load function B")

        object_b = mock.Mock(name="object B")
        operator_b = mock.Mock(name="operator B")
        operator_b.make_vertices.return_value = \
            netlistspec(vertex_b, load_fn_b)

        # Create a signal between the operators
        keyspace = mock.Mock(name="keyspace")
        keyspace.length = 32
        signal_ab = Signal(ObjectPort(operator_a, None),
                           ObjectPort(operator_b, None),
                           keyspace=keyspace, weight=43)

        # Create the model, add the items and then generate the netlist
        model = Model()
        model.object_operators[object_a] = operator_a
        model.object_operators[object_b] = operator_b
        model.connections_signals[None] = signal_ab
        netlist = model.make_netlist()

        # Check that the netlist is as expected
        assert set(netlist.vertices) == set([vertex_a0, vertex_a1, vertex_b])
        assert len(netlist.nets) == 2
        for net in netlist.nets:
            assert net.source in [vertex_a0, vertex_a1]
            assert net.sinks == [vertex_b]

        assert netlist.groups == [set([vertex_a0, vertex_a1])]
