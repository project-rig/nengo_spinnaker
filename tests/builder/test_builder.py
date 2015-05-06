import mock
from mock import patch
import nengo
from nengo.cache import NoDecoderCache
import numpy as np
import pytest

from nengo_spinnaker.builder.builder import Model, Signal, spec, _make_signal


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

    assert dict(model.object_intermediates) == dict()
    assert dict(model.connections_signals) == dict()

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

        if not use_make_object:
            # Patch the default builders
            with patch.object(Model, "builders", new=builders):
                # Create a model and build the mock network
                model = Model()
                model.build(network, extra_builders=extra_builders)
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

    def test_passthrough_nodes_removed(self):
        """Test that passthrough Nodes are removed."""
        with nengo.Network() as network:
            a = nengo.Ensemble(100, 3)
            b = nengo.Node(None, size_in=3, size_out=3)
            c = nengo.Ensemble(100, 3)

            nengo.Connection(a, b, synapse=None)
            nengo.Connection(b, c)

        # Create a generic builder which just ensures that it is NEVER passed
        # `b`
        def generic_builder_fn(model, obj):
            assert obj is not b

        def generic_getter(model, conn):
            pass

        generic_builder = mock.Mock(wraps=generic_builder_fn)
        builders = {nengo.base.NengoObject: generic_builder}

        # Build the model
        with patch.object(Model, "builders", new=builders),\
                patch.object(Model, "source_getters",
                             new={object: generic_getter}),\
                patch.object(Model, "sink_getters",
                             new={object: generic_getter}):
            model = Model()
            model.build(network)

            assert generic_builder.call_count == 2


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
        source = mock.Mock()
        sink = mock.Mock()

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

        # Create a mock network
        network = mock.Mock()
        network.seed = None
        network.connections = [connection]
        network.ensembles = []
        network.nodes = []
        network.networks = []

        if use_registered_dicts:
            # Patch the getters, add a null builder
            with patch.object(model, "source_getters", {A: source_getter}), \
                    patch.object(model, "sink_getters", {B: sink_getter}):
                # Build the network
                model.build(network)
        else:
            model.build(network,
                        extra_source_getters={A: source_getter},
                        extra_sink_getters={B: sink_getter})

        # Check that seeds were provided
        if seed is not None:
            assert model.seeds[connection] == seed
        else:
            assert model.seeds[connection] is not None

        # Assert the getters were called
        assert source_getter.call_count == 1
        assert sink_getter.call_count == 1

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

        # Patch the getters, add a null builder
        with patch.object(model, "source_getters", {A: source_getter}), \
                patch.object(model, "sink_getters", {A: sink_getter}):
            # Build the network
            model.build(network)

        # Assert that no signal exists
        assert connection not in model.connections_signals


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
