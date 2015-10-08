import mock
from mock import patch
import nengo
from nengo.cache import NoDecoderCache
import numpy as np
import pytest

from nengo_spinnaker.builder.builder import (
    Model, spec, ObjectPort, _make_signal_parameters
)
from nengo_spinnaker.builder.model import SignalParameters
from nengo_spinnaker.builder.netlist import netlistspec
from nengo_spinnaker.netlist import Vertex, VertexSlice
from nengo_spinnaker import operators


# used for testing _make_signal_parameters
class DummyNode(object):
    size_in = 0


# used for testing _make_signal_parameters
class DummyConnection(object):
    post_obj = DummyNode()


def test_model_init():
    """Test initialising a model, should be completely empty."""
    model = Model()
    assert model.dt == 0.001
    assert model.machine_timestep == 1000

    assert model.params == dict()
    assert model.seeds == dict()

    assert dict(model.object_operators) == dict()
    assert model.extra_operators == list()

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
    """Test the building of connections."""
    @pytest.mark.parametrize("use_make_connection", (True, False))
    def test_standard(self, use_make_connection):
        """Test building a single connection, ensure that all appropriate
        methods are called and that the signal is added to the connection map.
        """
        class A(object):
            pass

        # Create the connection (as a mock)
        connection_source = A()
        connection_sink = A()

        connection = mock.Mock()
        connection.pre_obj = connection_source
        connection.post_obj = connection_sink

        # Create the Model which we'll build with
        m = Model()

        # Modify the Model so that we can interpret calls to the connection map
        m.connection_map = mock.Mock(name="ConnectionMap")

        source = mock.Mock(name="Source Object")
        source_port = mock.Mock(name="Source Port")
        sink = mock.Mock(name="Sink Object")
        sink_port = mock.Mock(name="Sink Port")

        # Add some build methods
        def source_getter(model, conn):
            assert model is m
            assert conn is connection
            return spec(ObjectPort(source, source_port))

        def sink_getter(model, conn):
            assert model is m
            assert conn is connection
            return spec(ObjectPort(sink, sink_port))

        source_getters = {A: mock.Mock(side_effect=source_getter)}
        sink_getters = {A: mock.Mock(side_effect=sink_getter)}

        transmission_parameters = mock.Mock(name="Transmission Params")

        def transmission_builder(model, conn):
            assert model is m
            assert conn is connection
            return transmission_parameters

        reception_parameters = mock.Mock(name="Reception Params")

        def reception_builder(model, conn):
            assert model is m
            assert conn is connection
            return reception_parameters

        transmission_parameter_builders = {
            A: mock.Mock(side_effect=transmission_builder)
        }
        reception_parameter_builders = {
            A: mock.Mock(side_effect=reception_builder)
        }

        # Make the connection
        if use_make_connection:
            # Set an RNG to build with
            m.rng = np.random

            # Set the builders
            m._source_getters = source_getters
            m._sink_getters = sink_getters
            m._transmission_parameter_builders = \
                transmission_parameter_builders
            m._reception_parameter_builders = reception_parameter_builders

            # Build the connection directly
            m.make_connection(connection)
        else:
            # Embed the connection in a mock Nengo network and build that
            # instead.
            network = mock.Mock()
            network.seed = None
            network.connections = [connection]
            network.ensembles = []
            network.nodes = []
            network.networks = []
            network.probes = []

            # Build this (having overridden the builders)
            with mock.patch.object(m, "source_getters", source_getters), \
                    mock.patch.object(m, "sink_getters", sink_getters), \
                    mock.patch.object(m, "transmission_parameter_builders",
                                      transmission_parameter_builders), \
                    mock.patch.object(m, "reception_parameter_builders",
                                      reception_parameter_builders):
                m.build(network)

        # Assert the connection map received an appropriate call
        m.connection_map.add_connection.assert_called_once_with(
            source, source_port, SignalParameters(),
            transmission_parameters, sink, sink_port, reception_parameters
        )

    @pytest.mark.parametrize("no_source, no_sink", ((True, False),
                                                    (False, True)))
    def test_source_is_none(self, no_source, no_sink):
        """Test that if either the source or sink is none no connection is
        added to the model.
        """
        class A(object):
            pass

        # Create the connection (as a mock)
        connection_source = A()
        connection_sink = A()

        connection = mock.Mock()
        connection.pre_obj = connection_source
        connection.post_obj = connection_sink

        # Create the Model which we'll build with
        m = Model()

        # Modify the Model so that we can interpret calls to the connection map
        m.connection_map = mock.Mock(name="ConnectionMap")

        obj = mock.Mock(name="Object")
        obj_port = mock.Mock(name="Port")

        # Add some build methods
        m._source_getters = ({A: lambda m, c: None} if no_source else
                             {A: lambda m, c: ObjectPort(obj, obj_port)})
        m._sink_getters = ({A: lambda m, c: None} if no_sink else
                           {A: lambda m, c: ObjectPort(obj, obj_port)})
        m._transmission_parameter_builders = {A: lambda m, c: None}
        m._reception_parameter_builders = {A: lambda m, c: None}

        # Make the connection
        # Set an RNG to build with
        m.rng = np.random

        # Build the connection directly
        m.make_connection(connection)

        # Assert no call was made to add_connection
        assert not m.connection_map.add_connection.called


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


class TestMakeSignalParameters(object):
    """Test constructing signal parameters from spec objects."""
    @pytest.mark.parametrize("a_is_latching, b_is_latching, latching",
                             [(False, False, False),
                              (True, False, True),
                              (False, True, True),
                              (True, True, True)])
    def test_latching(self, a_is_latching, b_is_latching, latching):
        # Construct the specs
        a_spec = spec(None, latching=a_is_latching)
        b_spec = spec(None, latching=b_is_latching)

        # Make the signal parameters, check they are correct
        sig_pars = _make_signal_parameters(a_spec, b_spec, DummyConnection())
        assert sig_pars.latching is latching

    @pytest.mark.parametrize("source_weight, sink_weight",
                             [(4, 7), (5, 2), (2, 2)])
    def test_weight(self, source_weight, sink_weight):
        """Test that the greatest specified weight is used."""
        # Construct the specs
        a_spec = spec(None, weight=source_weight)
        b_spec = spec(None, weight=sink_weight)

        # Make the signal parameters, check they are correct
        sig_pars = _make_signal_parameters(a_spec, b_spec, DummyConnection())
        assert sig_pars.weight == max((source_weight, sink_weight))

    def test_keyspace_from_source(self):
        """Check that the source keyspace is used if provided."""
        ks = mock.Mock(name="Keyspace")
        a_spec = spec(None, keyspace=ks)
        b_spec = spec(None)

        # Make the signal parameters, check they are correct
        sig_pars = _make_signal_parameters(a_spec, b_spec, DummyConnection())
        assert sig_pars.keyspace is ks

    def test_keyspace_from_sink(self):
        """Check that the sink keyspace is used if provided."""
        ks = mock.Mock(name="Keyspace")
        a_spec = spec(None)
        b_spec = spec(None, keyspace=ks)

        # Make the signal parameters, check they are correct
        sig_pars = _make_signal_parameters(a_spec, b_spec, DummyConnection())
        assert sig_pars.keyspace is ks

    def test_keyspace_collision(self):
        """Test that if both the source and spec provide a keyspace an error is
        raised.
        """
        a_spec = spec(None, keyspace=mock.Mock())
        b_spec = spec(None, keyspace=mock.Mock())

        # Make the signal parameters, this should raise an error
        with pytest.raises(NotImplementedError):
            _make_signal_parameters(a_spec, b_spec, DummyConnection())


class TestMakeNetlist(object):
    """Test production of netlists from operators and signals."""
    def test_calls_add_default_keyspace(self):
        """Test that creating a netlist assigns from default keyspace to the
        network.
        """
        # Create a model and patch out the default keyspace and the connection
        # map.
        default_ks = mock.Mock()
        model = Model(keyspaces={"nengo": default_ks})

        # Create the netlist, ensure that this results in a call to
        # `add_default_keyspace'
        with mock.patch.object(model.connection_map,
                               "add_default_keyspace") as f:
            model.make_netlist()

        f.assert_called_once_with(default_ks)

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
        constraint_a = mock.Mock(name="Constraint B")

        object_a = mock.Mock(name="object A")
        operator_a = mock.Mock(name="operator A")
        operator_a.make_vertices.return_value = \
            netlistspec(vertex_a, load_fn_a, pre_fn_a, post_fn_a,
                        constraint_a)

        # Create the second operator
        vertex_b = mock.Mock(name="vertex B")
        load_fn_b = mock.Mock(name="load function B")
        constraint_b = mock.Mock(name="Constraint B")

        object_b = mock.Mock(name="object B")
        operator_b = mock.Mock(name="operator B")
        operator_b.make_vertices.return_value = \
            netlistspec(vertex_b, load_fn_b, constraints=[constraint_b])

        # Create a signal between the operators
        keyspace = mock.Mock(name="keyspace")
        keyspace.length = 32
        signal_ab_parameters = SignalParameters(keyspace=keyspace, weight=43)

        # Create the model, add the items and then generate the netlist
        model = Model()
        model.object_operators[object_a] = operator_a
        model.object_operators[object_b] = operator_b
        model.connection_map.add_connection(
            operator_a, None, signal_ab_parameters, None,
            operator_b, None, None
        )
        netlist = model.make_netlist()

        # Check that the make_vertices functions were called
        operator_a.make_vertices.assert_called_once_with(model)
        operator_b.make_vertices.assert_called_once_with(model)

        # Check that the netlist is as expected
        assert len(netlist.nets) == 1
        for net in netlist.nets:
            assert net.sources == [vertex_a]
            assert net.sinks == [vertex_b]
            assert net.keyspace is keyspace
            assert net.weight == signal_ab_parameters.weight

        assert set(netlist.vertices) == set([vertex_a, vertex_b])
        assert netlist.keyspaces is model.keyspaces
        assert netlist.groups == list()
        assert set(netlist.constraints) == set([constraint_a, constraint_b])
        assert set(netlist.load_functions) == set([load_fn_a, load_fn_b])
        assert netlist.before_simulation_functions == [pre_fn_a]
        assert netlist.after_simulation_functions == [post_fn_a]

    def test_removes_sinkless_filters(self):
        """Test that making a netlist correctly filters out passthrough Nodes
        with no outgoing connections.
        """
        # Create the first operator
        object_a = mock.Mock(name="object A")
        vertex_a = mock.Mock(name="vertex A")
        load_fn_a = mock.Mock(name="load function A")
        pre_fn_a = mock.Mock(name="pre function A")
        post_fn_a = mock.Mock(name="post function A")

        operator_a = mock.Mock(name="operator A")
        operator_a.make_vertices.return_value = \
            netlistspec(vertex_a, load_fn_a, pre_fn_a, post_fn_a)

        # Create the second operator
        object_b = mock.Mock(name="object B")
        operator_b = operators.Filter(16)  # Shouldn't need building

        # Create the model, add the items and add an entry to the connection
        # map.
        model = Model()
        model.object_operators[object_a] = operator_a
        model.object_operators[object_b] = operator_b
        model.connection_map.add_connection(
            operator_a, None, SignalParameters(), None,
            operator_b, None, None
        )
        netlist = model.make_netlist(1)

        # The netlist should contain vertex a and no nets
        assert netlist.nets == list()
        assert netlist.vertices == [vertex_a]

    def test_extra_operators_and_signals(self):
        """Test the operators in the extra_operators list are included when
        building netlists.
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

        # Create the model, add the items and then generate the netlist
        model = Model()
        model.extra_operators = [operator_a, operator_b]
        netlist = model.make_netlist()

        # Check that the make_vertices functions were called
        operator_a.make_vertices.assert_called_once_with(model)
        operator_b.make_vertices.assert_called_once_with(model)

        # Check that the netlist is as expected
        assert len(netlist.nets) == 0

        assert set(netlist.vertices) == set([vertex_a, vertex_b])
        assert netlist.keyspaces is model.keyspaces
        assert netlist.groups == list()
        assert len(netlist.constraints) == 0
        assert set(netlist.load_functions) == set([load_fn_a, load_fn_b])
        assert netlist.before_simulation_functions == [pre_fn_a]
        assert netlist.after_simulation_functions == [post_fn_a]

    def test_multiple_sink_vertices(self):
        """Test that each of the vertices associated with a sink is correctly
        included in the sinks of a net.
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
        vertex_b0 = mock.Mock(name="vertex B0")
        vertex_b1 = mock.Mock(name="vertex B1")
        load_fn_b = mock.Mock(name="load function B")

        object_b = mock.Mock(name="object B")
        operator_b = mock.Mock(name="operator B")
        operator_b.make_vertices.return_value = \
            netlistspec([vertex_b0, vertex_b1], load_fn_b)

        # Create a third operator, which won't accept the signal
        vertex_c = mock.Mock(name="vertex C")
        vertex_c.accepts_signal.side_effect = lambda _, __: False

        object_c = mock.Mock(name="object C")
        operator_c = mock.Mock(name="operator C")
        operator_c.make_vertices.return_value = netlistspec(vertex_c)

        # Create a signal between the operators
        keyspace = mock.Mock(name="keyspace")
        keyspace.length = 32
        signal_ab_parameters = SignalParameters(keyspace=keyspace, weight=3)

        # Create the model, add the items and then generate the netlist
        model = Model()
        model.object_operators[object_a] = operator_a
        model.object_operators[object_b] = operator_b
        model.object_operators[object_c] = operator_c
        model.connection_map.add_connection(
            operator_a, None, signal_ab_parameters, None,
            operator_b, None, None
        )
        model.connection_map.add_connection(
            operator_a, None, signal_ab_parameters, None,
            operator_c, None, None
        )
        netlist = model.make_netlist()

        # Check that the "accepts_signal" method of vertex_c was called with
        # reasonable arguments
        assert vertex_c.accepts_signal.called

        # Check that the netlist is as expected
        assert set(netlist.vertices) == set(
            [vertex_a, vertex_b0, vertex_b1, vertex_c])
        assert len(netlist.nets) == 1
        for net in netlist.nets:
            assert net.sources == [vertex_a]
            assert net.sinks == [vertex_b0, vertex_b1]
            assert net.keyspace is keyspace
            assert net.weight == signal_ab_parameters.weight

        # Check that the groups are correct
        assert netlist.groups == [set([vertex_b0, vertex_b1])]

        assert len(netlist.constraints) == 0

    def test_multiple_source_vertices(self):
        """Test that each of the vertices associated with a source is correctly
        included in the sources of a net.
        """
        class MyVertexSlice(VertexSlice):
            def __init__(self, *args, **kwargs):
                super(MyVertexSlice, self).__init__(*args, **kwargs)
                self.args = None

            def transmits_signal(self, signal_parameters,
                                 transmission_parameters):
                self.args = (signal_parameters, transmission_parameters)
                return False

        # Create the first operator
        vertex_a0 = VertexSlice(slice(0, 1))
        vertex_a1 = VertexSlice(slice(1, 2))
        vertex_a2 = MyVertexSlice(slice(2, 3))
        load_fn_a = mock.Mock(name="load function A")
        pre_fn_a = mock.Mock(name="pre function A")
        post_fn_a = mock.Mock(name="post function A")

        object_a = mock.Mock(name="object A")
        operator_a = mock.Mock(name="operator A")
        operator_a.make_vertices.return_value = \
            netlistspec([vertex_a0, vertex_a1, vertex_a2],
                        load_fn_a, pre_fn_a, post_fn_a)

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
        signal_ab_parameters = SignalParameters(keyspace=keyspace, weight=43)

        # Create the model, add the items and then generate the netlist
        model = Model()
        model.object_operators[object_a] = operator_a
        model.object_operators[object_b] = operator_b
        model.connection_map.add_connection(
            operator_a, None, signal_ab_parameters, None,
            operator_b, None, None
        )
        netlist = model.make_netlist()

        # Check that the netlist is as expected
        assert set(netlist.vertices) == set([vertex_a0, vertex_a1,
                                             vertex_a2, vertex_b])
        assert len(netlist.nets) == 1
        for net in netlist.nets:
            assert net.sources == [vertex_a0, vertex_a1]
            assert net.sinks == [vertex_b]

        assert netlist.groups == [set([vertex_a0, vertex_a1, vertex_a2])]
        assert len(netlist.constraints) == 0

        # Check that `transmit_signal` was called correctly
        sig, tp = vertex_a2.args
        assert sig.keyspace is keyspace
        assert tp is None
