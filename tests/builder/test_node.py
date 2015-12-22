import mock
import nengo
import numpy as np
import pytest
import threading

from nengo_spinnaker import add_spinnaker_params
from nengo_spinnaker.builder import Model
from nengo_spinnaker.builder.model import OutputPort, InputPort
from nengo_spinnaker.builder.node import (
    NodeIOController, InputNode, OutputNode,
    build_node_transmission_parameters
)
from nengo_spinnaker.operators import ValueSink


class TestNodeIOController(object):
    def test_init(self):
        """Test that creating a new NodeIOController instantiates it with an
        empty host network.
        """
        nioc = NodeIOController()
        assert isinstance(nioc.host_network, nengo.Network)
        assert nioc.host_network.all_objects == list()
        assert nioc.host_network.all_connections == list()
        assert nioc.host_network.all_probes == list()

    def test_prepare(self):
        """Preparing the default NodeIOController does nothing."""
        model = mock.Mock()
        controller = mock.Mock(spec_set=[])
        netlist = mock.Mock(spec_set=[])

        nioc = NodeIOController()
        nioc.prepare(model, controller, netlist)

    def test_close(self):
        """Closing the default NodeIOController does nothing."""
        nioc = NodeIOController()
        nioc.close()

    def test_builder_kwargs(self):
        """Test getting builder keyword arguments."""
        nioc = NodeIOController()

        assert nioc.builder_kwargs == {
            "extra_builders": {nengo.Node: nioc.build_node},
            "extra_source_getters": {nengo.Node: nioc.get_node_source},
            "extra_sink_getters": {nengo.Node: nioc.get_node_sink},
            "extra_probe_builders": {nengo.Node: nioc.build_node_probe},
        }

    def test_build_node(self):
        """Test that building a Node does nothing.
        """
        with nengo.Network() as net:
            a = nengo.Node(lambda t, x: x**2, size_in=3, size_out=3)

        # Create the model
        model = Model()
        model.config = net.config

        # Build the Node
        nioc = NodeIOController()
        nioc.build_node(model, a)

        # Assert that no new operators were created
        assert model.object_operators == dict()
        assert model.extra_operators == list()

    @pytest.mark.parametrize("period", [None, 23.0])
    def test_build_node_function_of_time(self, period):
        """Test that building a function of time Node creates a new operator.
        """
        with nengo.Network() as net:
            a = nengo.Node(lambda t: [t, t**2], size_in=0)

        # Mark the Node as a function of time
        add_spinnaker_params(net.config)
        net.config[a].function_of_time = True
        if period is not None:
            net.config[a].function_of_time_period = period

        # Create the model
        model = Model()
        model.config = net.config

        # Build the Node
        nioc = NodeIOController()
        nioc.build_node(model, a)

        # Assert that this added a new operator to the model
        assert model.object_operators[a].function is a.output
        if period is not None:
            assert model.object_operators[a].period == period
        else:
            assert model.object_operators[a].period is None

        assert model.extra_operators == list()

    def test_build_node_constant_value_is_function_of_time(self):
        """Test that building a Node with a constant value is equivalent to
        building a function of time Node.
        """
        with nengo.Network() as net:
            a = nengo.Node(np.array([0.5, 0.1]))

        # Create the model
        model = Model()
        model.config = net.config

        # Build the Node
        nioc = NodeIOController()
        nioc.build_node(model, a)

        # Assert that this added a new operator to the model
        assert model.object_operators[a].function is a.output
        assert model.object_operators[a].period is model.dt

        assert model.extra_operators == list()

    def test_build_node_process_is_not_constant(self):
        """Test that building a Node with a process is not treated the same as
        building a constant valued Node.
        """
        with nengo.Network() as net:
            a = nengo.Node(nengo.processes.Process())

        # Create the model
        model = Model()
        model.config = net.config

        # Build the Node
        nioc = NodeIOController()
        nioc.build_node(model, a)

        # Assert that this added a new operator to the model
        assert model.object_operators[a].function is a.output
        assert model.object_operators[a].period is None

        assert model.extra_operators == list()

    def test_build_node_probe(self):
        """Test that building a Probe of a Node results in adding a new object
        to the model and tries to build a new connection from the Node to the
        Probe.
        """
        with nengo.Network():
            a = nengo.Node(lambda t, x: x**2, size_in=3, size_out=3)
            p = nengo.Probe(a)

        # Create a model to manipulate
        model = mock.Mock(name="model", spec_set=[
            "object_operators", "seeds", "make_connection", "dt"
        ])
        model.object_operators = dict()
        model.seeds = {p: 123}
        model.dt = 0.001

        def make_conn_fn(connection):
            assert connection.pre_obj is a
            assert connection.post_obj is p
            assert connection.synapse is p.synapse

        model.make_connection = make_conn = mock.Mock(wraps=make_conn_fn)

        # Create the builder and build
        nioc = NodeIOController()
        nioc.build_node_probe(model, p)

        # Assert that make_connection was called ONCE
        assert make_conn.call_count == 1

        # Assert that a ValueSink was inserted in the model
        assert isinstance(model.object_operators[p], ValueSink)
        assert model.object_operators[p].probe is p

    def test_get_node_source_standard(self):
        """Test that calling a NodeIOController to get the source for a
        connection which originates at a Node calls the method
        `get_spinnaker_source_for_node` and creates a new OutputNode and adds
        it to the host network.
        """
        with nengo.Network():
            a = nengo.Node(lambda t: t)
            b = nengo.Ensemble(100, 1)
            a_b = nengo.Connection(a, b)

        model = mock.Mock()

        # Create the IO controller
        nioc = NodeIOController()

        # Get the source
        with mock.patch.object(nioc, "get_spinnaker_source_for_node") as gssfn:
            spec = gssfn.return_value = mock.Mock(name="spec")
            assert nioc.get_node_source(model, a_b) is spec

            # Assert this called `get_spinnaker_source_for_node` correctly
            gssfn.assert_called_once_with(model, a_b)

        # Check that `a` is in the host_network as is a new OutputNode, and a
        # connection between them with a synapse of None.
        for node in nioc.host_network.all_nodes:
            if node is not a:
                assert isinstance(node, OutputNode)
                assert node.target is a
                out_node = node
            else:
                assert node is a

        # Check that there is a connection from a to the output node
        assert len(nioc.host_network.all_connections) == 1
        conn = nioc.host_network.all_connections[0]
        assert conn.pre_obj is a
        assert conn.post_obj is out_node
        assert conn.synapse is None

    def test_get_node_source_repeated(self):
        """Test that the same OutputNode is reused if already present.
        """
        with nengo.Network():
            a = nengo.Node(lambda t: t)
            b = nengo.Ensemble(100, 1)
            c = nengo.Ensemble(100, 1)
            a_b = nengo.Connection(a, b)
            a_c = nengo.Connection(a, c)

        model = mock.Mock()

        # Create the IO controller
        nioc = NodeIOController()

        # Get the sources
        with mock.patch.object(nioc, "get_spinnaker_source_for_node"):
            nioc.get_node_source(model, a_b)
            nioc.get_node_source(model, a_c)

        # Check that `a` is in the host_network as is a new OutputNode, and a
        # connection between them with a synapse of None.
        assert len(nioc.host_network.all_nodes) == 2
        for node in nioc.host_network.all_nodes:
            if node is not a:
                assert isinstance(node, OutputNode)
                assert node.target is a
                out_node = node
            else:
                assert node is a

        # Check that there is ONLY ONE connection from a to the output node
        assert len(nioc.host_network.all_connections) == 1
        for conn in nioc.host_network.all_connections:
            assert conn.pre_obj is a
            assert conn.post_obj is out_node
            assert conn.synapse is None

    def test_get_node_source_node_to_node(self):
        """Test that calling the NodeIOController with a Node->Node connection
        doesn't ask for a SpiNNaker source and instead adds the connection to
        the host network.
        """
        with nengo.Network():
            a = nengo.Node(lambda t: t)
            b = nengo.Node(lambda t, x: None, size_in=1)
            a_b = nengo.Connection(a, b)

        model = mock.Mock()

        # Create the IO controller
        nioc = NodeIOController()

        # Get the source
        with mock.patch.object(nioc, "get_spinnaker_source_for_node") as gssfn:
            assert nioc.get_node_source(model, a_b) is None

            # Assert this _didn't_ call `get_spinnaker_source_for_node`
            assert not gssfn.called

        # Check that `a` and `b` are both in the host network and their
        # connection.
        assert nioc.host_network.all_nodes == [a, b]
        assert nioc.host_network.all_connections == [a_b]

    def test_get_node_source_f_of_t(self):
        """Test that calling the NodeIOController for a f_of_t->xx connection
        doesn't ask for a SpiNNaker sorce and instead returns the value source
        that was associated with the Node.
        """
        with nengo.Network() as net:
            a = nengo.Node(lambda t: t)
            b = nengo.Ensemble(100, 1)
            a_b = nengo.Connection(a, b)

        # Mark the Node as being a function of time
        add_spinnaker_params(net.config)
        net.config[a].function_of_time = True

        # Create a model and build the Node
        model = Model()
        model.config = net.config
        nioc = NodeIOController()
        nioc.build_node(model, a)

        # Get the source and ensure that the appropriate object is returned
        with mock.patch.object(nioc, "get_spinnaker_source_for_node") as gssfn:
            spec = nioc.get_node_source(model, a_b)
            assert spec.target.obj is model.object_operators[a]
            assert spec.target.port is OutputPort.standard

            # Assert this _didn't_ call `get_spinnaker_source_for_node`
            assert not gssfn.called

        # There should be nothing in the host network
        assert nioc.host_network.all_nodes == list()
        assert nioc.host_network.all_connections == list()

    @pytest.mark.parametrize("width", [1, 3])
    def test_passthrough_nodes(self, width):
        """Test the handling of passthrough Nodes."""
        with nengo.Network() as net:
            a = nengo.Ensemble(100, width)
            b = nengo.Node(None, size_in=width, label="Passthrough Node")
            c = nengo.Ensemble(100, width)

            a_b = nengo.Connection(a, b)
            b_c = nengo.Connection(b, c)

        # Create a model and build the Node
        model = Model()
        model.config = net.config
        nioc = NodeIOController()
        nioc.build_node(model, b)

        # Check the passthrough Node resulted in a new operator
        assert model.object_operators[b].size_in == b.size_in

        # Get the source and ensure that the appropriate object is returned
        with mock.patch.object(nioc, "get_spinnaker_source_for_node") as gssfn:
            spec = nioc.get_node_source(model, b_c)
            assert spec.target.obj is model.object_operators[b]
            assert spec.target.port is OutputPort.standard

            # Assert this _didn't_ call `get_spinnaker_source_for_node`
            assert not gssfn.called

        # Get the sink and ensure that the appropriate object is returned
        with mock.patch.object(nioc, "get_spinnaker_sink_for_node") as gssfn:
            spec = nioc.get_node_sink(model, a_b)
            assert spec.target.obj is model.object_operators[b]
            assert spec.target.port is InputPort.standard

            # Assert this _didn't_ call `get_spinnaker_sink_for_node`
            assert not gssfn.called

        # There should be nothing in the host network
        assert nioc.host_network.all_nodes == list()
        assert nioc.host_network.all_connections == list()

    def test_passthrough_nodes_with_other_nodes(self):
        """Test the handling of passthrough when other Nodes are present."""
        with nengo.Network() as net:
            a = nengo.Node(lambda t: t, size_in=0, size_out=1)
            b = nengo.Node(None, size_in=1, label="Passthrough Node")
            c = nengo.Node(lambda t, x: None, size_in=1, size_out=0)

            a_b = nengo.Connection(a, b)
            b_c = nengo.Connection(b, c)

        # Create a model and build the Nodes
        model = Model()
        model.config = net.config
        nioc = NodeIOController()
        nioc.build_node(model, a)
        nioc.build_node(model, b)
        nioc.build_node(model, c)

        # Check the passthrough Node resulted in a new operator but that the
        # others didn't
        assert a not in model.object_operators
        assert b in model.object_operators
        assert c not in model.object_operators

        # Get the source and ensure that the appropriate object is returned
        with mock.patch.object(nioc, "get_spinnaker_source_for_node") as gssfn:
            spec = nioc.get_node_source(model, b_c)
            assert spec.target.obj is model.object_operators[b]
            assert spec.target.port is OutputPort.standard

        # Get the sink and ensure that the appropriate object is returned
        with mock.patch.object(nioc, "get_spinnaker_sink_for_node"):
            assert nioc.get_node_sink(model, b_c) is not None
            assert c in nioc._input_nodes

        # Get the sink and ensure that the appropriate object is returned
        with mock.patch.object(nioc, "get_spinnaker_sink_for_node") as gssfn:
            spec = nioc.get_node_sink(model, a_b)
            assert spec.target.obj is model.object_operators[b]
            assert spec.target.port is InputPort.standard

        # Get the source and ensure that the appropriate object is returned
        with mock.patch.object(nioc, "get_spinnaker_source_for_node") as gssfn:
            assert nioc.get_node_source(model, a_b) is not None
            assert a in nioc._output_nodes

    def test_get_node_sink_standard(self):
        """Test that calling a NodeIOController to get the sink for a
        connection which terminates at a Node calls the method
        `get_spinnaker_sink_for_node` and creates a new InputNode and adds
        it to the host network.
        """
        with nengo.Network():
            a = nengo.Ensemble(100, 2)
            b = nengo.Node(lambda t, x: None, size_in=2)
            a_b = nengo.Connection(a, b)

        model = mock.Mock()

        # Create the IO controller
        nioc = NodeIOController()

        # Get the sink
        with mock.patch.object(nioc, "get_spinnaker_sink_for_node") as gssfn:
            spec = gssfn.return_value = mock.Mock(name="spec")
            assert nioc.get_node_sink(model, a_b) is spec

            # Assert this called `get_spinnaker_sink_for_node` correctly
            gssfn.assert_called_once_with(model, a_b)

        # Check that `a` is in the host_network as is a new InputNode, and a
        # connection between them with a synapse of None.
        for node in nioc.host_network.all_nodes:
            if node is not b:
                assert isinstance(node, InputNode)
                assert node.target is b
                in_node = node
            else:
                assert node is b

        # Check that there is a connection from a to the output node
        assert len(nioc.host_network.all_connections) == 1
        conn = nioc.host_network.all_connections[0]
        assert conn.pre_obj is in_node
        assert conn.post_obj is b
        assert conn.synapse is None

        # Check that the Node is included in the Node input dictionary
        assert np.all(nioc.node_input[b] == np.zeros(b.size_in))

    def test_get_node_sink_repeated(self):
        """Test that calling a NodeIOController to get the sink for a
        connection which terminates at a Node calls the method
        `get_spinnaker_sink_for_node` and creates a new InputNode and adds
        it to the host network.
        """
        with nengo.Network():
            a = nengo.Ensemble(100, 1)
            c = nengo.Ensemble(100, 1)
            b = nengo.Node(lambda t, x: None, size_in=1)
            a_b = nengo.Connection(a, b)
            c_b = nengo.Connection(c, b)

        model = mock.Mock()

        # Create the IO controller
        nioc = NodeIOController()

        # Get the sinks
        with mock.patch.object(nioc, "get_spinnaker_sink_for_node"):
            nioc.get_node_sink(model, a_b)
            nioc.get_node_sink(model, c_b)

        # Check that `a` is in the host_network as is a new InputNode, and a
        # connection between them with a synapse of None.
        assert len(nioc.host_network.all_nodes) == 2
        for node in nioc.host_network.all_nodes:
            if node is not b:
                assert isinstance(node, InputNode)
                assert node.target is b
                in_node = node
            else:
                assert node is b

        # Check that there is ONLY ONE connection from a to the output node
        assert len(nioc.host_network.all_connections) == 1
        for conn in nioc.host_network.all_connections:
            assert conn.pre_obj is in_node
            assert conn.post_obj is b
            assert conn.synapse is None

    def test_get_node_sink_node_to_node(self):
        """Test that calling the NodeIOController with a Node->Node connection
        doesn't ask for a SpiNNaker sink.
        """
        with nengo.Network():
            a = nengo.Node(lambda t: t)
            b = nengo.Node(lambda t, x: None, size_in=1)
            a_b = nengo.Connection(a, b)

        model = mock.Mock()

        # Create the IO controller
        nioc = NodeIOController()

        # Get the source
        with mock.patch.object(nioc, "get_spinnaker_sink_for_node") as gssfn:
            assert nioc.get_node_sink(model, a_b) is None

            # Assert this _didn't_ call `get_spinnaker_source_for_node`
            assert not gssfn.called

    def test_get_source_then_sink_of_node_to_node(self):
        """Test that getting the source and then the sink of a Node->Node
        connection just adds those items to the host network.
        """
        with nengo.Network():
            a = nengo.Node(lambda t: [t, t], size_in=0, size_out=2)
            b = nengo.Node(lambda t, x: None, size_in=2, size_out=0)
            a_b = nengo.Connection(a, b)

        model = mock.Mock()

        # Create the IO controller
        nioc = NodeIOController()

        # Get the source and then the sink
        with mock.patch.object(nioc, "get_spinnaker_sink_for_node"), \
                mock.patch.object(nioc, "get_spinnaker_source_for_node"):
            nioc.get_node_source(model, a_b)
            nioc.get_node_sink(model, a_b)

        # The host network should contain a, b and a_b and nothing else
        assert nioc.host_network.all_nodes == [a, b]
        assert nioc.host_network.all_connections == [a_b]


class TestBuildNodeTransmissionParameters(object):
    def test_build_standard_node(self):
        # Create a network
        with nengo.Network():
            a = nengo.Node(lambda t: [t] * 5, size_out=5)
            b = nengo.Ensemble(100, 7)

            func = mock.Mock(side_effect=lambda x: x**2)
            a_b = nengo.Connection(a[0:2], b, function=func,
                                   transform=np.ones((7, 2)))

        # Create an empty model to build into
        model = Model()

        # Build the transmission parameters
        params = build_node_transmission_parameters(model, a_b)
        assert params.pre_slice == slice(0, 2)
        assert params.transform.shape == (7, 2)
        assert params.function is func
        assert np.all(params.transform == 1.0)

    def test_build_standard_node_global_inhibition(self):
        # Create a network
        with nengo.Network():
            a = nengo.Node(lambda t: [t] * 5, size_out=5)
            b = nengo.Ensemble(100, 1)

            a_b = nengo.Connection(a[0:2], b.neurons,
                                   transform=np.ones((b.n_neurons, 2)))

        # Create an empty model to build into
        model = Model()

        # Build the transmission parameters
        params = build_node_transmission_parameters(model, a_b)
        assert params.pre_slice == slice(0, 2)
        assert params.transform.shape == (1, 2)
        assert np.all(params.transform == 1.0)

    def test_build_passthrough_node(self):
        # Create a network
        with nengo.Network():
            a = nengo.Node(None, size_in=5)
            b = nengo.Ensemble(100, 7)

            a_b = nengo.Connection(a[0:2], b, transform=np.ones((7, 2)))

        # Create an empty model to build into
        model = Model()

        # Build the transmission parameters
        params = build_node_transmission_parameters(model, a_b)
        assert params.transform.shape == (7, 5)

    def test_build_passthrough_node_global_inhibition(self):
        # Create a network
        with nengo.Network():
            a = nengo.Node(None, size_in=5)
            b = nengo.Ensemble(100, 1)

            a_b = nengo.Connection(a[0:2], b.neurons,
                                   transform=np.ones((b.n_neurons, 2)))

        # Create an empty model to build into
        model = Model()

        # Build the transmission parameters
        params = build_node_transmission_parameters(model, a_b)
        assert params.transform.shape == (1, 5)


class TestInputNode(object):
    def test_init(self):
        """Test creating an new InputNode from an existing Node, this should
        set the size_in and size_out correctly.
        """
        controller = mock.Mock()
        controller.node_input = dict()

        with nengo.Network():
            a = nengo.Node(lambda t, x: None, size_in=3, size_out=0)
            inn = InputNode(a, controller)

        assert inn.target is a
        assert inn.size_in == 0
        assert inn.size_out == 3
        assert inn.controller is controller

    def test_output(self):
        """Test that calling `output` on an InputNode makes the correct call to
        the controller.
        """
        controller = mock.Mock()
        controller.node_input = dict()
        controller.node_input_lock = threading.Lock()

        with nengo.Network():
            a = nengo.Node(lambda t, x: None, size_in=3, size_out=0)
            inn = InputNode(a, controller)

        controller.node_input[a] = np.random.normal(size=a.size_in)

        assert np.all(inn.output(0.1) == controller.node_input[a])


class TestOutputNode(object):
    @pytest.mark.parametrize("size_out", [1, 4])
    def test_init(self, size_out):
        """Test creating an new OutputNode from an existing Node, this should
        set the size_in and size_out correctly.
        """
        controller = mock.Mock()
        with nengo.Network():
            a = nengo.Node(lambda t: [t] * size_out,
                           size_in=0, size_out=size_out)
            on = OutputNode(a, controller)

        assert on.target is a
        assert on.size_in == size_out
        assert on.size_out == 0
        assert on.controller is controller

    def test_output(self):
        """Test that calling `output` on an OutputNode makes the correct call
        to the controller.
        """
        controller = mock.Mock()
        with nengo.Network():
            a = nengo.Node(lambda t: [t] * 3, size_in=0, size_out=3)
            on = OutputNode(a, controller)

        on.output(0.01, [1, 2, 3])
        controller.set_node_output.assert_called_once_with(a, [1, 2, 3])
