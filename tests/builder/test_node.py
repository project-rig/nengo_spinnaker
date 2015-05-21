import mock
import nengo
import numpy as np
import pytest
import threading

from nengo_spinnaker.builder.node import (
    NodeIOController, InputNode, OutputNode
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
        controller = mock.Mock(spec_set=[])
        netlist = mock.Mock(spec_set=[])

        nioc = NodeIOController()
        nioc.prepare(controller, netlist)

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
        """Test that building a Node does nothing, at the moment.  Function of
        time Nodes may do something here later.
        """
        with nengo.Network():
            a = nengo.Node(lambda t, x: x**2, size_in=3, size_out=3)

        nioc = NodeIOController()
        nioc.build_node(None, a)

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

        # Check that there is a connection from a to the output node
        assert len(nioc.host_network.all_connections) == 2
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
