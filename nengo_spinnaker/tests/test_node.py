import mock
import nengo
import pytest

from nengo_spinnaker import intermediate_representation as ir
from nengo_spinnaker import netlist as nl
from nengo_spinnaker.node import (NodeIOController, IntermediateHostNode,
                                  OutputNode, InputNode)


class TestInputNode(object):
    @pytest.mark.parametrize(
        "node",
        [nengo.Node(lambda t, x: None, size_in=5, add_to_container=False),
         nengo.Node(lambda t, x: None, size_in=2, add_to_container=False)]
    )
    def test_init(self, node):
        cn = mock.Mock(spec_set=NodeIOController, name="IO Controller")
        input_node = InputNode(node, cn, add_to_container=False)

        assert input_node.size_in == 0
        assert input_node.size_out == node.size_in
        assert input_node.target is node
        assert input_node.controller is cn

        # Check that calling the input works
        cn.get_node_input.return_value = [300]
        assert input_node.output(0.5) == cn.get_node_input.return_value
        cn.get_node_input.assert_called_once_with(node)


class TestOutputNode(object):
    @pytest.mark.parametrize(
        "node",
        [nengo.Node(lambda t, x: None, size_in=5, add_to_container=False),
         nengo.Node(lambda t, x: None, size_in=2, add_to_container=False)]
    )
    def test_init(self, node):
        cn = mock.Mock(spec_set=NodeIOController, name="IO Controller")
        output_node = OutputNode(node, cn, add_to_container=False)

        assert output_node.size_in == node.size_out
        assert output_node.size_out == 0
        assert output_node.target is node
        assert output_node.controller is cn

        # Check that calling the output works
        output_node.output(0.5, [300])
        cn.set_node_output.assert_called_once_with(node, [300])


class TestNodeIOController(object):
    """Test for the base NodeIOController implementation."""

    def test_get_object_for_node(self):
        """Test returning an intermediate object for a Node."""
        node = mock.Mock(spec_set=[])

        # Should return an IntermediateHostNode
        nioc = NodeIOController()
        ihn = nioc.get_object_for_node(node, 1234)

        assert isinstance(ihn, IntermediateHostNode)
        assert ihn.seed == 1234

    def test_get_source_for_connection(self):
        """Test that getting the source for a Node calls the subclasses
        get_spinnaker_source_for_node method correctly, and adds a new Node and
        connection to the host network.
        """
        # Construct the Node and the connection
        node = nengo.Node(1, add_to_container=False)
        conn = mock.Mock(spec=[nengo.Connection])
        conn.pre_obj = node

        # Construct a mock IRN
        ir_node = IntermediateHostNode(node, 323)
        irn = ir.IntermediateRepresentation({node: ir_node}, {}, [], [])

        # Construct a mock NodeIOController
        nioc = NodeIOController()
        with mock.patch.object(nioc, "get_spinnaker_source_for_node") as fn:
            fn.return_value = None

            # Make the call to get the SpiNNaker source, assert that this call
            # is passed on
            assert nioc.get_source_for_connection(conn, irn) == fn.return_value
            fn.assert_called_once_with(node)

        # Assert that the host network is now not empty
        assert len(nioc.host_network.all_connections) == 1
        assert len(nioc.host_network.all_nodes) == 2

        # We should have the node itself and an output Node which points at it
        for n in nioc.host_network.all_nodes:
            if isinstance(n, OutputNode):
                assert n.target is node
                assert n.controller is nioc
            else:
                assert n is node

        # The connection should link these two items together
        c = nioc.host_network.all_connections[0]
        assert c.pre_obj is node
        assert c.post_obj.target is node

    def test_get_sink_for_node(self):
        """Test that getting the sink for a Node calls the subclasses
        get_spinnaker_sink_for_node method correctly, and adds a new Node and
        connection to the host network.
        """
        # Construct the Node and the connection
        node = nengo.Node(lambda t, v: None, size_in=3, add_to_container=False)
        conn = mock.Mock(spec=[nengo.Connection])
        conn.post_obj = node

        # Construct a mock IRN
        ir_node = IntermediateHostNode(node, 323)
        irn = ir.IntermediateRepresentation({node: ir_node}, {}, [], [])

        # Construct a mock NodeIOController
        nioc = NodeIOController()
        with mock.patch.object(nioc, "get_spinnaker_sink_for_node") as fn:
            fn.return_value = None

            # Make the call to get the SpiNNaker sink, assert that this call
            # is passed on
            assert nioc.get_sink_for_node(conn, irn) == fn.return_value
            fn.assert_called_once_with(node)

        # Assert that the host network is now not empty
        assert len(nioc.host_network.all_connections) == 1
        assert len(nioc.host_network.all_nodes) == 2

        # We should have the node itself and an input Node which points at it
        for n in nioc.host_network.all_nodes:
            if isinstance(n, InputNode):
                assert n.target is node
                assert n.controller is nioc
            else:
                assert n is node

        # The connection should link these two items together
        c = nioc.host_network.all_connections[0]
        assert c.pre_obj.target is node
        assert c.post_obj is node

    def test_get_probe_for_node(self):
        """Test that getting the probe for a Node gets the source for the Node
        and adds this and an extra connection to the intermediate
        representation.
        """
        # Build the Node, the Probe and an empty irn
        with nengo.Network():
            a = nengo.Node(lambda t: t**2, size_out=1)
            p = nengo.Probe(a)

        # Construct a mock IRN
        ir_node = IntermediateHostNode(a, 323)
        irn = ir.IntermediateRepresentation({a: ir_node}, {}, [], [])

        # Construct some extra objects and extra connections
        extra_objects = [mock.Mock(name="extra obj")]
        extra_connections = [mock.Mock(name="extra connection")]
        keyspace = mock.Mock(name="keyspace")
        source = mock.Mock(name="source")

        # Construct a mock NodeIOController
        nioc = NodeIOController()
        with mock.patch.object(nioc, "get_source_for_node") as fn:
            fn.return_value = ir.soss(
                source, extra_objects=extra_objects, 
                extra_nets=extra_connections, latching=True,
                keyspace=keyspace
            )

            ir_probe, extra_objs, extra_conns = nioc.get_probe_for_node(
                p, 454, irn)

        # Check that all of the returned objects are sensible
        assert isinstance(ir_probe, ir.IntermediateObject)

        assert extra_objs == extra_objects + [source]
        assert len(extra_conns) == len(extra_connections) + 1

        for c in (x for x in extra_conns if x not in extra_connections):
            assert isinstance(c, ir.IntermediateNet)
            assert c.seed == 454
            assert c.source.object is source
            assert c.source.port is nl.OutputPort.standard
            assert c.sink.object is ir_probe
            assert c.sink.port is nl.InputPort.standard
            assert c.keyspace is keyspace
            assert c.latching
            assert c.weight == p.size_in
