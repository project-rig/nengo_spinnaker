import nengo

from .annotations import AnnotatedNet, ObjectAnnotation
from . import annotations as anns


class IntermediateHostNode(AnnotatedNet):
    """Intermediate object representing a Node which will be simulated on the
    host and consequently will not directly result in a SpiNNaker executable.
    """


class NodeIOController(object):
    """Object which controls communication between SpiNNaker and a host
    computer for transmitting output and receiving input for Node objects.

    The standard NodeIOController should not be used, but rather one of its
    subclasses.  Subclasses are required to provide methods to fill in the
    SpiNNaker side of Node simulation; these methods are:

        - `get_spinnaker_source_for_node`
        - `get_spinnaker_sink_for_node`

    A `NodeIOController` can be used during the construction of an intermediate
    representation using the `extra_object_builders`, `extra_source_getters`,
    `extra_sink_getters` and `extra_probe_builders` arguments.

    The following two methods must also be implemented:

        - `get_node_input`
        - `set_node_output`
    """
    def __init__(self):
        # Create an empty Nengo network which will contain things which are to
        # be simulated on the host.
        self.host_network = nengo.Network()

    def get_object_for_node(self, node, seed):
        """Get an intermediate object for the Node."""
        # TODO Identify when a function of time Node is being passed and act
        # differently.
        return IntermediateHostNode(node, seed)

    def get_source_for_connection(self, conn, irn):
        """Get a source for a connection from a Node."""
        return self.get_source_for_node(conn.pre_obj)

    def get_source_for_node(self, node):
        # Get the SpiNNaker source objects
        spinn_source = self.get_spinnaker_source_for_node(node)

        # Add the Node, a new OutputNode and a connection between them to the
        # host network.
        with self.host_network:
            self.host_network.add(node)
            o_node = OutputNode(node, self)
            nengo.Connection(node, o_node, synapse=None)

        # Return the SpiNNaker source objects
        return spinn_source

    def get_spinnaker_source_for_node(self, node):  # pragma : no cover
        """Get the source for a connection from a given Node.

        **OVERRIDE THIS METHOD**  The return type should be as documented for
        `nengo_spinnaker.intermediate_representation.\
         IntermediateRepresentation.source_getters`.

        Parameters
        ----------
        node : :py:class:`nengo.Node`
        """
        raise NotImplementedError

    def get_sink_for_node(self, conn, irn):
        """Get a sink for a connection into a Node."""
        # Get the SpiNNaker sink objects
        spinn_sink = self.get_spinnaker_sink_for_node(conn.post_obj)

        # Add the Node, a new InputNode and a connection between them to the
        # host network.
        with self.host_network:
            self.host_network.add(conn.post_obj)
            i_node = InputNode(conn.post_obj, self)
            nengo.Connection(i_node, conn.post_obj, synapse=None)

        # Return the SpiNNaker sink objects
        return spinn_sink

    def get_spinnaker_sink_for_node(self, node):  # pragma : no cover
        """Get the sink for a connection to a given Node.

        **OVERRIDE THIS METHOD**  The return type should be as documented for
        `nengo_spinnaker.intermediate_representation.\
         IntermediateRepresentation.sink_getters`.

        Parameters
        ----------
        node : :py:class:`nengo.Node`
        """
        raise NotImplementedError

    def get_probe_for_node(self, probe, seed, irn):
        """Get a probe object for the Node."""
        # Get a source for the Node; then add a new probe object and a
        # connection from the source to the probe.
        source_spec = self.get_source_for_node(probe.target)
        probe_object = ObjectAnnotation(probe)
        probe_conn = AnnotatedNet(
            anns.NetAddress(source_spec.target, anns.OutputPort.standard),
            anns.NetAddress(probe_object, anns.InputPort.standard),
            keyspace=source_spec.keyspace,
            latching=source_spec.latching,
            weight=probe.size_in
        )

        objects = source_spec.extra_objects + [source_spec.target]
        conns = source_spec.extra_nets + [probe_conn]

        return probe_object, objects, conns

    def get_node_input(self, node):  # pragma : no cover
        """Return the current input value for a Node.

        Parameters
        ----------
        node : :py:class:`nengo.Node`

        Returns
        -------
        array
            Last received input value for a Node.
        """
        raise NotImplementedError

    def set_node_output(self, node, value):  # pragma : no cover
        """Send the current output of a Node to SpiNNaker.

        Parameters
        ----------
        node : :py:class:`nengo.Node`
        value : array
        """
        raise NotImplementedError


class InputNode(nengo.Node):
    """Node which transmits simulation data to SpiNNaker."""
    def __init__(self, node, io_controller):
        self.target = node
        self.controller = io_controller
        self.size_in = 0
        self.size_out = node.size_in

    def output(self, t):
        return self.controller.get_node_input(self.target)


class OutputNode(nengo.Node):
    """Node which transmits simulation data to SpiNNaker."""
    def __init__(self, node, io_controller):
        self.target = node
        self.controller = io_controller
        self.size_in = node.size_out
        self.size_out = 0

    def output(self, t, value):
        self.controller.set_node_output(self.target, value)
