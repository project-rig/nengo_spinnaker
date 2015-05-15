import collections

from ..builder.builder import spec, InputPort, OutputPort, ObjectPort
from ..builder.node import NodeIOController
from ..operators import SDPReceiver, SDPTransmitter


class Ethernet(NodeIOController):
    """Ethernet implementation of SpiNNaker to host node communication."""

    def __init__(self, transmission_period=0.01):
        """Create a new Ethernet based Node communicator.

        Parameters
        ----------
        transmission_period : float
            Period between transmitting SDP packets from SpiNNaker to the host
            in seconds.
        """
        super(Ethernet, self).__init__()

        # Store ethernet specific parameters
        self.transmission_period = transmission_period
        self._sdp_receivers = collections.defaultdict(SDPReceiver)
        self._sdp_transmitters = collections.defaultdict(SDPTransmitter)

    def get_spinnaker_source_for_node(self, model, connection):
        """Get the source for a connection originating from a Node.

        Arguments and return type are as for
        :py:attr:`~nengo_spinnaker.builder.Model.source_getters`.
        """
        return spec(ObjectPort(self._sdp_receivers[connection.pre_obj],
                               OutputPort.standard),
                    latching=True)

    def get_spinnaker_sink_for_node(self, model, connection):
        """Get the sink for a connection terminating at a Node.

        Arguments and return type are as for
        :py:attr:`~nengo_spinnaker.builder.Model.sink_getters`.
        """
        return spec(ObjectPort(self._sdp_transmitters[connection.post_obj],
                               InputPort.standard))

    def set_node_output(self, node, value):
        """Transmit the value output by a Node."""
        raise NotImplementedError

    def spawn(self):
        """Get a new thread which will manage transmitting and receiving Node
        values.
        """
        raise NotImplementedError
