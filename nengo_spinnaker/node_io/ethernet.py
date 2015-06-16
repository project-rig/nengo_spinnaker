import collections
import numpy as np
from rig.machine_control.packets import SCPPacket
from rig.machine import Cores
from six import iteritems
import socket
import threading

from ..builder.builder import spec, InputPort, OutputPort, ObjectPort
from ..builder.node import NodeIOController
from ..operators import SDPReceiver, SDPTransmitter
from ..utils import type_casts as tp


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
        self._sdp_receivers = dict()
        self._sdp_transmitters = dict()

        # Node -> [(Connection, (x, y, p), ...]
        self._node_outgoing = collections.defaultdict(list)

        # (x, y, p) -> Node
        self._node_incoming = dict()

        # Sockets
        self._hostname = None
        self.in_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.out_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def get_spinnaker_source_for_node(self, model, connection):
        """Get the source for a connection originating from a Node.

        Arguments and return type are as for
        :py:attr:`~nengo_spinnaker.builder.Model.source_getters`.
        """
        # Create a new SDPReceiver if there isn't already one for the Node
        if connection.pre_obj not in self._sdp_receivers:
            receiver = SDPReceiver()
            self._sdp_receivers[connection.pre_obj] = receiver
            model.extra_operators.append(receiver)

        return spec(ObjectPort(self._sdp_receivers[connection.pre_obj],
                               OutputPort.standard),
                    latching=True)

    def get_spinnaker_sink_for_node(self, model, connection):
        """Get the sink for a connection terminating at a Node.

        Arguments and return type are as for
        :py:attr:`~nengo_spinnaker.builder.Model.sink_getters`.
        """
        # Create a new SDPTransmitter if there isn't already one for the Node
        if connection.post_obj not in self._sdp_transmitters:
            transmitter = SDPTransmitter(connection.post_obj.size_in)
            self._sdp_transmitters[connection.post_obj] = transmitter
            model.extra_operators.append(transmitter)

        return spec(ObjectPort(self._sdp_transmitters[connection.post_obj],
                               InputPort.standard))

    def prepare(self, controller, netlist):
        """Prepare for simulation given the placed netlist and the machine
        controller.
        """
        # Store the hostname
        # TODO Store a map of (x, y) to hostname
        self._hostname = controller.initial_host

        # Set up the IP tag (will need to do this for each ethernet connected
        # chip that we expect to use).
        self.in_socket.bind(('', 50007))
        with controller(x=0, y=0):
            controller.iptag_set(1, *self.in_socket.getsockname())

        # Build a map of Node to outgoing connections and SDP receivers
        for node, sdp_rx in iteritems(self._sdp_receivers):
            for connection, vertex in iteritems(sdp_rx.connection_vertices):
                # Get the placement and core
                x, y = netlist.placements[vertex]
                p = netlist.allocations[vertex][Cores].start

                # Store this connection -> (x, y, p) map
                self._node_outgoing[node].append((connection, (x, y, p)))

        # Build a map of (x, y, p) to Node for incoming values
        for node, sdp_tx in iteritems(self._sdp_transmitters):
            # Get the placement and core
            x, y = netlist.placements[sdp_tx._vertex]
            p = netlist.allocations[sdp_tx._vertex][Cores].start

            # Store this mapping (x, y, p) -> Node
            self._node_incoming[(x, y, p)] = node

    def set_node_output(self, node, value):
        """Transmit the value output by a Node."""
        # Build an SDP packet to transmit for each outgoing connection for the
        # node
        for connection, (x, y, p) in self._node_outgoing[node]:
            # Apply the pre-slice, the connection function and the transform.
            c_value = value[connection.pre_slice]
            if connection.function is not None:
                c_value = connection.function(c_value)
            c_value = np.dot(connection.transform, c_value)

            # Transmit the packet
            packet_data = bytes(tp.np_to_fix(c_value).data)
            packet = SCPPacket(
                reply_expected=False, tag=0xff,
                dest_port=1, dest_cpu=p,
                src_port=7, src_cpu=31,
                dest_x=x, dest_y=y,
                src_x=0, src_y=0,
                cmd_rc=0, seq=0, arg1=0, arg2=0, arg3=0,
                data=packet_data
            )
            self.out_socket.sendto(packet.bytestring,
                                   (self._hostname, 17893))

    def spawn(self):
        """Get a new thread which will manage transmitting and receiving Node
        values.
        """
        return EthernetThread(self)

    def close(self):
        """Close the sockets used by the ethernet Node IO."""
        self.in_socket.close()
        self.out_socket.close()


class EthernetThread(threading.Thread):
    """Thread which handles transmitting and receiving IO values."""
    def __init__(self, ethernet_handler):
        # Initialise the thread
        super(EthernetThread, self).__init__(name="EthernetIO")

        # Set up internal references
        self.halt = False
        self.handler = ethernet_handler
        self.in_sock = ethernet_handler.in_socket
        self.in_sock.settimeout(0.0001)

    def run(self):
        while not self.halt:
            # Read as many packets from the socket as we can
            while True:
                try:
                    data = self.in_sock.recv(512)
                except IOError:
                    break  # No more to read

                # Unpack the data, and store it as the input for the
                # appropriate Node.
                packet = SCPPacket.from_bytestring(data)
                values = tp.fix_to_np(
                    np.frombuffer(packet.data, dtype=np.int32)
                )

                # Get the Node
                node = self.handler._node_incoming[(packet.src_x,
                                                    packet.src_y,
                                                    packet.src_cpu)]
                with self.handler.node_input_lock:
                    self.handler.node_input[node] = values[:]

    def stop(self):
        """Stop the thread from running."""
        self.halt = True
        self.join()
