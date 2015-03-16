"""Nengo/SpiNNaker Builder

Converts Nengo networks into forms suitable for simulation on SpiNNaker and a
connected PC.
"""
import nengo


def separate_networks(objects, connections):
    """Splits a single Nengo Network into a SpiNNaker-specific network and a
    host-specific network.

    .. warning::
        This function is best run _after_ removing pass through Nodes otherwise
        connections which could be between Ensembles may end up being routed
        via host computer.

    Parameters
    ----------
    objects : list
        List of Nengo objects.
    connections : list
        Connections between those objects.

    Returns
    -------
    (spinnaker_obj, spinnaker_conns), (host_obj, host_conns)
        Tuple of SpiNNaker and host objects and connections.
    """
    # Split the objects into Ensembles and Nodes, the first step in splitting
    # into SpiNNaker- and host- targeted networks.
    spinnaker_objs = {o for o in objects if isinstance(o, nengo.Ensemble)}
    host_objs = {o for o in objects if isinstance(o, nengo.Node)}

    # Go through the connections, any Node->Node connections are added to host
    # connections, Node->Ensemble and Ensemble->Node connections are added to
    # both and Ensemble->Ensemble / Neuron->Neuron / ...
    spinnaker_conns = list()
    host_conns = list()

    for conn in connections:
        if (isinstance(conn.pre_obj, nengo.Node) and
                isinstance(conn.post_obj, nengo.Ensemble)):
            # Node -> Ensemble connections result in the creation of new
            # objects in both the host and SpiNNaker networks and appropriate
            # connections.  The Node is added to the SpiNNaker network and a
            # new PCToBoardNode is added to the host network.
            spinnaker_objs.add(conn.pre_obj)
            spinnaker_conns.append(conn)

            pc_out_node = PCToBoardNode(conn.pre_obj, add_to_container=False)
            host_objs.add(pc_out_node)
            host_conns.append(
                nengo.Connection(conn.pre_obj, pc_out_node, synapse=None,
                                 add_to_container=False)
            )
        elif (isinstance(conn.pre_obj, nengo.Ensemble) and
                isinstance(conn.post_obj, nengo.Node)):
            # Ensemble -> Node connections result in the creation of new
            # objects in both the host and SpiNNaker networks and appropriate
            # connections.  The Node is added to the SpiNNaker network and a
            # new PCToBoardNode is added to the host network.
            spinnaker_objs.add(conn.post_obj)
            spinnaker_conns.append(conn)

            pc_in_node = PCFromBoardNode(conn.post_obj, add_to_container=False)
            host_objs.add(pc_in_node)
            host_conns.append(
                nengo.Connection(pc_in_node, conn.post_obj, synapse=None,
                                 add_to_container=False)
            )
        elif (isinstance(conn.pre_obj, nengo.Node) and
                isinstance(conn.post_obj, nengo.Node)):
            # Node -> Node connections are listed in the host connections, and
            # nothing else need be done.
            host_conns.append(conn)
        elif (isinstance(conn.pre_obj, nengo.Ensemble) and
                isinstance(conn.post_obj, nengo.Ensemble)):
            # Ensemble -> Ensemble connections are listed in the SpiNNaker
            # connections and nothing else happens.
            spinnaker_conns.append(conn)
        else:  # pragma : no cover
            # TODO Support Ensemble->Neuron, etc.
            raise NotImplementedError(conn)

    return (spinnaker_objs, spinnaker_conns), (host_objs, host_conns)


class PCToBoardNode(nengo.Node):
    """Node which will transmit data from a Node to SpiNNaker."""
    def __init__(self, node):
        """Create a new Node to handle communication from a Node to SpiNNaker.

        Parameters
        ----------
        node : :py:class:`~nengo.Node`
        """
        super(PCToBoardNode, self).__init__(
            self.output_fn, node.size_out, 0,
            "{}->SpiNNaker".format(node.label)
        )
        self.node = node

    def output_fn(self, t, x):
        raise NotImplementedError


class PCFromBoardNode(nengo.Node):
    """Node which will receive data for a Node from SpiNNaker."""
    def __init__(self, node):
        """Create a new Node to handle communication to a Node from SpiNNaker.

        Parameters
        ----------
        node : :py:class:`~nengo.Node`
        """
        super(PCFromBoardNode, self).__init__(
            self.output_fn, 0, node.size_in,
            "SpiNNaker->{}".format(node.label)
        )
        self.node = node

    def output_fn(self, t):
        raise NotImplementedError
