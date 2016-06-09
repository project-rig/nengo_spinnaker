from __future__ import absolute_import

import collections
import logging
import nengo.synapses
import numpy as np
from six.moves import filter as sfilter
from six import iteritems, itervalues
from toposort import toposort_flatten

from nengo_spinnaker.builder.ensemble import EnsembleTransmissionParameters
from nengo_spinnaker.builder.model import (
    ReceptionParameters, SignalParameters
)
from nengo_spinnaker.builder.node import (
    PassthroughNodeTransmissionParameters, NodeTransmissionParameters
)
from nengo_spinnaker.builder.ports import EnsembleInputPort
from nengo_spinnaker.utils.config import getconfig


logger = logging.getLogger(__name__)


def get_force_removal_passnodes(network):
    """Get a set of which passthrough Nodes should be forcibly removed
    regardless of the configuration options, currently this is any passthrough
    Nodes which are directly or indirectly connected (through other passthrough
    Nodes) to Neurons.
    """
    def is_ptn(n):
        return isinstance(n, nengo.Node) and n.output is None

    # Build a dictionary representing which objects are connected to which
    # other objects (undirected graph).
    all_io = collections.defaultdict(list)
    for conn in network.all_connections:
        all_io[conn.pre_obj].append(conn.post_obj)
        all_io[conn.post_obj].append(conn.pre_obj)

    # Find passthrough Nodes which are directly connected to neurons and mark
    # them for removal.
    force_removal = {n for n in network.all_nodes if is_ptn(n) and
                     any(isinstance(o, nengo.ensemble.Neurons)
                         for o in all_io[n])}

    # For each passthrough Node perform a search to determine which other
    # passthrough Nodes it is connected to.
    def find_connected_nodes(node, all_children=None):
        all_children = all_children or set()  # Ensure we have a set
        all_children.add(node)  # Mark ourselves as visited

        # Recursively get all child nodes, updating `all_children` in place
        for n in sfilter(lambda m: is_ptn(m) and m not in all_children,
                         all_io[node]):
            find_connected_nodes(n, all_children)

        return all_children

    # Find all the passthrough Nodes that connect directly or indirectly to
    # neurons.
    remaining_nodes = set(force_removal)
    while remaining_nodes:
        # Get a start node then mark all connected Nodes as requiring removal
        # and remove them from the list of remaining Nodes.
        start_node = next(iter(remaining_nodes))
        nodes = find_connected_nodes(start_node)
        force_removal.update(nodes)
        remaining_nodes.difference_update(nodes)

    return force_removal


def get_passthrough_node_dependencies(model, passthrough_nodes):
    """Create a dictionary mapping from nodes and operators to all the nodes
    and operators which transmit to it.

    The resultant dictionary may be used to topologically sort the passthrough
    Nodes.
    """
    # Create an empty dependency dictionary
    deps = {n_op: set() for n_op in iteritems(passthrough_nodes)}

    # Store a mapping : {operator: {Node, operator}
    n_ops = {op: (n, op) for n, op in iteritems(passthrough_nodes)}

    # Iterate through the signals to get those whose sinks include an operator
    # about which we care and whose source is also an operator we care about.
    for sig, _ in model.connection_map.get_signals():
        if sig.source in n_ops:
            for sink in sig.sinks:
                if sink in n_ops:
                    # The sink and source are both passthrough nodes.
                    deps[n_ops[sink]].add(n_ops[sig.source])

    return deps


def order_passthrough_nodes(model, passthrough_nodes):
    """Order passthrough Nodes for removal."""
    return reversed(toposort_flatten(
        get_passthrough_node_dependencies(model, passthrough_nodes)
    ))


def optimise_out_passthrough_nodes(model, passthrough_nodes, config,
                                   forced_removals=set()):
    """Remove passthrough Nodes from a network.

    Other Parameters
    ----------------
    forced_removals : {Node, ...}
        Set of Nodes which should be removed regardless of the configuration
        settings.
    """
    for node, operator in order_passthrough_nodes(model, passthrough_nodes):
        removed = False

        # Determine whether to remove the Node or not (if True then definitely
        # remove, if None then remove if it doesn't worsen network usage).
        remove_node = (node in forced_removals or
                       getconfig(config, node, "optimize_out"))
        if remove_node or remove_node is None:
            removed = remove_operator_from_connection_map(
                model.connection_map, operator, force=bool(remove_node),
                weight=len(operator.groups)
            )

        # Log if the Node was removed
        if removed:
            if node in model.object_operators:
                model.object_operators.pop(node)
            else:
                model.extra_operators.remove(operator)

            logger.info("Passthrough Node {!s} was optimized out".format(node))


def remove_operator_from_connection_map(conn_map, target, force=True,
                                        weight=1.0):
    """Remove an operator from a connection map by combining the connections
    that lead to and from the operator.

    Parameters
    ----------
    conn_map : `ConnectionMap`
        Connection map from which the operator will be removed.  Note that the
        connection map will be modified.
    target : object
        Operator to remove from the connection map.

    Other Parameters
    ----------------
    weight : int or float
        Multiplier of the cost of the network traffic with the operator left in
        place (used, for example, to represent the number of column divisions
        of a matrix multiply).
    force : bool
        If False then the operator will only be optimised out if it is expected
        that doing so will break network performance. If True (the default)
        then the operator will be removed regardless.
    """
    # Grab the old connection map
    old_conns = conn_map._connections.copy()
    saved_conns = conn_map._connections.copy()

    # Grab all the connections which are transmitted by the operator.
    if target not in old_conns:
        return True
    out_conns = list(_get_port_kwargs(old_conns.pop(target)))

    # Compute the most packets any object which receives packets from the
    # operator will receive if the operator is not removed.
    old_max_packets = _get_max_packets_received(out_conns) * weight

    # Prepare to compute the new equivalent of this value
    new_rx = collections.defaultdict(lambda: 0)

    # Create a new empty connection map dictionary and update the connection
    # map to use the new dictionary
    conns = collections.defaultdict(lambda: collections.defaultdict(list))
    conn_map._connections = conns

    # Copy across all connections from the old connection map; every time we
    # encounter the operator as the sink of a signal we multiply that signal by
    # each of the outgoing connections in turn and add those connections to the
    # new connection map instead.
    for kwargs in _iter_connection_map_as_kwargs(old_conns):
        # The target will never appear as a source object as we removed it as a
        # key before. Therefore determine what to do by looking at the sink
        # object.
        if kwargs["sink_object"] is target:
            # If the sink object is the target then create multiply this
            # connection with each of the outgoing signals for the target.
            for new_kwargs in _multiply_signals(kwargs, out_conns):
                conn_map.add_connection(**new_kwargs)

                # Update the new number of packets that will be received by
                # each sink (assuming that zeroed rows will be removed from the
                # transform)
                weight = np.sum(np.any(
                    new_kwargs["transmission_parameters"].transform != 0.0,
                    axis=1)
                )
                new_rx[new_kwargs["sink_object"]] += weight
        else:
            # Otherwise (re-)add this connection to the connection map.
            conn_map.add_connection(**kwargs)

    # Determine whether the changes made should be discarded.
    new_max_packets = 0 if not new_rx else max(itervalues(new_rx))
    discard_changes = new_max_packets > old_max_packets

    # If not forced and we caused a worsening in network usage then copy the
    # old connection map back in and discard our changes.
    if not force and discard_changes:
        conn_map._connections = saved_conns
        return False
    else:
        return True


def _iter_connection_map_as_kwargs(conn_dict):
    """Iterate over a connection map and yield keywords for every connection.

    Yields
    ------
    dict
        Keyword arguments for `ConnectionMap.add_connection`
    """
    # Iterate through the dictionary
    for source_object, ports_and_signals in iteritems(conn_dict):
        for kwargs in _get_port_kwargs(ports_and_signals):
            # Add the source object to the kwargs and yield
            kwargs["source_object"] = source_object
            yield kwargs


def _get_port_kwargs(ports_and_signals):
    """
    Yields
    ------
    dict
        Keyword arguments for `ConnectionMap.add_connection`
    """
    # Iterate through the dictionary
    for source_port, signals_and_sinks in iteritems(ports_and_signals):
        for signal_and_sink in signals_and_sinks:
            for (sink_object, sink_port,
                 reception_parameters) in signal_and_sink.sinks:
                # Break out the signal and transmission parameters
                signal_parameters, transmission_parameters = \
                    signal_and_sink.parameters

                # Yield the keyword arguments for this connection
                yield {
                    "source_port": source_port,
                    "signal_parameters": signal_parameters,
                    "transmission_parameters": transmission_parameters,
                    "sink_object": sink_object,
                    "sink_port": sink_port,
                    "reception_parameters": reception_parameters,
                }


def _get_max_packets_received(port_kwargs):
    """Get the maximum number of packets received by any single object in the
    specified port kwargs.
    """
    rx = collections.defaultdict(lambda: 0)
    for kwargs in port_kwargs:
        rx[kwargs["sink_object"]] += kwargs["signal_parameters"].weight

    return max(itervalues(rx))


def _multiply_signals(in_kwargs, out_conn_kwargs):
    """Multiply an input connection by a selection of outgoing connection and
    yield keywords for every new connection.

    Yields
    ------
    dict
        Keyword arguments for `ConnectionMap.add_connection`
    """
    # For every outgoing connection
    for out_conn in out_conn_kwargs:
        # Combine the transmission parameters
        transmission_parameters, sink_port = _combine_transmission_params(
            in_kwargs["transmission_parameters"],
            out_conn["transmission_parameters"],
            out_conn["sink_port"]
        )

        # If the connection has been optimised out then move on
        if transmission_parameters is None and sink_port is None:
            continue

        # Combine the reception parameters
        reception_parameters = _combine_reception_params(
            in_kwargs["reception_parameters"],
            out_conn["reception_parameters"],
        )

        # Combine the signal parameters: the new signal will be latching if
        # either the input or the output signals require it be so, it will have
        # the weight assigned by the reception parameters. If either the input
        # or output keyspace are None then the keyspace assigned to the other
        # signal will be used; if neither are None then we break because
        # there's no clear way to merge keyspaces.
        in_sig_pars = in_kwargs["signal_parameters"]
        out_sig_pars = out_conn["signal_parameters"]

        latching = in_sig_pars.latching or out_sig_pars.latching
        weight = out_sig_pars.weight

        if in_sig_pars.keyspace is None or out_sig_pars.keyspace is None:
            keyspace = in_sig_pars.keyspace or out_sig_pars.keyspace
        else:
            raise NotImplementedError("Cannot merge keyspaces")

        # Construct the new signal parameters
        signal_parameters = SignalParameters(latching, weight, keyspace)

        # Yield the new keyword arguments
        yield {
            "source_object": in_kwargs["source_object"],
            "source_port": in_kwargs["source_port"],
            "signal_parameters": signal_parameters,
            "transmission_parameters": transmission_parameters,
            "sink_object": out_conn["sink_object"],
            "sink_port": sink_port,
            "reception_parameters": reception_parameters,
        }


def _combine_transmission_params(in_transmission_parameters,
                                 out_transmission_parameters,
                                 final_port):
    """Combine transmission parameters to join two signals into one, e.g., for
    optimising out a passthrough Node.

    Returns
    -------
    transmission_parameters
        New transmission parameters
    port
        New receiving port for the connection
    """
    assert isinstance(out_transmission_parameters,
                      PassthroughNodeTransmissionParameters)

    # Compute the new transform
    new_transform = np.dot(out_transmission_parameters.transform,
                           in_transmission_parameters.transform)

    # If the resultant transform is empty then we return None to indicate that
    # the connection should be dropped.
    if np.all(new_transform == 0.0):
        return None, None

    # If the connection is a global inhibition connection then truncate the
    # transform and modify the final port to reroute the connection.
    if (final_port is EnsembleInputPort.neurons and
            np.all(new_transform[0] == new_transform[1:])):
        # Truncate the transform
        new_transform = new_transform[0]
        new_transform.shape = (1, -1)  # Ensure the result is a matrix

        # Change the final port
        final_port = EnsembleInputPort.global_inhibition

    # Construct the new transmission parameters
    if isinstance(in_transmission_parameters,
                  EnsembleTransmissionParameters):
        transmission_params = EnsembleTransmissionParameters(
            in_transmission_parameters.untransformed_decoders,
            new_transform, in_transmission_parameters.learning_rule
        )
    elif isinstance(in_transmission_parameters,
                    NodeTransmissionParameters):
        transmission_params = NodeTransmissionParameters(
            in_transmission_parameters.pre_slice,
            in_transmission_parameters.function,
            new_transform
        )
    elif isinstance(in_transmission_parameters,
                    PassthroughNodeTransmissionParameters):
        transmission_params = PassthroughNodeTransmissionParameters(
            new_transform
        )
    else:
        raise NotImplementedError

    return transmission_params, final_port


def _combine_reception_params(in_reception_parameters,
                              out_reception_parameters):
    """Combine reception parameters to join two signals into one, e.g., for
    optimising out a passthrough Node.
    """
    # Construct the new reception parameters
    # Combine the filters
    filter_in = in_reception_parameters.filter
    filter_out = out_reception_parameters.filter

    if (filter_in is None or filter_out is None):
        # If either filter is None then just use the filter from the other
        # connection
        new_filter = filter_in or filter_out
    elif (isinstance(filter_in, nengo.LinearFilter) and
            isinstance(filter_out, nengo.LinearFilter)):
        # Both filters are linear filters, so multiply the numerators and
        # denominators together to get a new linear filter.
        new_num = np.polymul(filter_in.num, filter_out.num)
        new_den = np.polymul(filter_in.den, filter_out.den)

        new_filter = nengo.LinearFilter(new_num, new_den)
    else:
        raise NotImplementedError

    # Take the size in from the second reception parameter, construct the new
    # reception parameters.
    return ReceptionParameters(new_filter, out_reception_parameters.width,
                               out_reception_parameters.learning_rule)
