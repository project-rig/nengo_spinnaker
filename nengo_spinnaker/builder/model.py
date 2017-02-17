"""Objects used to represent Nengo networks as instantiated on SpiNNaker.
"""
from copy import copy
from collections import namedtuple, defaultdict, deque
from nengo import LinearFilter
import numpy as np
from .ports import EnsembleInputPort, InputPort, OutputPort
from six import iteritems, itervalues, iterkeys

from nengo_spinnaker.operators.filter import Filter


class ConnectionMap(object):
    """A container which represents all of the connections in a model and the
    parameters which are associated with them.

    A `ConnectionMap` maps from source objects to their outgoing ports to lists
    of transmission parameters and the objects which receive packets. This may
    be best expressed as::

        {source_object:
            {source_port: {
                (signal_parameters, transmit_parameters):
                 [(sink_object, sink_port, reception_parameters), ...],
                ...},
            ...},
        ...}

    Alternatively: A nengo_spinnaker object can have multiple outgoing "ports".
    Each port can transmit multiple outgoing signals, which are classified by
    both how they are represented on SpiNNaker (their "signal_parameters") and
    by what they represent in Nengo terms (their "transmit_parameters"). Each
    signal can target multiple "sinks". Each sink is a combination of a
    nengo_spinnaker object, a receiving port in that object and some
    description of what to do with the signal when it is received. As there are
    multiple objects this data structure is repeated for each.

                  /-> Port ---> Signal --> Sink
                 /         \--> Signal --> Sink
                /
        Object -
                \
                 \                     /-> Sink
                  \-> Port --> Signal ---> Sink
                                       \-> Sink

        Object -----> Port --> Signal ---> Sink
    """
    def __init__(self):
        """Create a new empty connection map."""
        # Construct the connection map internal structure
        self._connections = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )

    def add_connection(self, source_object, source_port, signal_parameters,
                       transmission_parameters, sink_object, sink_port,
                       reception_parameters):
        """Add a new connection to the map of connections.

        Parameters
        ----------
        signal_parameters : :py:class:`~.SignalParameters`
            Parameters describing how the signal will be transmitted (keyspace,
            weight, etc.).
        transmission_parameters :
            Source-specific parameters of how the signal is to be created
            (e.g., decoders, transform, etc.).
        reception_parameters : :py:class:`~.ReceptionParameters`
            Sink-specific parameters of how the received packets are to be
            treated.
        """
        # Swap out the connection for a global inhibition connection if
        # possible.
        if (sink_port is EnsembleInputPort.neurons and
                transmission_parameters.supports_global_inhibition):
            sink_port = EnsembleInputPort.global_inhibition
            transmission_parameters = \
                transmission_parameters.as_global_inhibition_connection

        # Combine the signal parameters with the transmission parameters
        # (These represent the signal and can be hashed)
        pars = (signal_parameters, transmission_parameters)

        # See if we can combine the connection with an existing set of
        # transmission parameters.
        sinks = self._connections[source_object][source_port][pars]
        sinks.append(_SinkPars(sink_object, sink_port, reception_parameters))

    def get_signals_from_object(self, source_object):
        """Get the signals transmitted by a source object.

        Returns
        -------
        {port : [signal_parameters, ...], ...}
            Dictionary mapping ports to lists of parameters for the signals
            that originate from them.
        """
        signals = defaultdict(list)

        # For every port and list of (transmission pars, sinks) associated with
        # it add the transmission parameters to the correct list of signals.
        for port, sigs in iteritems(self._connections[source_object]):
            signals[port] = list(pars for pars in iterkeys(sigs))

        return signals

    def get_signals_to_all_objects(self):
        """Get the signals received by all sink objects.

        Returns
        -------
        {object: {port : [ReceptionSpec, ...], ...}, ...}
            Dictionary mapping objects to mappings from ports to lists of
            objects specifying incoming signals.
        """
        incoming_signals = defaultdict(lambda: defaultdict(list))

        for port_conns in itervalues(self._connections):
            for conns in itervalues(port_conns):
                for (sig_params, _), sinks in iteritems(conns):
                    # For each sink, if the sink object is the specified object
                    # then add signal to the list.
                    for sink in sinks:
                        # This is the desired sink object, so remember the
                        # signal. First construction the reception
                        # specification.
                        incoming_signals[sink.sink_object][sink.port].append(
                            ReceptionSpec(sig_params,
                                          sink.reception_parameters)
                        )

        return incoming_signals

    def get_signals(self):
        """Extract all the signals from the connection map.

        Yields
        ------
        Signal
            Signal objects derived from the contents of the connection map.
        """
        # For each source object and set of sinks yield a new signal
        for source, port_conns in iteritems(self._connections):
            # For each connection look at the sinks and the signal parameters
            for conns in itervalues(port_conns):
                for (sig_pars, transmission_pars), par_sinks in \
                        iteritems(conns):
                    # Create a signal using these parameters
                    yield (Signal(source,
                                  (ps.sink_object for ps in par_sinks),
                                  sig_pars),
                           transmission_pars)

    def insert_interposers(self):
        """Get a new connection map with the passthrough nodes removed and with
        interposers inserted into the network at appropriate places.

        Returns
        -------
        ([Interposer, ...], ConnectionMap)
            A collection of new interposer operators and a new connection map
            with passthrough nodes removed and interposers introduced.
        """
        # Create a new connection map and a store of interposers
        interposers = list()
        cm = ConnectionMap()

        # For every clique in this connection map we identify which connections
        # to replace with interposers and then insert the modified connectivity
        # into the new connection map.
        for sources, nodes in self.get_cliques():
            # Extract all possible interposers from the clique
            possible_interposers = (
                (node, port, conn, {s.sink_object for s in sinks})
                for node in nodes
                for port, conns in iteritems(
                    self._connections[node])
                for conn, sinks in iteritems(conns)
            )

            # Of these possible connections determine which would benefit from
            # replacement with interposers. For these interposers build a set
            # of other potential interposers whose input depends on the output
            # of the interposer.
            potential_interposers = dict()
            for node, port, conn, sink_objects in possible_interposers:
                _, tps = conn  # Extract the transmission parameters

                # Determine if the interposer connects to anything
                if not self._connects_to_non_passthrough_node(sink_objects):
                    continue

                # For each connection look at the fan-out and fan-in vs the
                # cost of the interposer.
                trans = tps.full_transform(False, False)
                mean_fan_in = np.mean(np.sum(trans != 0.0, axis=0))
                mean_fan_out = np.mean(np.sum(trans != 0.0, axis=1))
                interposer_fan_in = np.ceil(float(trans.shape[1]) / float(128))
                interposer_fan_out = np.ceil(float(trans.shape[0]) / float(64))

                # If the interposer would improve connectivity then add it to
                # the list of potential interposers.
                if (mean_fan_in > interposer_fan_in or
                        mean_fan_out > interposer_fan_out):
                    # Store the potential interposer along with a list of nodes
                    # who receive its output.
                    potential_interposers[(node, port, conn)] = [
                        s for s in sink_objects
                        if isinstance(s, PassthroughNode)
                    ]

            # Get the set of potential interposers whose input is independent
            # of the output of any other interposer.
            top_level_interposers = set(potential_interposers)
            for dependent_interposers in itervalues(potential_interposers):
                # Subtract from the set of independent interposers any whose
                # input node is listed in the output nodes for another
                # interposer.
                remove_interposers = {
                    (node, port, conn) for (node, port, conn) in
                    top_level_interposers if node in dependent_interposers
                }
                top_level_interposers.difference_update(remove_interposers)

            # Create an operator for all of the selected interposers
            clique_interposers = dict()
            for node, port, conn in top_level_interposers:
                # Extract the input size
                _, transmission_pars = conn
                size_in = transmission_pars.size_in

                # Create the interposer
                clique_interposers[node, port, conn] = Filter(size_in)

            # Insert connections into the new connection map inserting
            # connections to interposers as we go and remembering those which a
            # connection is added.
            used_interposers = set()  # Interposers who receive non-zero input
            for source in sources:
                used_interposers.update(
                    self._copy_connections_from_source(
                        source=source, target_map=cm,
                        interposers=clique_interposers
                    )
                )

            # Insert connections from the new interposers.
            for (node, port, conn) in used_interposers:
                # Get the interposer, add it to the new operators to include in
                # the model and add its output to the new connection map.
                interposer = clique_interposers[(node, port, conn)]
                interposers.append(interposer)

                # Add outgoing connections
                self._copy_connections_from_interposer(
                    node=node, port=port, conn=conn, interposer=interposer,
                    target_map=cm
                )

        return interposers, cm

    def _copy_connections_from_source(self, source, target_map,
                                      interposers=dict()):
        """Copy the pattern of connectivity from a source object, inserting the
        determined connectivity into a new connection map instance and
        inserting connections to interposers as appropriate.

        Parameters
        ----------
        source : object
            Object in the connection map whose outgoing connections should be
            copied into the new connection map.
        target_map: :py:class:`~.ConnectionMap`
            Connection map into which the new connections should be copied.
        interposers : {(PassthroughNode, port, conn): Filter, ...}
            Dictionary mapping selected nodes, ports and connections to the
            operators which will be used to simulate them.

        Returns
        -------
        {(PassthroughNode, port, conn), ...}
            Set of interposers who were reached by connections from this
            object.
        """
        used_interposers = set()  # Interposers fed by this object

        # For every port and set of connections originating at the source
        for source_port, conns_sinks in iteritems(self._connections[source]):
            for conn, sinks in iteritems(conns_sinks):
                # Extract the signal and transmission parameters
                signal_parameters, transmission_parameters = conn

                # Copy the connections and mark which interposers are reached.
                for sink in sinks:
                    # NOTE: The None indicates that no additional reception
                    # parameters beyond those in the sink are to be considered.
                    used_interposers.update(self._copy_connection(
                        target_map, interposers, source, source_port,
                        signal_parameters, transmission_parameters,
                        sink.sink_object, sink.port, sink.reception_parameters
                    ))

        return used_interposers

    def _copy_connections_from_interposer(self, node, port, conn, interposer,
                                          target_map):
        """Copy the pattern of connectivity from a node, replacing a specific
        connection with an interposer.
        """
        # Get the sinks of this connection
        sinks = self._connections[node][port][conn]

        # Extract parameters
        signal_pars, transmission_pars = conn

        # Copy the connections to the sinks, note that we specify an empty
        # interposers dictionary so that no connections to further interposers
        # are inserted, likewise we specify no additional reception parameters.
        # The connectivity from the sink will be recursed if it is a
        # passthrough node.
        for sink in sinks:
            self._copy_connection(
                target_map, dict(), interposer, OutputPort.standard,
                signal_pars, transmission_pars,
                sink.sink_object, sink.port, sink.reception_parameters
            )

    def _copy_connection(self, target_map, interposers, source, source_port,
                         signal_pars, transmission_pars,
                         sink_object, sink_port, reception_pars):
        """Copy a single connection from this connection map to another,
        recursing if an appropriate node is identified.

        Parameters
        ----------
        target_map: :py:class:`~.ConnectionMap`
            Connection map into which the new connections should be copied.
        interposers : {(PassthroughNode, port, conn): Filter, ...}
            Dictionary mapping selected nodes, ports and connections to the
            operators which will be used to simulate them.

        All other parameters as in :py:method:`~.ConnectionMap.add_connection`.

        Returns
        -------
        {(PassthroughNode, port, conn), ...}
            Interposers which are reached.
        """
        used_interposers = set()  # Reached interposers

        if not isinstance(sink_object, PassthroughNode):
            # If the sink is not a passthrough node then just add the
            # connection to the new connection map.
            target_map.add_connection(
                source, source_port, copy(signal_pars), transmission_pars,
                sink_object, sink_port, reception_pars)
        else:
            # If the sink is a passthrough node then we consider each outgoing
            # connection in turn. If the connection is to be replaced by an
            # interposer then we add a connection to the relevant interposer,
            # otherwise we recurse to add further new connections.
            for port, conns_sinks in iteritems(self._connections[sink_object]):
                for conn, sinks in iteritems(conns_sinks):
                    # Determine if this combination of (node, port, conn) is to
                    # be replaced by an interposer.
                    interposer = interposers.get((sink_object, port, conn))

                    if interposer is not None:
                        # Insert a connection to the interposer
                        target_map.add_connection(
                            source, source_port, copy(signal_pars),
                            transmission_pars, interposer,
                            InputPort.standard, reception_pars
                        )

                        # Mark the interposer as reached
                        used_interposers.add((sink_object, port, conn))
                    else:
                        # Build the new signal and transmission parameters.
                        (this_signal_pars, this_transmission_pars) = conn
                        sink_signal_pars = signal_pars.concat(this_signal_pars)
                        sink_transmission_pars = transmission_pars.concat(
                            this_transmission_pars)

                        if sink_transmission_pars is not None:
                            # Add onward connections if the transmission
                            # parameters aren't empty.
                            for new_sink in sinks:
                                # Build the reception parameters and recurse
                                sink_reception_pars = reception_pars.concat(
                                    new_sink.reception_parameters
                                )

                                used_interposers.update(self._copy_connection(
                                    target_map, interposers,
                                    source, source_port,
                                    sink_signal_pars, sink_transmission_pars,
                                    new_sink.sink_object, new_sink.port,
                                    sink_reception_pars
                                ))

        return used_interposers

    def _connects_to_non_passthrough_node(self, sink_objects):
        """Determine whether any of the sink objects are not passthrough nodes,
        or if none are whether those passthrough nodes eventually connect to a
        non-passthrough node.
        """
        # Extract passthrough nodes from the sinks
        ptns = [s for s in sink_objects if isinstance(s, PassthroughNode)]

        # If any of the sink objects are not passthrough nodes then return
        if len(ptns) < len(sink_objects):
            return True
        else:
            # Otherwise loop over the connections from each connected
            # passthrough node and see if any of those connect to a sink.
            for obj in ptns:
                for conn_sinks in itervalues(self._connections[obj]):
                    for sinks in itervalues(conn_sinks):
                        sink_objs = [s.sink_object for s in sinks]
                        if self._connects_to_non_passthrough_node(sink_objs):
                            return True

        # Otherwise return false to indicate that a non-passthrough node object
        # is never reached.
        return False

    def get_coarsened_graph(self):
        """Get a coarser representation of the connectivity represented in this
        connection map.

        Returns
        -------
        {obj: Edges(input, output), ...}
            Mapping from objects in the network to tuples containing sets
            representing objects which provide input and receive output.
        """
        graph = defaultdict(Edges)

        for signal, _ in self.get_signals():
            source = signal.source

            for sink in signal.sinks:
                graph[sink].inputs.add(source)
                graph[source].outputs.add(sink)

        return graph

    def get_cliques(self):
        """Extract cliques of connected nodes from the connection map.

        For example, the following network consists of two cliques:

            1 ->-\    /->- 5 ->-\
            2 ->--> 4 -->- 6 ->--> 8 ->- 9
            3 ->-/    \->- 7 ->-/

            \=======v=====/\=======v======/
                Clique 1       Clique 2

        Where 4, 8 and 9 are passthrough nodes.

        Clique 1 has the following sources: {1, 2, 3}
        Clique 2 has the sources: {5, 6, 7}

        Adding a recurrent connection results in there being a single clique:

                    /-<------------------<--\
                    |                       |
            1 ->-\  v /->- 5 ->-\           |
            2 ->--> 4 -->- 6 ->--> 8 ->- 9 -/
            3 ->-/    \->- 7 ->-/

        Where the sources are: {1, 2, 3, 5, 6, 7}

        Yields
        ------
        ({source, ...}, {Node, ...})
            A set of objects which form the inputs to the clique and the set of
            passthrough nodes contained within the clique (possibly empty).
        """
        # Coarsen the connection map
        graph = self.get_coarsened_graph()

        # Construct a set of source objects which haven't been visited
        unvisited_sources = {
            obj for obj, edges in iteritems(graph) if
            len(edges.outputs) > 0 and
            not isinstance(obj, PassthroughNode)
        }

        # While unvisited sources remain inspect the graph.
        while unvisited_sources:
            sources = set()  # Set of objects which feed the clique
            ptns = set()  # Passthrough nodes contained in the clique
            sinks = set()  # Set of objects which receive values

            # Each node that is visited in the following breadth-first search
            # is treated as EITHER a source or a sink node. If the node is a
            # sink then we're interested in connected nodes which provide its
            # input, if it's a source then we care about nodes to which it
            # provides input and if it's a passthrough node then we care about
            # both.
            queue = deque()  # Queue of nodes to visit
            queue.append((True, unvisited_sources.pop()))  # Add a source

            while len(queue) > 0:  # While there remain items in the queue
                is_source, node = queue.pop()  # Get an item from the queue
                queue_sources = queue_sinks = False

                if isinstance(node, PassthroughNode) and node not in ptns:
                    # If the node is a passthrough node then we add it to the
                    # set of passthrough nodes and then add both objects which
                    # feed it and those which receive from it to the queue.
                    ptns.add(node)
                    queue_sources = queue_sinks = True
                elif (not isinstance(node, PassthroughNode) and
                        is_source and node not in sources):
                    # If the node is a source then we add it to the set of
                    # sources for the clique and then add all objects which it
                    # feeds to the queue.
                    sources.add(node)
                    queue_sinks = True
                elif (not isinstance(node, PassthroughNode) and
                        not is_source and node not in sinks):
                    # If the node is a sink then we add it to the set of sinks
                    # for the clique and add all objects which feed it to the
                    # queue.
                    sinks.add(node)
                    queue_sources = True

                # Queue the selected items
                if queue_sources:
                    queue.extend((True, obj) for obj in graph[node].inputs if
                                 obj not in sources and obj not in ptns)
                if queue_sinks:
                    queue.extend((False, obj) for obj in graph[node].outputs if
                                 obj not in sinks and obj not in ptns)

            # Once the queue is empty we yield the contents of the clique
            unvisited_sources.difference_update(sources)
            yield sources, ptns


class SignalParameters(object):
    """Basic parameters that can be applied to a signal.

    Attributes
    ----------
    latching : bool
        If False (the default) any receiving filters should reset their inputs
        every time-step.  If True then receiving filters should hold their
        inputs until new values are received.
    weight : int
        Number of packets expected to be sent every time-step.
    keyspace : :py:class:`~rig.bitfield.BitField` or None
        Keyspace which will be used to assign keys to packets.
    """
    def __init__(self, latching=False, weight=0, keyspace=None):
        # Store the parameters
        self.latching = latching
        self.weight = weight
        self.keyspace = keyspace

    def __hash__(self):
        return hash((self.latching, self.weight))

    def __eq__(self, other):
        # Equivalent if the latching spec is the same, the weight is the same
        # and the keyspaces are equivalent.
        if ((self.latching is other.latching) and
                (self.weight == other.weight) and
                (self.keyspace == other.keyspace)):
            return True
        else:
            return False

    def __ne__(self, b):
        return not self == b

    def concat(a, b):
        """Get new signal parameters as the result of combining this set of
        signal parameters with another.
        """
        # We cannot combine two different keyspaces
        if a.keyspace is not None and b.keyspace is not None:
            raise Exception("Cannot merge keyspaces {!s} and {!s}".format(
                a.keyspace, b.keyspace))

        # Merge the parameters
        return SignalParameters(
            latching=a.latching or b.latching,
            weight=b.weight,
            keyspace=a.keyspace if a.keyspace is not None else b.keyspace
        )


class Edges(namedtuple("Edges", "inputs, outputs")):
    """Edges in a simplified representation of a connection map."""
    def __new__(cls):
        return super(Edges, cls).__new__(cls, set(), set())


class ReceptionParameters(namedtuple("ReceptionParameters",
                          "filter, width, learning_rule")):
    """Basic reception parameters that relate to the reception of a series of
    multicast packets.

    Attributes
    ----------
    filter : :py:class:`~nengo.synapses.Synapse`
        Synaptic filter which should be applied to received values.
    width : int
        Width of the post object
    """
    def concat(self, other):
        """Create new reception parameters by combining this set of reception
        parameters with another.
        """
        # Combine the filters
        if self.filter is None:
            new_filter = other.filter
        elif other.filter is None:
            new_filter = self.filter
        elif (isinstance(self.filter, LinearFilter) and
                isinstance(other.filter, LinearFilter)):
            # Combine linear filters by multiplying their numerators and
            # denominators.
            new_filter = LinearFilter(
                np.polymul(self.filter.num, other.filter.num),
                np.polymul(self.filter.den, other.filter.den)
            )
        else:
            raise NotImplementedError(
                "Cannot combine filters of type {} and {}".format(
                    type(self.filter), type(other.filter)))

        # Combine the learning rules
        if self.learning_rule is not None and other.learning_rule is not None:
            raise NotImplementedError(
                "Cannot combine learning rules {} and {}".format(
                    self.learning_rule, other.learning_rule))

        new_learning_rule = self.learning_rule or other.learning_rule

        # Create the new reception parameters
        return ReceptionParameters(new_filter, other.width, new_learning_rule)


_SinkPars = namedtuple("_SinkPars", ["sink_object", "port",
                       "reception_parameters"])
"""Collection of parameters for a sink."""


ReceptionSpec = namedtuple("ReceptionSpec", ["signal_parameters",
                                             "reception_parameters"])
"""Specification of an incoming connection.

Attributes
----------
signal_parameters : :py:class:`~.SignalParameters`
    Description of how the signal will be transmitted.
reception_parameters : :py:class:`~.ReceptionParameters`
    Object specific description of how the received signal is to be handled
    (e.g., the type of filter to use).
"""


class Signal(object):
    """Details of a stream of multicast packets that will be transmitted across
    the SpiNNaker system.

    Attributes
    ----------
    source :
        Object representing the source of the stream of packets.
    sinks :
        List of objects representing the sinks of the stream of packets.
    keyspace : :py:class:`~rig.bitfield.BitField`
        Keyspace used to derive keys for the packets.
    weight : int
        Number of packets expected to be sent across the packet each time-step.
    """
    def __init__(self, source, sinks, params):
        """Create a new signal."""
        # Store all the parameters, copying the list of sinks.
        self.source = source
        self.sinks = list(sinks)
        self._params = params

    @property
    def keyspace(self):
        return self._params.keyspace

    @keyspace.setter
    def keyspace(self, ks):
        self._params.keyspace = ks

    @property
    def weight(self):
        return self._params.weight

    @property
    def width(self):
        return self.weight


class PassthroughNode(object):
    """A non-computational node which will be optimised out of the model but
    acts as a useful point to which signals can be connected.
    """
    def __init__(self, label=None):
        self._label = label

    def __repr__(self):
        return "PassthroughNode({})".format(self._label)

    def __str__(self):
        return "PassthroughNode({})".format(self._label)
