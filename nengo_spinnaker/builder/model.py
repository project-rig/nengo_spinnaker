"""Objects used to represent Nengo networks as instantiated on SpiNNaker.
"""
from collections import namedtuple, defaultdict
import enum
from .ports import EnsembleInputPort
from six import iteritems, itervalues, iterkeys


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


class OutputPort(enum.Enum):
    """Indicate the intended transmitting part of an executable."""
    standard = 0
    """Standard, value-based, output port."""


class InputPort(enum.Enum):
    """Indicate the intended receiving part of an executable."""
    standard = 0
    """Standard, value-based, output port."""


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


ReceptionParameters = namedtuple("ReceptionParameters",
                                 "filter, width, learning_rule")
"""Basic reception parameters that relate to the reception of a series of
multicast packets.

Attributes
----------
filter : :py:class:`~nengo.synapses.Synapse`
    Synaptic filter which should be applied to received values.
width : int
    Width of the post object
"""


class _ParsSinksPair(namedtuple("_PSP", "parameters, sinks")):
    """Pair of transmission parameters and sink tuples."""
    def __new__(cls, signal_parameters, sinks=list()):
        # Copy the sinks list before calling __new__
        sinks = list(sinks)
        return super(_ParsSinksPair, cls).__new__(cls, signal_parameters,
                                                  sinks)


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


def remove_sinkless_signals(conn_map):
    """Remove any signals which do not have any sinks from a connection map.

    Parameters
    ----------
    conn_map : :py:class:`~.ConnectionMap`
        A connection map to modify.
    """
    for port_and_signals in itervalues(conn_map._connections):
        # Prepare to remove any ports which don't connect to anything
        remove_ports = list()

        # Remove any parameter: sinks mappings where there are no sinks
        for port, params_and_signals in iteritems(port_and_signals):
            to_remove = [p for p, s in iteritems(params_and_signals) if
                         len(s) == 0]

            for r in to_remove:
                params_and_signals.pop(r)

            # If there is now nothing coming from this port then mark the port
            # for removal.
            if len(params_and_signals) == 0:
                remove_ports.append(port)

        # Remove any port: {parameters: sinks} where {parameters: sinks} is
        # empty.
        for port in remove_ports:
            port_and_signals.pop(port)


def remove_sinkless_objects(conn_map, cls):
    """Remove all objects of a given type which have no outgoing connections
    from a connection map.

    Parameters
    ----------
    conn_map : :py:class:`~.ConnectionMap`
        A connection map to modify.
    cls :
        Type of objects to remove.

    Returns
    -------
    set
        Set of all objects removed from the connection map.
    """
    # Begin by removing all sinkless signals
    remove_sinkless_signals(conn_map)

    # Find all of the objects of the given type that do not have any outgoing
    # connections.
    transmitting_objects = set()
    receiving_objects = set()

    for source_object, source_port_and_signals in \
            iteritems(conn_map._connections):
        # Store all the sinks for connections out of this object
        has_sinks = False  # Count of objects receiving from this source
        for signals in itervalues(source_port_and_signals):
            for sinks in itervalues(signals):
                for sink_object, _, _ in sinks:
                    has_sinks = True

                    # Store each object in the sinks
                    if isinstance(sink_object, cls):
                        receiving_objects.add(sink_object)

        # Store this object as transmitting a value
        if has_sinks and isinstance(source_object, cls):
            transmitting_objects.add(source_object)

    # Identify all objects which receive but do not transmit
    remove_objects = receiving_objects - transmitting_objects

    # If there are no objects to remove then return the empty set
    if len(remove_objects) == 0:
        return set()

    # Remove target objects from the source of connections
    for o in remove_objects:
        conn_map._connections.pop(o, None)

    # Remove target objects from the sinks of connections
    for sp_and_signals in itervalues(conn_map._connections):
        for signals in itervalues(sp_and_signals):
            for pars, sinks in iteritems(signals):
                # Construct a new sinks list
                signals[pars] = [sp for sp in sinks if
                                 sp.sink_object not in remove_objects]

    # Recurse to remove any objects which we've just made sinkless
    return remove_objects | remove_sinkless_objects(conn_map, cls)
