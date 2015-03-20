"""Intermediate Representation

An intermediate representation is a part of the Nengo/SpiNNaker compile
process.

    Nengo Network -> Intermediate Representation --(Builder)--> Model
"""
import collections
import enum
import nengo
from nengo.utils.builder import full_transform
from nengo.utils import numpy as npext
import numpy as np

from .netlist import NetAddress, InputPort, OutputPort
from .utils.mro_dict import MRODict


class IntermediateRepresentation(
        collections.namedtuple("IntermediateRepresentation",
                               ["object_map", "connection_map",
                                "extra_objects", "extra_connections"])):
    """An intermediate representation is an annotation of a Nengo network which
    is more easily mapped to the mix of applications and connections present on
    the SpiNNaker system.

    Attributes
    ----------
    object_map : {:py:class:`nengo.base.NengoObject`:
                    :py:class:`.IntermediateObject`, ...}
        Map from objects in the original Nengo network to intermediate objects
        which may contain extra data used to simulate the object on SpiNNaker.
    connection_map : {:py:class:`nengo.Connection`:
                        :py:class:`.IntermediateNet`, ...}
        Map from connections in the original Nengo network to nets which
        represent the flow of packets through the SpiNNaker system.  Some
        connections may be rerouted, for example modulatory connections.
    extra_objects : list
        List of extra objects that have been inserted into the network and may
        have a tenuous link to objects that existed in the Nengo network.
    extra_connections : list
        List of extra connections that have been inserted into the network,
        such as the connection from an Ensemble to a Probe.
    """

    object_builders = MRODict()
    """Callables which can construct appropriate intermediate representation
    annotations for different kinds of objects.

    Each callable must accept as arguments the object to construct an
    intermediate representation for and the seed associated with the object.
    They must return an object (or None) to use as the intermediate
    representation for the original object.

    The `register` decorator may be used to add a callable to this dictionary.
    For example, the default build action is defined as::

        @IntermediateRepresentation.object_builders.register(
            nengo.base.NengoObject)
        class IntermediateObject(object):
            def __init__(self, obj, seed):
                self.seed = seed

    One could remove all instances of `DeletedNode` from the model with::

        @IntermediateRepresentation.object_builders.register(DeletedNode)
        def build_deleted_node(node):
            return None  # Returning `None` removes the object
    """

    source_getters = MRODict()
    """Callables which are used to determine how to annotate the source of a
    connection.

    Each callable must accept as arguments the connection that the source is
    for and a :py:class:`.IntermediateRepresentation` which it can use to get
    information about other objects in the network.  A callable must return a
    4-tuple of (:py:class:`nengo_spinnaker.netlist.NetAddr`, keyspace or None,
    list of extra objects to add, list of extra connections to add).

    For example, the standard source getter which just returns a source
    indicating the source object and the standard output port is implemented
    as::

        @IntermediateRepresentation.source_getters.register(
            nengo.base.NengoObject)
        def get_source_standard(conn, ir_network):
            source_obj = ir_network.object_map[conn.pre_obj]
            source = NetAddr(source_obj, OutputPort.standard)
            return (source, None, None, None)
    """

    sink_getters = MRODict()
    """Callables which are used to determine how to annotate the sink of a
    connection.

    Each callable must accept as arguments the connection that the sink is
    for and a :py:class:`.IntermediateRepresentation` which it can use to get
    information about other objects in the network.  A callable must return a
    4-tuple of (:py:class:`nengo_spinnaker.netlist.NetAddr`, keyspace or None,
    list of extra objects to add, list of extra connections to add).

    For example, the standard sink getter which just returns a sink
    indicating the sink object and the standard input port is implemented as::

        @IntermediateRepresentation.sink_getters.register(
            nengo.base.NengoObject)
        def get_sink_standard(conn, ir_network):
            sink_obj = ir_network.object_map[conn.post_obj]
            sink = NetAddr(sink_obj, InputPort.standard)
            return (sink, None, None, None)
    """

    @classmethod
    def from_objs_conns_probes(cls, objs, conns, probes):
        """Create a new intermediate representation from a list of objects,
        connections and probes.

        Returns
        -------
        :py:class:`.IntermediateRepresentation`
            Intermediate representation of the objects and connections.
        """
        # For each of the objects generate the appropriate type of intermediate
        # representation.
        obj_map = {obj: _get_intermediate_object(cls.object_builders, obj)
                   for obj in objs}

        conn_map = dict()  # Map from Connections to annotations
        extra_objs = list()  # Extra objects
        extra_conns = list()  # Extra connections

        # For each of the connections generate the appropriate types of
        # intermediate representation or modify an existing intermediate
        # representation object.
        for conn in conns:
            # Current status of intermediate representation
            irn = cls(obj_map, conn_map, extra_objs, extra_conns)

            # Get the net and any extras
            net, _, _ = _get_intermediate_net(
                cls.source_getters, cls.sink_getters, conn, irn)

            # Add the objects
            conn_map[conn] = net

        # For each of the probes either add an extra object and connection or
        # modify an existing intermediate representation.
        for probe in probes:
            if isinstance(probe.target, nengo.ensemble.Neurons):
                # Probe of Neuron objects should just modify the intermediate
                # representation for the Ensemble.
                obj_map[probe.target.ensemble].local_probes.append(probe)

        # Return a new instance of the namedtuple with the built intermediate
        # representation.
        return cls(obj_map, conn_map, extra_objs, extra_conns)


class IntermediateNet(collections.namedtuple("IntermediateNet",
                                             "seed source sink keyspace")):
    """Intermediate representation of a Nengo Connection.

    Attributes
    ----------
    seed : int
        Seed used for random number generation for the net.
    source : :py:class:`~nengo_spinnaker.netlist.NetAddress`
        Source of packets to be transmitted across the net.
    sink : :py:class:`~nengo_spinnaker.netlist.NetAddress`
        Target of packets transmitted across the net.
    keyspace : :py:class:`rig.bitfield.BitField` or None
        Keyspace used to route packets across the network.
    """


@IntermediateRepresentation.object_builders.register(nengo.base.NengoObject)
class IntermediateObject(object):
    """Thin wrapper for objects which exist in a Nengo network and need
    representing in an intermediate form.

    Attributes
    ----------
    seed : int
        Seed for any random state used by the object.
    """
    __slots__ = ["seed"]

    def __init__(self, obj, seed):
        """Create a new intermediate representation for the given object.
        """
        self.seed = seed


@IntermediateRepresentation.source_getters.register(nengo.base.NengoObject)
def get_source_standard(conn, irn):
    """Return a standard source for objects which have no special handler.

    Parameters
    ----------
    conn : :py:class:`nengo.Connection`
    irn : :py:class:`.IntermediateRepresentation`
    """
    return (NetAddress(irn.object_map[conn.pre_obj], OutputPort.standard),
            None, None, None)


@IntermediateRepresentation.sink_getters.register(nengo.base.NengoObject)
def get_sink_standard(conn, irn):
    """Return a standard sink for objects which have no special handler.

    Parameters
    ----------
    conn : :py:class:`nengo.Connection`
    irn : :py:class:`.IntermediateRepresentation`
    """
    return (NetAddress(irn.object_map[conn.post_obj], InputPort.standard),
            None, None, None)


@IntermediateRepresentation.object_builders.register(nengo.Ensemble)
class IntermediateEnsemble(IntermediateObject):
    """Thin wrapper around an ensemble that can store input from
    constant-valued Nodes and indicate which voltage/spike probes are present.

    Attributes
    ----------
    seed : int
        Seed for any random state used by the object.
    direct_input : nd_array
        NumPy array (the same shape as the Ensemble's input) which can store
        input from constant-valued Nodes.
    local_probes : [:py:class:`nengo.Probe`, ...]
        Probes which store data local to the Ensemble (e.g., voltage or
        spikes).
    """
    __slots__ = ["direct_input", "local_probes"]

    def __init__(self, ensemble, seed):
        """Create an intermediate representation for an Ensemble."""
        super(IntermediateEnsemble, self).__init__(ensemble, seed)

        # Create a holder for direct inputs and a list of local probes
        self.direct_input = np.zeros(ensemble.size_in)
        self.local_probes = list()


@IntermediateRepresentation.sink_getters.register(nengo.Ensemble)
def get_ensemble_sink(conn, irn):
    """Get the sink object for a connection into an Ensemble.

    Parameters
    ----------
    conn : :py:class:`nengo.Connection`
    irn : :py:class:`.IntermediateRepresentation`
    """
    if (isinstance(conn.pre_obj, nengo.Node) and
            not isinstance(conn.pre_obj.output, collections.Callable)):
        # We can optimise out connections from constant values Nodes by
        # eventually including their contributions in bias currents, we do this
        # by annotating the intermediate representation and refusing to accept
        # the connection.
        if conn.function is None:
            val = conn.pre_obj.output[conn.pre_slice]
        else:
            val = conn.function(conn.pre_obj.output[conn.pre_slice])

        irn.object_map[conn.post_obj].direct_input += np.dot(
            full_transform(conn, slice_pre=False), val)
        return None, None, None, None  # No connection should be made

    # Otherwise connecting to an Ensemble is just like connecting to anything
    # else.
    return get_sink_standard(conn, irn)


@IntermediateRepresentation.sink_getters.register(nengo.ensemble.Neurons)
def get_neurons_sink(conn, irn):
    """Get the sink object for a connection into an Ensemble's neurons.

    Parameters
    ----------
    conn : :py:class:`nengo.Connection`
    irn : :py:class:`.IntermediateRepresentation`
    """
    if isinstance(conn.pre_obj, nengo.ensemble.Neurons):
        # Neurons -> Neurons connection
        return (NetAddress(irn.object_map[conn.post_obj.ensemble],
                           InputPort.neurons), None, None, None)
    elif (conn.transform.ndim > 0 and
            np.all(conn.transform == conn.transform[0])):
        # This is a global inhibition connection and can be optimised
        return (NetAddress(irn.object_map[conn.post_obj.ensemble],
                           InputPort.global_inhibition), None, None, None)
    raise NotImplementedError


# Helper functions below this point
# ---------------------------------
# Used internally to allow easier testing of
# `IntermediateRepresentation.from_objs_conns_probes`.

def _get_intermediate_object(builders, obj):
    """Get the seed for the intermediate object and call the appropriate
    builder.

    Parameters
    ----------
    builders : {type: callable, ...}
    obj : object
    """
    # Get or generate a seed for the object
    seed = (np.random.randint(npext.maxint)
            if getattr(obj, 'seed', None) is None else obj.seed)

    # Call the appropriate builder
    try:
        builder = builders[obj.__class__]
    except KeyError:
        raise TypeError(
            "Could not construct intermediate representation for "
            "object of type {}".format(obj.__class__)
        )
    return builder(obj, seed)


class _EndpointType(enum.Enum):
    """Endpoint requested from endpoint getter."""
    source = 0
    sink = 1


def _get_intermediate_endpoint(endpoint, getters, connection, irn):
    """Get the endpoint for an intermediate representation for a connection.

    Parameters
    ----------
    endpoint : :py:class:`._EndpointType`
        Type of the endpoint to extract (pre or post).
    getters : {type: callable, ...}
    connection : :py:class:`nengo.Connection`
    irn : :py:class:`~.IntermediateRepresentation`
    """
    if endpoint is _EndpointType.source:
        try:
            builder = getters[connection.pre_obj.__class__]
        except KeyError:
            raise TypeError(
                "Could not determine the source for connections "
                "originating at object of type {}".format(
                    connection.pre_obj.__class__.__name__)
            )
    elif endpoint is _EndpointType.sink:
        try:
            builder = getters[connection.post_obj.__class__]
        except KeyError:
            raise TypeError(
                "Could not determine the sink for connections "
                "terminating at object of type {}".format(
                    connection.post_obj.__class__.__name__)
            )
    else:
        raise ValueError(endpoint)

    # Get the endpoint
    return builder(connection, irn)


def _get_intermediate_net(source_getters, sink_getters, connection, irn):
    """Create an intermediate representation for a net.

    Parameters
    ----------
    source_getters : {type: callable, ...}
    sink_getters : {type: callable, ...}
    connection : :py:class:`nengo.Connection`
    irn : :py:class:`~.IntermediateRepresentation`

    Returns
    -------
    (:py:class:`.IntermediateNet`, extra objects, extra connections)
        The intermediate net and any extra objects or connections that were
        deemed necessary.
    """
    # Get or generate a seed for the connection
    seed = (np.random.randint(npext.maxint)
            if getattr(connection, "seed", None) is None else connection.seed)

    # Get the source for the connection
    (source, source_ks, source_objs, source_conns) = \
        _get_intermediate_endpoint(
            _EndpointType.source, source_getters, connection, irn)

    # If no source is specified then we abort the connection
    if source is None:
        return None, [], []

    # Get the sink for the connection
    (sink, sink_ks, sink_objs, sink_conns) = \
        _get_intermediate_endpoint(
            _EndpointType.sink, sink_getters, connection, irn)

    # If no sink is specified then we abort the connection
    if sink is None:
        return None, [], []

    # Resolve the keyspaces
    if source_ks is None and sink_ks is None:
        ks = None
    elif source_ks is not None and sink_ks is None:
        ks = source_ks
    elif source_ks is None and sink_ks is not None:
        ks = sink_ks
    else:
        raise NotImplementedError("Cannot merge two keyspaces")

    # Build the new net
    return IntermediateNet(seed, source, sink, ks), [], []
