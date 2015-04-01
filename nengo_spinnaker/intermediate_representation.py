"""Intermediate Representation

An intermediate representation is a part of the Nengo/SpiNNaker compile
process.

    Nengo Network -> Intermediate Representation --(Builder)--> Model
"""
import collections
import enum
import itertools
import nengo
from nengo.utils.builder import full_transform
from nengo.utils import numpy as npext
import numpy as np
import six

from .netlist import NetAddress, InputPort, OutputPort
from .utils.dicts import mrolookupdict, registerabledict


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

    object_builders = registerabledict()
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

    source_getters = registerabledict()
    """Callables which are used to determine how to annotate the source of a
    connection.

    Each callable must accept as arguments the connection that the source is
    for and a :py:class:`.IntermediateRepresentation` which it can use to get
    information about other objects in the network.  A callable must return a
    2-tuple of (:py:class:`nengo_spinnaker.netlist.NetAddr`, **kwargs) where
    accepted keys are currently "keyspace", "extra_objects",
    "extra_connections" and "latching".

    For example, the standard source getter which just returns a source
    indicating the source object and the standard output port is implemented
    as::

        @IntermediateRepresentation.source_getters.register(
            nengo.base.NengoObject)
        def get_source_standard(conn, ir_network):
            source_obj = ir_network.object_map[conn.pre_obj]
            source = NetAddr(source_obj, OutputPort.standard)
            return source, {}
    """

    sink_getters = registerabledict()
    """Callables which are used to determine how to annotate the sink of a
    connection.

    Callables must be the same as the form used in
    :py:attr:`.IntermediateRepresention.source_getters`.  For example, the
    standard sink getter which just returns a sink indicating the sink object
    and the standard input port is implemented as::

        @IntermediateRepresentation.sink_getters.register(
            nengo.base.NengoObject)
        def get_sink_standard(conn, ir_network):
            sink_obj = ir_network.object_map[conn.post_obj]
            sink = NetAddr(sink_obj, InputPort.standard)
            return sink, {}
    """

    probe_builders = registerabledict()
    """Callables which are used to modify the intermediate representation to
    account for probes.

    Each callable should accept a probe object, a seed and a representation of
    the current state of the intermediate representation for the network and
    return an intermediate object to represent the probe and a list of new
    objects and new connections.

    For example, the probe builder for probes of Node output is implemented
    as::

        @IntermediateRepresentation.probe_builders.register(nengo.None)
        def get_node_probe(probe, seed, ir_network):
            # Add a new object for the probe and a new connection from the
            # node to the probe.
            ir_probe = IntermediateObject(probe, seed)

            source = NetAddress(irn.object_map[probe.target],
                                OutputPort.standard)
            sink = NetAddress(ir_probe, InputPort.standard)
            net = IntermediateNet(seed, source, sink, None, False)

            # Return the probe object, no other objects and the new connection
            return ir_probe, [], [net]
    """

    def __new__(cls, obj_map, conn_map, extra_objs, extra_conns):
        return super(IntermediateRepresentation, cls).__new__(
            cls, dict(obj_map), dict(conn_map),
            list(extra_objs), list(extra_conns)
        )

    @classmethod
    def from_objs_conns_probes(cls, objs, conns, probes,
                               extra_object_builders={},
                               extra_source_getters={},
                               extra_sink_getters={},
                               extra_probe_builders={}):
        """Create a new intermediate representation from a list of objects,
        connections and probes.

        Returns
        -------
        :py:class:`.IntermediateRepresentation`
            Intermediate representation of the objects and connections.
        """
        # Update the builders with any extras we've been given
        object_builders = mrolookupdict(cls.object_builders)
        object_builders.update(extra_object_builders)

        source_getters = mrolookupdict(cls.source_getters)
        source_getters.update(extra_source_getters)

        sink_getters = mrolookupdict(cls.sink_getters)
        sink_getters.update(extra_sink_getters)

        probe_builders = mrolookupdict(cls.probe_builders)
        probe_builders.update(extra_probe_builders)

        # For each of the objects generate the appropriate type of intermediate
        # representation, if None then we remove the object from the
        # intermediate representation.
        obj_map = dict()
        for obj in objs:
            replaced_obj = _get_intermediate_object(object_builders, obj)

            if replaced_obj is not None:
                obj_map[obj] = replaced_obj

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
            net, eobjs, econns = _get_intermediate_net(
                source_getters, sink_getters, conn, irn)

            # If the returned Net was None then skip to the next connection
            if net is None:
                continue

            # Add the objects
            conn_map[conn] = net
            extra_objs += eobjs
            extra_conns += econns

        # For each of the probes either add an extra object and connection or
        # modify an existing intermediate representation.
        for probe in probes:
            # Current status of intermediate representation
            irn = cls(obj_map, conn_map, extra_objs, extra_conns)

            # Call the appropriate builder and add any additional components it
            # requires.
            p_obj, eobjs, econns = _get_intermediate_probe(
                probe_builders, probe, irn)

            obj_map[probe] = p_obj
            extra_objs += eobjs
            extra_conns += econns

        # Return a new instance of the namedtuple with the built intermediate
        # representation.
        return cls(obj_map, conn_map, extra_objs, extra_conns)

    def get_nets_starting_at(self, obj):
        """Return all nets which begin at a given intermediate representation
        object.

        Parameters
        ----------
        obj : :py:class:`.IntermediateObject`

        Returns
        -------
        {:py:class:`nengo_spinnaker.netlist.OutputPort`:
                {:py:class:`.IntermediateNet`: :py:class:`nengo.Connection`,
                 ...}, ...}
            Mapping of port to a dictionary mapping nets to the Nengo
            Connections (if available, otherwise None) which they represent.
        """
        return self._filter_nets(lambda n: n.source.object is obj,
                                 lambda n: n.source.port)

    def get_nets_ending_at(self, obj):
        """Return all nets which terminate at a given intermediate
        representation object.

        Parameters
        ----------
        obj : :py:class:`.IntermediateObject`

        Returns
        -------
        {:py:class:`nengo_spinnaker.netlist.InputPort`:
                {:py:class:`.IntermediateNet`: :py:class:`nengo.Connection`,
                 ...}, ...}
            Mapping of port to a dictionary mapping nets to the Nengo
            Connections (if available, otherwise None) which they represent.
        """
        return self._filter_nets(lambda n: n.sink.object is obj,
                                 lambda n: n.sink.port)

    def _filter_nets(self, f, key):
        """Filter nets and construct a dictionary with a custom key the entries
        of which map nets to the Nengo connections they represent (or None).

        Returns
        -------
        {:py:class:`nengo_spinnaker.netlist.OutputPort`:
                {:py:class:`.IntermediateNet`: :py:class:`nengo.Connection`,
                 ...}, ...}
            Mapping of port to a dictionary mapping nets to the Nengo
            Connections (if available, otherwise None) which they represent.
        """
        nets = collections.defaultdict(dict)

        # Go through the two sets of nets and pick out the ones we care about
        for (conn, net) in itertools.chain(
                six.iteritems(self.connection_map),
                ((None, n) for n in self.extra_connections)):
            if f(net):
                nets[key(net)][net] = conn

        return nets


class IntermediateNet(
        collections.namedtuple("IntermediateNet",
                               "seed source sink keyspace latching")):
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
    latching : bool
        Indicates that the receiving buffer must *not* be reset every
        simulation timestep but must hold its value until it next receives a
        packet.
    """
    # TODO At some point (when neuron->neuron connections become possible) nets
    # will need to support multiple sinks.


@IntermediateRepresentation.object_builders.register(nengo.base.NengoObject)
class IntermediateObject(object):
    """Thin wrapper for objects which exist in a Nengo network and need
    representing in an intermediate form.

    Attributes
    ----------
    seed : int
        Seed for any random state used by the object.
    constraints : [IntermediateConstraint, ...]
        List of intermediate constraints that should be applied the vertex
        built from this intermediate representation.
    """
    __slots__ = ["seed", "constraints"]

    def __init__(self, obj, seed, constraints=list()):
        """Create a new intermediate representation for the given object.
        """
        self.seed = seed
        self.constraints = constraints[:]


@IntermediateRepresentation.source_getters.register(nengo.base.NengoObject)
def get_source_standard(conn, irn):
    """Return a standard source for objects which have no special handler.

    Parameters
    ----------
    conn : :py:class:`nengo.Connection`
    irn : :py:class:`.IntermediateRepresentation`
    """
    return (NetAddress(irn.object_map[conn.pre_obj], OutputPort.standard), {})


@IntermediateRepresentation.sink_getters.register(nengo.base.NengoObject)
def get_sink_standard(conn, irn):
    """Return a standard sink for objects which have no special handler.

    Parameters
    ----------
    conn : :py:class:`nengo.Connection`
    irn : :py:class:`.IntermediateRepresentation`
    """
    return (NetAddress(irn.object_map[conn.post_obj], InputPort.standard), {})


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
        return None, {}  # No connection should be made

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
                           InputPort.neurons), {})
    elif (conn.transform.ndim > 0 and
            np.all(conn.transform == conn.transform[0])):
        # This is a global inhibition connection and can be optimised
        return (NetAddress(irn.object_map[conn.post_obj.ensemble],
                           InputPort.global_inhibition), {})
    raise NotImplementedError


@IntermediateRepresentation.probe_builders.register(nengo.Node)
def get_output_probe(probe, seed, irn):
    """Add the probe for the output of a Node or Ensemble.

    Parameters
    ----------
    probe : :py:class:`nengo.Probe`
    seed : int
    irn : :py:class:`.IntermediateRepresentation`
    """
    # Build an intermediate object for the probe then add a new connection from
    # the target to the probe.
    ir_probe = IntermediateObject(probe, seed)

    source = NetAddress(irn.object_map[probe.target], OutputPort.standard)
    sink = NetAddress(ir_probe, InputPort.standard)
    net = IntermediateNet(seed, source, sink, None, False)

    # Return the probe object, no other objects and the new connection
    return ir_probe, [], [net]


@IntermediateRepresentation.probe_builders.register(nengo.Ensemble)
def get_ensemble_probe(probe, seed, irn):
    """Add the probe for an Ensemble.

    Parameters
    ----------
    probe : :py:class:`nengo.Probe`
    seed : int
    irn : :py:class:`.IntermediateRepresentation`
    """
    if probe.attr == "decoded_output":
        return get_output_probe(probe, seed, irn)
    else:
        raise NotImplementedError(probe)


@IntermediateRepresentation.probe_builders.register(nengo.ensemble.Neurons)
def get_neurons_probe(probe, seed, irn):
    """Add the probe for a set of Neurons.

    Parameters
    ----------
    probe : :py:class:`nengo.Probe`
    seed : int
    irn : :py:class:`.IntermediateRepresentation`
    """
    # Add the probe to the intermediate representation for the targeted
    # ensemble
    irn.object_map[probe.target.ensemble].local_probes.append(probe)

    # Return no extra objects or connections
    return None, [], []


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
    """Endpoint requested from endpoint getter.

    Values are of the form:
    (f : connection -> pre/post class, "name", "verb")
    """
    source = (lambda c: c.pre_obj.__class__, "source", "originating")
    sink = (lambda c: c.post_obj.__class__, "sink", "terminating")


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
    def _get_intermediate_endpoint(endpoint, getters, connection, irn):
        (cls, name, verb) = endpoint.value
        try:
            builder = getters[cls(connection)]
        except KeyError:
            raise TypeError(
                "Could not determine the {} for connections {} at object of  "
                "type {}".format(name, verb, cls(connection).__name__)
            )

        # Get the endpoint
        return builder(connection, irn)

    # Get or generate a seed for the connection
    seed = (np.random.randint(npext.maxint)
            if getattr(connection, "seed", None) is None else connection.seed)

    # Get the source for the connection
    source, source_extras = _get_intermediate_endpoint(
        _EndpointType.source, source_getters, connection, irn)

    # If no source is specified then we abort the connection
    if source is None:
        return None, [], []

    # Get the sink for the connection
    sink, sink_extras = _get_intermediate_endpoint(
        _EndpointType.sink, sink_getters, connection, irn)

    # If no sink is specified then we abort the connection
    if sink is None:
        return None, [], []

    # Resolve the keyspaces, allow either end to require a keyspace: if both
    # ends require keyspaces then we fail.
    source_ks = source_extras.pop("keyspace", None)
    sink_ks = sink_extras.pop("keyspace", None)
    if source_ks is None and sink_ks is None:
        ks = None
    elif source_ks is not None and sink_ks is None:
        ks = source_ks
    elif source_ks is None and sink_ks is not None:
        ks = sink_ks
    else:
        raise NotImplementedError("Cannot merge two keyspaces")

    # Determine whether the net should be latching (not by default).  There
    # shouldn't be any case where there is a mismatch between the source and
    # sink in this regard, or where a mismatch would result in incorrect
    # behaviour.
    latching = (source_extras.pop("latching", False) or
                sink_extras.pop("latching", False))

    # Combine the sets of extra objects and connections requested by the sink
    # and sources.
    extra_objs = (source_extras.pop("extra_objects", list()) +
                  sink_extras.pop("extra_objects", list()))
    extra_conns = (source_extras.pop("extra_connections", list()) +
                   sink_extras.pop("extra_connections", list()))

    # Complain if there were any keywords that we didn't understand.
    for key in itertools.chain(*[six.iterkeys(s) for s in
                                 [source_extras, sink_extras]]):
        raise NotImplementedError(
            "Unrecognised source/sink parameter {}".format(key)
        )

    # Build the new net
    return (IntermediateNet(seed, source, sink, ks, latching),
            extra_objs, extra_conns)


def _get_intermediate_probe(builders, probe, irn):
    """Get the seed for a probe and call the appropriate intermediate
    representation builder.

    Parameters
    ----------
    builders : {type: callable, ...}
    probe : :py:class:`nengo.Probe`
        Probe to build the intermediate representation for.
    irn : :py:class:`.IntermediateRepresentation`
    """
    # Get or generate a seed for the probe
    seed = (np.random.randint(npext.maxint)
            if getattr(probe, "seed", None) is None else probe.seed)

    # Get the build function
    try:
        builder = builders[probe.target.__class__]
    except KeyError:
        raise TypeError(
            "Could not determine how to deal with probe with target of type "
            "{}".format(probe.target.__class__.__name__)
        )

    # Call the builder function
    return builder(probe, seed, irn)
