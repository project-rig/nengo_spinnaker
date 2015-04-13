"""SpiNNaker annotations

The Nengo/SpiNNaker analogue of the standard Nengo model.  This annotation
build process takes an existing Nengo model and performs optimisations to
produce something which can be used to build a set of vertices and nets for
placing and routing for a SpiNNaker machine.
"""
import collections
import enum
import itertools
import nengo
from nengo.utils import numpy as npext
import numpy as np
import six

from .keyspaces import keyspaces
from .utils.collections import (mrolookupdict, noneignoringlist,
                                registerabledict)


class Annotations(collections.namedtuple(
    "Annotations", ["objects", "connections",
                    "extra_objects", "extra_connections"])):
    """An annotation of a Nengo model which is more easily mapped to the mix of
    applications and connections present on the SpiNNaker system.

    Attributes
    ----------
    objects : {:py:class:`nengo.base.NengoObject`:
                   :py:class:`.ObjectAnnotation`, ...}
        Map from objects in the original Nengo network to annotations which may
        contain extra data used to simulate the object on SpiNNaker.
    connections : {:py:class:`nengo.Connection`:
                       :py:class:`.AnnotatedNet`, ...}
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
    """Callables which can construct annotations for different kinds of
    objects.

    Each callable must accept as arguments the object to construct an
    annotation for.  They must return an object (or None) to use as the
    annotation for the original object.

    The `register` decorator may be used to add a callable to this dictionary.
    For example, the default build action is defined as::

        @Annotations.object_builders.register(
            nengo.base.NengoObject)
        class Annotation(object):
            def __init__(self, obj):
                pass
    """

    source_getters = registerabledict()
    """Callables which are used to determine how to annotate the source of a
    connection.

    Each callable must accept as arguments the connection that the source is
    for and a :py:class:`.IntermediateRepresentation` which it can use to get
    information about other objects in the network.  A callable must return a
    :py:class:`.SinkOrSourceSpecification` (see also :py:class:`.soss`).

    In the specific case of `source_getter`s, returning an existing
    `AnnotatedNet` will cause that net to be modified by adding the
    appropriate sink.

    For example, the standard source getter which just returns a source
    indicating the source object and the standard output port is implemented
    as::

        @Annotations.source_getters.register(nengo.base.NengoObject)
        def get_source_standard(conn, annotations):
            source_obj = annotations.objects[conn.pre_obj]
            source = NetAddr(source_obj, OutputPort.standard)
            return soss(source)
    """

    sink_getters = registerabledict()
    """Callables which are used to determine how to annotate the sink of a
    connection.

    Callables must be the same as the form used in
    :py:attr:`.IntermediateRepresention.source_getters`.  For example, the
    standard sink getter which just returns a sink indicating the sink object
    and the standard input port is implemented as::

        @Annotations.sink_getters.register(nengo.base.NengoObject)
        def get_sink_standard(conn, annotations):
            sink_obj = annotations.objects[conn.post_obj]
            sink = NetAddr(sink_obj, InputPort.standard)
            return soss(sink)
    """

    probe_builders = registerabledict()
    """Callables which are used to modify the annotation to account for probes.

    Each callable should accept a probe object and a representation of the
    current state of the annotation for the network and return an annotation
    for the probe and a list of new objects and new connections if necessary.

    For example, the probe builder for probes of Node output is implemented
    as::

        @Annotations.probe_builders.register(nengo.None)
        def get_node_probe(probe, annotations):
            # Add a new object for the probe and a new connection from the
            # node to the probe.
            annotation = ObjectAnnotation(probe)

            # Return the probe object
            return annotation, [], []
    """

    def __new__(cls, objects, connections, extra_objects, extra_connections):
        return super(Annotations, cls).__new__(
            cls, dict(objects), dict(connections),
            list(extra_objects), list(extra_connections)
        )

    @classmethod
    def from_model(cls, model, extra_object_builders={},
                   extra_source_getters={}, extra_sink_getters={},
                   extra_probe_builders={}):
        """Create a new annotation from a built Nengo model.

        Parameters
        ----------
        model : :py:class:`nengo.builder.Model`
            A built Nengo model containing objects to create the annotations
            for.

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

        probe_builders = mrolookupdict(cls.probe_builders)
        probe_builders.update(extra_probe_builders)

        sink_getters = mrolookupdict(cls.sink_getters)
        sink_getters.update(extra_sink_getters)

        # We can split the parameters objects in the Model into 3 dicts
        conns = {c: b for c, b in six.iteritems(model.params) if
                 isinstance(c, nengo.Connection)}
        probes = {p: b for p, b in six.iteritems(model.params) if
                  isinstance(p, nengo.Probe)}
        objs = {o: b for o, b in six.iteritems(model.params) if
                o not in conns and o not in probes}

        # For each of the objects get an annotation.
        obj_map = dict()
        for obj, built_obj in six.iteritems(objs):
            obj_map[obj] = _get_object_annotation(
                object_builders, obj, built_obj)

        conn_map = dict()  # Map from Connections to annotations
        extra_objs = list()  # Extra objects
        extra_conns = list()  # Extra connections

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

        # For each of the connections generate the appropriate types of
        # intermediate representation or modify an existing intermediate
        # representation object.
        for conn in conns:
            # Current status of intermediate representation
            irn = cls(obj_map, conn_map, extra_objs, extra_conns)

            # Get the net and any extras
            net, eobjs, econns = _get_intermediate_net(
                source_getters, sink_getters, conn, irn)

            # Add the objects
            conn_map[conn] = net

            # If the returned Net was None then skip to the next connection
            if net is None:
                continue

            extra_objs += eobjs
            extra_conns += econns

        # Return a new instance of the namedtuple with the built intermediate
        # representation.
        return cls(obj_map, conn_map, extra_objs, extra_conns)

    def get_nets_starting_at(self, obj):
        """Return all nets which begin at a given annotation.

        Parameters
        ----------
        obj : :py:class:`.Annotation`

        Returns
        -------
        {:py:class:`nengo_spinnaker.netlist.OutputPort`:
                {:py:class:`.AnnotatedNet`: [:py:class:`nengo.Connection`],
                 ...}, ...}
            Mapping of port to a dictionary mapping nets to the Nengo
            Connections (if available, otherwise None) which they represent.
        """
        return self._filter_nets(lambda n: n.source.object is obj,
                                 lambda n: n.source.port)

    def get_nets_ending_at(self, obj):
        """Return all nets which terminate at a given annotation.

        Parameters
        ----------
        obj : :py:class:`.Annotation`

        Returns
        -------
        {:py:class:`nengo_spinnaker.netlist.InputPort`:
                {:py:class:`.AnnotatedNet`: [:py:class:`nengo.Connection`],
                 ...}, ...}
            Mapping of port to a dictionary mapping nets to the Nengo
            Connections (if available, otherwise None) which they represent.
        """
        def get_port(net):
            for s in net.sinks:
                if s.object is obj:
                    return s.port
            else:  # pragma : no cover
                assert False  # Should be unreachable

        return self._filter_nets(
            lambda n: any(s.object is obj for s in n.sinks),
            get_port
        )

    def _filter_nets(self, f, key):
        """Filter nets and construct a dictionary with a custom key the entries
        of which map nets to the Nengo connections they represent (or None).

        Returns
        -------
        {:py:class:`nengo_spinnaker.netlist.OutputPort`:
                {:py:class:`.AnnotatedNet`: [:py:class:`nengo.Connection`],
                 ...}, ...}
            Mapping of port to a dictionary mapping nets to the Nengo
            Connections (if available, otherwise None) which they represent.
        """
        nets = collections.defaultdict(
            lambda: collections.defaultdict(noneignoringlist)
        )

        # Go through the two sets of nets and pick out the ones we care about
        for (conn, net) in itertools.chain(
                six.iteritems(self.connections),
                ((None, n) for n in self.extra_connections)):
            if f(net):
                nets[key(net)][net].append(conn)

        return nets

    def apply_default_keyspace(self, keyspace=keyspaces["nengo"]):
        """Apply a default keyspace to all Nets with `None` as their current
        keyspace.

        The default keyspace should have the fields `nengo_object` and
        `nengo_connection`.

        Parameters
        ----------
        keyspace : :py:class:`rig.bitfield.BitField`
        """
        # For each object look up the nets which originate from it and add
        # keyspaces where they do not currently exist.
        nets_req_keyspaces = self._filter_nets(
            lambda x: x.keyspace is None,
            key=lambda x: x.source.object
        )
        for obj_id, nets in enumerate(six.itervalues(nets_req_keyspaces)):
            for net_id, net in enumerate(nets):
                net.keyspace = keyspace(object=obj_id,
                                        connection=net_id)


class SinkOrSourceSpecification(collections.namedtuple(
        "SOSS",
        "target extra_objects extra_nets keyspace latching weight"
        )):
    """Specification for a source or sink as returned by a source or sink
    getter.
    """
    def __new__(cls, source_or_sink, extra_objects=list(),
                extra_nets=list(), keyspace=None, latching=False, weight=None):
        return super(SinkOrSourceSpecification, cls).__new__(
            cls, source_or_sink, list(extra_objects), list(extra_nets),
            keyspace, latching, weight
        )


soss = SinkOrSourceSpecification
"""Quick reference to :py:class:`.SinkOrSourceSpecification`"""


class OutputPort(enum.Enum):
    """Indicate the intended transmitting part of an executable."""
    standard = 0  # Standard, value based transmission

    # Ensembles only
    neurons = 1  # Transmits spike data


class InputPort(enum.Enum):
    """Indicate the intended receiving part of an executable."""
    standard = 0  # Standard, value based transmission

    # Ensembles only
    neurons = 1  # Receives spike data
    global_inhibition = 2  # Receives value-encoded inhibition data


NetAddress = collections.namedtuple("NetAddress", "object port")
"""Source or sink of a stream of packets.

Predominantly used in intermediate representations.

Parameters
----------
object : :py:class:`.ObjectAnnotation`
port : :py:class:`.OutputPort` or :py:class:`.InputPort`
"""


class AnnotatedNet(object):
    """SpiNNaker specific annotation of a Nengo Connection.

    Attributes
    ----------
    source : :py:class:`.NetAddress`
        Source of packets to be transmitted across the net.
    sinks : [:py:class:`.NetAddress`]
        Targets of packets transmitted across the net.
    keyspace : :py:class:`rig.bitfield.BitField` or None
        Keyspace used to route packets across the network.
    latching : bool
        Indicates that the receiving buffer must *not* be reset every
        simulation timestep but must hold its value until it next receives a
        packet.
    weight : int
        Indication of the number of packets expected to flow over the net every
        time-step.
    """
    __slots__ = ["source", "sinks", "keyspace", "latching", "weight"]

    def __init__(self, source, sinks, keyspace=None, latching=False, weight=0):
        self.source = source
        self.keyspace = keyspace
        self.latching = latching
        self.weight = weight

        # Copy the sinks if they are a list, otherwise make a list containing
        # the single sink
        if isinstance(sinks, list):
            self.sinks = list(sinks)
        else:
            self.sinks = [sinks]


@Annotations.object_builders.register(nengo.base.NengoObject)
class ObjectAnnotation(object):
    """Thin wrapper for objects which exist in a Nengo network and need
    representing in an intermediate form.

    Attributes
    ----------
    constraints : [IntermediateConstraint, ...]
        List of intermediate constraints that should be applied the vertex
        built from this intermediate representation.
    """
    __slots__ = ["seed", "constraints"]

    def __init__(self, obj, constraints=list()):
        """Create a new intermediate representation for the given object.
        """
        self.constraints = constraints[:]


@Annotations.source_getters.register(nengo.base.NengoObject)
def get_source_standard(conn, anns):
    """Return a standard source for objects which have no special handler.

    Parameters
    ----------
    conn : :py:class:`nengo.Connection`
    anns : :py:class:`.Annotations`
    """
    return soss(NetAddress(anns.objects[conn.pre_obj], OutputPort.standard))


@Annotations.sink_getters.register(nengo.base.NengoObject)
def get_sink_standard(conn, anns):
    """Return a standard sink for objects which have no special handler.

    Parameters
    ----------
    conn : :py:class:`nengo.Connection`
    anns : :py:class:`.Annotations`
    """
    return soss(NetAddress(anns.objects[conn.post_obj], InputPort.standard))


@Annotations.probe_builders.register(nengo.base.NengoObject)
def get_probe(probe, irn):
    """Add a standard probe.

    Parameters
    ----------
    probe : :py:class:`nengo.Probe`
    irn : :py:class:`.Annotations`
    """
    ir_probe = ObjectAnnotation(probe)
    return ir_probe, [], []


# Helper functions below this point
# ---------------------------------

def _get_object_annotation(builders, obj, built_obj):
    """Get the annotation for the object.

    Parameters
    ----------
    builders : {type: callable, ...}
    obj : object
    built_obj : object
    """
    # Call the appropriate builder
    try:
        builder = builders[obj.__class__]
    except KeyError:
        raise TypeError(
            "Could not construct intermediate representation for "
            "object of type {}".format(obj.__class__.__name__)
        )
    return builder(obj, built_obj)


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
    (:py:class:`.AnnotatedNet`, extra objects, extra connections)
        The intermediate net and any extra objects or connections that were
        deemed necessary.
    """
    def _get_intermediate_endpoint(endpoint, getters, connection, irn):
        (cls, name, verb) = endpoint.value
        try:
            builder = getters[cls(connection)]
        except KeyError:
            raise TypeError(
                "Could not determine the {} for connections {} at object of "
                "type {}".format(name, verb, cls(connection).__name__)
            )

        # Get the endpoint
        return builder(connection, irn)

    # Get the source for the connection
    source_spec = _get_intermediate_endpoint(
        _EndpointType.source, source_getters, connection, irn)

    # Get the sink for the connection
    sink_spec = _get_intermediate_endpoint(
        _EndpointType.sink, sink_getters, connection, irn)

    # If the source_spec is an existing net, then add the sink to that net and
    # return
    if isinstance(source_spec, AnnotatedNet):
        net = source_spec
        net.sinks.append(sink_spec.target)
        return net, sink_spec.extra_objects, sink_spec.extra_nets

    # If no source is specified then we abort the connection
    if source_spec is None or source_spec.target is None:
        return None, [], []

    # If no sink is specified then we abort the connection
    if sink_spec is None or sink_spec.target is None:
        return None, [], []

    # Resolve the keyspaces, allow either end to require a keyspace: if both
    # ends require keyspaces then we fail.
    source_ks = source_spec.keyspace
    sink_ks = sink_spec.keyspace
    if source_ks is None and sink_ks is None:
        ks = None
    elif source_ks is not None and sink_ks is None:
        ks = source_ks
    elif source_ks is None and sink_ks is not None:
        ks = sink_ks
    else:
        raise NotImplementedError("Cannot merge two keyspaces")

    if source_spec.weight is not None or sink_spec.weight is not None:
        # Take the largest specified weight
        source_weight = 0 if source_spec.weight is None else source_spec.weight
        sink_weight = 0 if sink_spec.weight is None else sink_spec.weight

        weight = max(source_weight, sink_weight)
    else:
        # The weight of the net is taken from the size_out of the Connection.
        # This is correct for value-transmission in the majority of cases and
        # is over-cautious for spike-transmission.
        weight = connection.size_out

    # Determine whether the net should be latching (not by default).  There
    # shouldn't be any case where there is a mismatch between the source and
    # sink in this regard, or where a mismatch would result in incorrect
    # behaviour.
    latching = source_spec.latching or sink_spec.latching

    # Combine the sets of extra objects and connections requested by the sink
    # and sources.
    extra_objs = source_spec.extra_objects + sink_spec.extra_objects
    extra_conns = source_spec.extra_nets + sink_spec.extra_nets

    # Build the new net
    return (AnnotatedNet(source_spec.target, sink_spec.target, ks,
                         latching, weight), extra_objs, extra_conns)


def _get_intermediate_probe(builders, probe, ann):
    """Get the seed for a probe and call the appropriate intermediate
    representation builder.

    Parameters
    ----------
    builders : {type: callable, ...}
    probe : :py:class:`nengo.Probe`
        Probe to build the intermediate representation for.
    ann : :py:class:`.Annotation`
    """
    # Get the build function
    try:
        builder = builders[probe.target.__class__]
    except KeyError:
        raise TypeError(
            "Could not determine how to deal with probe with target of type "
            "{}".format(probe.target.__class__.__name__)
        )

    # Call the builder function
    return builder(probe, ann)
