"""SpiNNaker builder for Nengo models."""
import collections
import itertools
import nengo
from nengo.cache import NoDecoderCache
from nengo.utils import numpy as npext
import numpy as np
from six import itervalues

from . import model
from nengo_spinnaker.netlist import NMNet, Netlist
from nengo_spinnaker.utils import collections as collections_ext
from nengo_spinnaker.utils.keyspaces import KeyspaceContainer

BuiltConnection = collections.namedtuple(
    "BuiltConnection", "decoders, eval_points, transform, solver_info"
)
"""Parameters which describe a Connection."""


def get_seed(obj, rng):
    seed = rng.randint(npext.maxint)
    return (seed if getattr(obj, "seed", None) is None else obj.seed)


class Model(object):
    """Model which has been built specifically for simulation on SpiNNaker.

    Attributes
    ----------
    dt : float
        Simulation timestep in seconds.
    machine_timestep : int
        Real-time duration of a simulation timestep in microseconds.
    decoder_cache :
        Cache used to reduce the time spent solving for decoders.
    params : {object: build details, ...}
        Map of Nengo objects (Ensembles, Connections, etc.) to their built
        equivalents.
    seeds : {object: int, ...}
        Map of Nengo objects to the seeds used in their construction.
    keyspaces : {keyspace_name: keyspace}
        Map of keyspace names to the keyspace which they may use.
    objects_operators : {object: operator, ...}
        Map of objects to the operators which will simulate them on SpiNNaker.
    extra_operators: [operator, ...]
        Additional operators.
    connection_map :
        Data structure which performs insertion-minimisation on connections
        wherein each source object is associated with a dictionary mapping
        ports to lists of unique signals.
    """

    builders = collections_ext.registerabledict()
    """Builders for Nengo objects.

    Each object in the Nengo network is built by calling a builder function
    registered in this dictionary.  The builder function must be of the form:

        .. py:function:: builder(model, object)

    It is free to modify the model as required (including doing nothing to
    suppress SpiNNaker simulation of the object).
    """

    transmission_parameter_builders = collections_ext.registerabledict()
    """Functions which can provide the parameters for transmitting values to
    simulate a connection.

    The parameters required to form multicast packets to simulate a Nengo
    Connection vary depending on the type of the object at the start of the
    connection. Functions to build these parameters can be registered in this
    dictionary against the type of the originating object. Functions must be of
    the form:

        .. py:function:: builder(model, connection)

    It is recommended that functions set the value of
    `model.params[connection]` to an instance of :py:class:`~.BuiltConnection`
    alongside returning an appropriate value to use as the transmission
    parameters.
    """

    source_getters = collections_ext.registerabledict()
    """Functions to retrieve the specifications for the sources of signals.

    Before a connection is built an attempt is made to determine where the
    signal it represents on SpiNNaker will originate; a source getter is called
    to perform this task.  A source getter should resemble:

        .. py:function:: getter(model, connection)

    The returned item can be one of two things:
     * `None` will suppress simulation of the connection on SpiNNaker -- an
       example of this being useful is in optimising out connections from
       constant valued Nodes to ensembles or reusing an existing connection.
     * a :py:class:`~.spec` object which will be used to determine nature of
       the signal (in particular, the key and mask that it should use, whether
       it is latching or otherwise and the cost of the signal in terms of the
       frequency of packets across it).
    """

    reception_parameter_builders = collections_ext.registerabledict()
    """Functions which can provide the parameters for receiving values which
    simulate a connection.

    The parameters required to interpret multicast packets can vary based on
    the type of the object at the end of a Nengo Connection. Functions to build
    these parameters can be registered in this dictionary against the type of
    the terminating object.  Functions must of the form:

        .. py:function:: builder(model, connection)
    """

    sink_getters = collections_ext.registerabledict()
    """Functions to retrieve the specifications for the sinks of signals.

    A sink getter is analogous to a `source_getter`, but refers to the
    terminating end of a signal.
    """

    probe_builders = collections_ext.registerabledict()
    """Builder functions for probes.

    Probes can either require the modification of an existing object or the
    insertion of a new object into the model. A probe builder can be registered
    against the target of the probe and must be of the form:

        .. py:function:: probe_builder(model, probe)

    And is free the modify the model and existing objects as required.
    """

    def __init__(self, dt=0.001, machine_timestep=1000,
                 decoder_cache=NoDecoderCache(), keyspaces=None):
        self.dt = dt
        self.machine_timestep = machine_timestep
        self.decoder_cache = decoder_cache

        self.params = dict()
        self.seeds = dict()
        self.rngs = dict()
        self.rng = None

        self.config = None
        self.object_operators = dict()
        self.extra_operators = list()
        self.connection_map = model.ConnectionMap()

        if keyspaces is None:
            keyspaces = KeyspaceContainer()
        self.keyspaces = keyspaces

        # Builder dictionaries
        self._builders = dict()
        self._transmission_parameter_builders = dict()
        self._source_getters = dict()
        self._reception_parameter_builders = dict()
        self._sink_getters = dict()
        self._probe_builders = dict()

    def build(self, network, **kwargs):
        """Build a Network into this model.

        Parameters
        ----------
        network : :py:class:`~nengo.Network`
            Nengo network to build.  Passthrough Nodes will be removed.
        """
        # Store the network config
        self.config = network.config

        # Get a clean set of builders and getters
        self._builders = collections_ext.mrolookupdict()
        self._builders.update(self.builders)
        self._builders.update(kwargs.get("extra_builders", {}))

        self._transmission_parameter_builders = \
            collections_ext.mrolookupdict()
        self._transmission_parameter_builders.update(
            self.transmission_parameter_builders)
        self._transmission_parameter_builders.update(
            kwargs.get("extra_transmission_parameter_builders", {}))

        self._source_getters = collections_ext.mrolookupdict()
        self._source_getters.update(self.source_getters)
        self._source_getters.update(kwargs.get("extra_source_getters", {}))

        self._reception_parameter_builders = collections_ext.mrolookupdict()
        self._reception_parameter_builders.update(
            self.reception_parameter_builders)
        self._reception_parameter_builders.update(
            kwargs.get("extra_reception_parameter_builders", {}))

        self._sink_getters = collections_ext.mrolookupdict()
        self._sink_getters.update(self.sink_getters)
        self._sink_getters.update(kwargs.get("extra_sink_getters", {}))

        self._probe_builders = dict()
        self._probe_builders.update(self.probe_builders)
        self._probe_builders.update(kwargs.get("extra_probe_builders", {}))

        # Build
        with self.decoder_cache:
            self._build_network(network)

    def _build_network(self, network):
        # Get the seed for the network
        self.seeds[network] = get_seed(network, np.random)

        # Build all subnets
        for subnet in network.networks:
            self._build_network(subnet)

        # Get the random number generator for the network
        self.rngs[network] = np.random.RandomState(self.seeds[network])
        self.rng = self.rngs[network]

        # Build all objects
        for obj in itertools.chain(network.ensembles, network.nodes):
            self.make_object(obj)

        # Build all the connections
        for connection in network.connections:
            self.make_connection(connection)

        # Build all the probes
        for probe in network.probes:
            self.make_probe(probe)

    def make_object(self, obj):
        """Call an appropriate build function for the given object.
        """
        self.seeds[obj] = get_seed(obj, self.rng)
        self._builders[type(obj)](self, obj)

    def make_connection(self, conn):
        """Make a Connection and add a new signal to the Model.

        This method will build a connection and construct a new signal which
        will be included in the model.
        """
        # Set the seed for the connection
        self.seeds[conn] = get_seed(conn, self.rng)

        # Get the transmission parameters and reception parameters for the
        # connection.
        pre_type = type(conn.pre_obj)
        transmission_params = \
            self._transmission_parameter_builders[pre_type](self, conn)
        post_type = type(conn.post_obj)
        reception_params = \
            self._reception_parameter_builders[post_type](self, conn)

        # Get the source and sink specification, then make the signal provided
        # that neither of specs is None.
        source = self._source_getters[pre_type](self, conn)
        sink = self._sink_getters[post_type](self, conn)

        if not (source is None or sink is None):
            # Construct the signal parameters
            signal_params = _make_signal_parameters(source, sink, conn)

            # Add the connection to the connection map, this will automatically
            # merge connections which are equivalent.
            self.connection_map.add_connection(
                source.target.obj, source.target.port, signal_params,
                transmission_params, sink.target.obj, sink.target.port,
                reception_params
            )

    def make_probe(self, probe):
        """Call an appropriate build function for the given probe."""
        self.seeds[probe] = get_seed(probe, self.rng)

        # Get the target type
        target_obj = probe.target
        if isinstance(target_obj, nengo.base.ObjView):
            target_obj = target_obj.obj

        # Build
        self._probe_builders[type(target_obj)](self, probe)

    def get_signals_from_object(self, *args):  # pragma : no cover
        """Get the signals transmitted by a source object.

        Returns
        -------
        {port : [signal_parameters, ...], ...}
            Dictionary mapping ports to lists of parameters for the signals
            that originate from them.
        """
        return self.connection_map.get_signals_from_object(*args)

    def get_signals_to_object(self, *args):  # pragma : no cover
        """Get the signals received by a sink object.

        Returns
        -------
        {port : [ReceptionSpec, ...], ...}
            Dictionary mapping ports to the lists of objects specifying
            incoming signals.
        """
        return self.connection_map.get_signals_to_object(*args)

    def make_netlist(self, *args, **kwargs):
        """Convert the model into a netlist for simulating on SpiNNaker.

        Returns
        -------
        :py:class:`~nengo_spinnaker.netlist.Netlist`
            A netlist which can be placed and routed to simulate this model on
            a SpiNNaker machine.
        """
        # Remove any passthrough Nodes which don't connect to anything
        from nengo_spinnaker import operators
        removed_operators = model.remove_sinkless_objects(self.connection_map,
                                                          operators.Filter)

        # Apply the default keyspace to any signals without keyspaces
        self.connection_map.add_default_keyspace(self.keyspaces["nengo"])

        # Call each operator to make vertices
        operator_vertices = dict()
        vertices = collections_ext.flatinsertionlist()
        load_functions = collections_ext.noneignoringlist()
        before_simulation_functions = collections_ext.noneignoringlist()
        after_simulation_functions = collections_ext.noneignoringlist()
        constraints = collections_ext.flatinsertionlist()

        for op in itertools.chain(itervalues(self.object_operators),
                                  self.extra_operators):
            # Skip any operators that were previously removed
            if op in removed_operators:
                continue

            # Otherwise call upon the operator to build vertices for the
            # netlist.
            vxs, load_fn, pre_fn, post_fn, constraint = op.make_vertices(
                self, *args, **kwargs
            )

            operator_vertices[op] = vxs
            vertices.append(vxs)

            load_functions.append(load_fn)
            before_simulation_functions.append(pre_fn)
            after_simulation_functions.append(post_fn)

            if constraint is not None:
                constraints.append(constraint)

        # Construct the groups set
        groups = list()
        for vxs in itervalues(operator_vertices):
            # If multiple vertices were provided by an operator then we add
            # them as a new group.
            if isinstance(vxs, collections.Iterable):
                groups.append(set(vxs))

        # Construct nets from the signals
        nets = list()
        for signal, transmission_parameters in \
                self.connection_map.get_signals():
            # Get the source and sink vertices
            original_sources = operator_vertices[signal.source]
            if not isinstance(original_sources, collections.Iterable):
                original_sources = (original_sources, )

            # Filter out any sources which have an `accepts_signal` method and
            # return False when this is called with the signal and transmission
            # parameters.
            sources = list()
            for source in original_sources:
                # For each source which either doesn't have a
                # `transmits_signal` method or returns True when this is called
                # with the signal and transmission parameters add a new net to
                # the netlist.
                if (hasattr(source, "transmits_signal") and not
                        source.transmits_signal(signal,
                                                transmission_parameters)):
                    pass  # This source is ignored
                else:
                    # Add the source to the final list of sources
                    sources.append(source)

            sinks = collections_ext.flatinsertionlist()
            for sink in signal.sinks:
                # Get all the sink vertices
                sink_vertices = operator_vertices[sink]
                if not isinstance(sink_vertices, collections.Iterable):
                    sink_vertices = (sink_vertices, )

                # Include any sinks which either don't have an `accepts_signal`
                # method or return true when this is called with the signal and
                # transmission parameters.
                sinks.append(s for s in sink_vertices if
                             not hasattr(s, "accepts_signal") or
                             s.accepts_signal(signal, transmission_parameters))

            # Create the net(s)
            nets.append(NMNet(sources, list(sinks),
                              signal.weight, signal.keyspace))

        # Return a netlist
        return Netlist(
            nets=nets,
            vertices=vertices,
            keyspaces=self.keyspaces,
            groups=groups,
            constraints=constraints,
            load_functions=load_functions,
            before_simulation_functions=before_simulation_functions,
            after_simulation_functions=after_simulation_functions
        )


ObjectPort = collections.namedtuple("ObjectPort", "obj port")
"""Source or sink of a signal.

Parameters
----------
obj : intermediate object
    Intermediate representation of a Nengo object, or other object, which is
    the source or sink of a signal.
port : port
    Port that is the source or sink of a signal.
"""


class spec(collections.namedtuple("spec",
                                  "target, keyspace, weight, latching")):
    """Specification of a signal which can be returned by either a source or
    sink getter.

    Attributes
    ----------
    target : :py:class:`ObjectPort`
        Source or sink of a signal.

    The other attributes and arguments are as for :py:class:`~.Signal`.
    """
    def __new__(cls, target, keyspace=None, weight=0, latching=False):
        return super(spec, cls).__new__(cls, target, keyspace,
                                        weight, latching)


def _make_signal_parameters(source_spec, sink_spec, connection):
    """Create parameters for a signal using specifications provided by the
    source and sink.

    Parameters
    ----------
    source_spec : spec
        Signal specification parameters from the source of the signal.
    sink_spec : spec
        Signal specification parameters from the sink of the signal.
    connection : nengo.Connection
        The Connection for this signal

    Returns
    -------
    :py:class:`~.SignalParameters`
        Description of the signal.
    """
    # Raise an error if keyspaces are specified by the source and sink
    if source_spec.keyspace is not None and sink_spec.keyspace is not None:
        raise NotImplementedError("Cannot merge keyspaces")

    weight = max((0 or source_spec.weight,
                  0 or sink_spec.weight,
                  getattr(connection.post_obj, "size_in", 0)))

    # Create the signal parameters
    return model.SignalParameters(
        latching=source_spec.latching or sink_spec.latching,
        weight=weight,
        keyspace=source_spec.keyspace or sink_spec.keyspace,
    )
