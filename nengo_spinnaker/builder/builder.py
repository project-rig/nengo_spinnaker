"""SpiNNaker builder for Nengo models."""
import collections
import enum
import itertools
import nengo
from nengo.cache import NoDecoderCache
from nengo.utils import numpy as npext
import numpy as np
from six import iteritems, itervalues

from nengo_spinnaker.netlist import Net, Netlist
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
    connections_signals : {connection: :py:`~.Signal`, ...}
        Map of connections to the signals that simulate them.
    extra_signals: [operator, ...]
        Additional signals.
    """

    builders = collections_ext.registerabledict()
    """Builders for Nengo objects.

    Each object in the Nengo network is built by calling a builder function
    registered in this dictionary.  The builder function must be of the form:

        .. py:function:: builder(model, object)

    It is free to modify the model as required (including doing nothing to
    suppress SpiNNaker simulation of the object).
    """

    connection_parameter_builders = collections_ext.registerabledict()
    """Functions which can build the parameters for a connection.

    The parameters for a connection are built differently depending on the type
    of the object at the start of the connection.  Functions to perform this
    building can be registered in this dictionary against this type of the
    originating object.  Functions must be of the form:

        .. py:function:: builder(model, connection)

    It is recommended that builders return a :py:class:`~.BuiltConnection`
    object as this will be inserted into the `params` dictionary in the
    :py:class:`~.Model`.
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
        self.connections_signals = dict()
        self.extra_signals = list()

        if keyspaces is None:
            keyspaces = KeyspaceContainer()
        self.keyspaces = keyspaces

        # Internally used dictionaries to construct keyspace information
        self._obj_ids = collections.defaultdict(collections_ext.counter())
        self._obj_conn_ids = collections.defaultdict(
            lambda: collections.defaultdict(collections_ext.counter())
        )

        # Internally used dictionaries of build methods
        self._builders = dict()
        self._connection_parameter_builders = dict()
        self._source_getters = dict()
        self._sink_getters = dict()
        self._probe_builders = dict()

    def _get_object_and_connection_id(self, obj, connection):
        """Get a unique ID for the object and connection pair for use in
        building instances of the default Nengo keyspace.
        """
        # Get the object ID and then the connection ID
        obj_id = self._obj_ids[obj]
        conn_id = self._obj_conn_ids[obj][connection]
        return (obj_id, conn_id)

    def build(self, network, extra_builders={},
              extra_source_getters={}, extra_sink_getters={},
              extra_connection_parameter_builders={},
              extra_probe_builders={}):
        """Build a Network into this model.

        Parameters
        ----------
        network : :py:class:`~nengo.Network`
            Nengo network to build.  Passthrough Nodes will be removed.
        extra_builders : {type: fn, ...}
            Extra builder methods.
        extra_source_getters : {type: fn, ...}
            Extra source getter methods.
        extra_sink_getters : {type: fn, ...}
            Extra sink getter methods.
        extra_connection_parameter_builder : {type: fn, ...}
            Extra connection parameter builders.
        extra_probe_builders : {type: fn, ...}
            Extra probe builder methods.
        """
        # Store the network config
        self.config = network.config

        # Get a clean set of builders and getters
        self._builders = collections_ext.mrolookupdict()
        self._builders.update(self.builders)
        self._builders.update(extra_builders)

        self._connection_parameter_builders = collections_ext.mrolookupdict()
        self._connection_parameter_builders.update(
            self.connection_parameter_builders
        )
        self._connection_parameter_builders.update(
            extra_connection_parameter_builders
        )

        self._source_getters = collections_ext.mrolookupdict()
        self._source_getters.update(self.source_getters)
        self._source_getters.update(extra_source_getters)

        self._sink_getters = collections_ext.mrolookupdict()
        self._sink_getters.update(self.sink_getters)
        self._sink_getters.update(extra_sink_getters)

        self._probe_builders = dict()
        self._probe_builders.update(self.probe_builders)
        self._probe_builders.update(extra_probe_builders)

        # Build
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
        self.seeds[conn] = get_seed(conn, self.rng)
        self.params[conn] = \
            self._connection_parameter_builders[type(conn.pre_obj)](self, conn)

        # Get the source and sink specification, then make the signal provided
        # that neither of specs is None.
        source = self._source_getters[type(conn.pre_obj)](self, conn)
        sink = self._sink_getters[type(conn.post_obj)](self, conn)

        if source is not None and sink is not None:
            assert conn not in self.connections_signals
            self.connections_signals[conn] = _make_signal(self, conn,
                                                          source, sink)

    def make_probe(self, probe):
        """Call an appropriate build function for the given probe."""
        self.seeds[probe] = get_seed(probe, self.rng)

        # Get the target type
        target_obj = probe.target
        if isinstance(target_obj, nengo.base.ObjView):
            target_obj = target_obj.obj

        # Build
        self._probe_builders[type(target_obj)](self, probe)

    def get_signals_connections_from_object(self, obj):
        """Get a dictionary mapping ports to signals to connections which
        originate from a given intermediate object.
        """
        ports_sigs_conns = collections.defaultdict(
            lambda: collections.defaultdict(collections_ext.noneignoringlist)
        )

        for (conn, signal) in itertools.chain(
                iteritems(self.connections_signals),
                ((None, s) for s in self.extra_signals)):
            if signal.source.obj is obj:
                ports_sigs_conns[signal.source.port][signal].append(conn)

        return ports_sigs_conns

    def get_signals_connections_to_object(self, obj):
        """Get a dictionary mapping ports to signals to connections which
        terminate at a given intermediate object.
        """
        ports_sigs_conns = collections.defaultdict(
            lambda: collections.defaultdict(collections_ext.noneignoringlist)
        )

        for (conn, signal) in itertools.chain(
                iteritems(self.connections_signals),
                ((None, s) for s in self.extra_signals)):
            for sink in signal.sinks:
                if sink.obj is obj:
                    ports_sigs_conns[sink.port][signal].append(conn)

        return ports_sigs_conns

    def make_netlist(self, *args, **kwargs):
        """Convert the model into a netlist for simulating on SpiNNaker.

        Returns
        -------
        :py:class:`~nengo_spinnaker.netlist.Netlist`
            A netlist which can be placed and routed to simulate this model on
            a SpiNNaker machine.
        """
        # Call each operator to make vertices
        operator_vertices = dict()
        vertices = collections_ext.flatinsertionlist()
        load_functions = collections_ext.noneignoringlist()
        before_simulation_functions = collections_ext.noneignoringlist()
        after_simulation_functions = collections_ext.noneignoringlist()

        for op in itertools.chain(itervalues(self.object_operators),
                                  self.extra_operators):
            vxs, load_fn, pre_fn, post_fn = op.make_vertices(
                self, *args, **kwargs
            )

            operator_vertices[op] = vxs
            vertices.append(vxs)

            load_functions.append(load_fn)
            before_simulation_functions.append(pre_fn)
            after_simulation_functions.append(post_fn)

        # Construct the groups set
        groups = list()
        for vxs in itervalues(operator_vertices):
            # If multiple vertices were provided by an operator then we add
            # them as a new group.
            if isinstance(vxs, collections.Iterable):
                groups.append(set(vxs))

        # Construct nets from the signals
        nets = list()
        for signal in itertools.chain(itervalues(self.connections_signals),
                                      self.extra_signals):
            # Get the source and sink vertices
            sources = operator_vertices[signal.source.obj]
            if not isinstance(sources, collections.Iterable):
                sources = (sources, )

            sinks = collections_ext.flatinsertionlist()
            for sink in signal.sinks:
                sinks.append(operator_vertices[sink.obj])

            # Create the net(s)
            for source in sources:
                nets.append(Net(source, list(sinks),
                            signal.weight, signal.keyspace))

        # Return a netlist
        return Netlist(
            nets=nets,
            vertices=vertices,
            keyspaces=self.keyspaces,
            groups=groups,
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


class OutputPort(enum.Enum):
    """Indicate the intended transmitting part of an executable."""
    standard = 0
    """Standard, value-based, output port."""


class InputPort(enum.Enum):
    """Indicate the intended receiving part of an executable."""
    standard = 0
    """Standard, value-based, output port."""


class netlistspec(collections.namedtuple(
        "netlistspec", "vertices, load_function, before_simulation_function, "
                       "after_simulation_function")):
    """Specification of how an operator should be added to a netlist."""
    def __new__(cls, vertices, load_function=None,
                before_simulation_function=None,
                after_simulation_function=None):
        return super(netlistspec, cls).__new__(
            cls, vertices, load_function, before_simulation_function,
            after_simulation_function
        )


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


class Signal(
        collections.namedtuple("Signal",
                               "source, sinks, keyspace, weight, latching")):
    """Represents a stream of multicast packets across a SpiNNaker machine.

    Attributes
    ----------
    source : :py:class:`~.ObjectPort`
        Source object and port of signal.
    sinks : [:py:class:`.~ObjectPort`, ...]
        Sink objects and ports of the signal.
    keyspace : keyspace
        Keyspace used for packets representing the signal.
    weight : int
        Number of packets expected to represent the signal during a single
        timestep.
    latching : bool
        Indicates that the receiving buffer must *not* be reset every
        simulation timestep but must hold its value until it next receives a
        packet.
    """
    def __new__(cls, source, sinks, keyspace, weight=0, latching=False):
        # Ensure the sinks are a list
        if (not isinstance(sinks, ObjectPort) and
                isinstance(sinks, collections.Iterable)):
            sinks = list(sinks)
        else:
            sinks = [sinks]

        # Create the tuple
        return super(Signal, cls).__new__(cls, source, sinks, keyspace,
                                          weight, latching)

    def __hash__(self):
        return hash(id(self))


def _make_signal(model, connection, source_spec, sink_spec):
    """Create a Signal."""
    # Get the keyspace
    if source_spec.keyspace is None and sink_spec.keyspace is None:
        # Using the default keyspace, get the object and connection ID
        obj_id, conn_id = model._get_object_and_connection_id(
            connection.pre_obj, connection
        )

        # Create the keyspace from the default one provided by the model
        keyspace = model.keyspaces["nengo"](object=obj_id, connection=conn_id)
    elif source_spec.keyspace is not None and sink_spec.keyspace is None:
        # Use the keyspace required by the source
        keyspace = source_spec.keyspace
    elif source_spec.keyspace is None and sink_spec.keyspace is not None:
        # Use the keyspace required by the sink
        keyspace = sink_spec.keyspace
    else:
        # Collision between the keyspaces
        raise NotImplementedError("Cannot merge two keyspaces")

    # Get the weight
    weight = max((0 or source_spec.weight,
                  0 or sink_spec.weight,
                  getattr(connection.post_obj, "size_in", 0)))

    # Determine if the connection is latching - there should probably never be
    # a case where these requirements differ, but this may need revisiting.
    latching = source_spec.latching or sink_spec.latching

    # Create the signal
    return Signal(
        source_spec.target, sink_spec.target, keyspace, weight, latching
    )
