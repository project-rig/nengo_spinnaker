"""Higher and lower level netlist items.
"""
import logging
import rig.netlist
from rig import place_and_route  # noqa : F401
from rig.place_and_route.constraints import ReserveResourceConstraint

from rig.place_and_route.utils import (build_application_map,
                                       build_routing_tables)
from rig.machine import Cores
from rig.machine_control.utils import sdram_alloc_for_vertices
from six import iteritems

from .partition_and_cluster import identify_clusters
from .utils.itertools import flatten

logger = logging.getLogger(__name__)


class Net(rig.netlist.Net):
    """A net represents connectivity from one vertex (or vertex slice) to many
    vertices and vertex slices.

    ..note::
        This extends the Rig :py:class:`~rig.netlist.Netlist` to add Nengo
        specific attributes and terms.

    Attributes
    ----------
    source : :py:class:`.Vertex` or :py:class:`.VertexSlice`
        Vertex or vertex slice which is the source of the net.
    sinks : [:py:class:`.Vertex` or :py:class:`.VertexSlice`, ...]
        List of vertices and vertex slices which are the sinks of the net.
    weight : int
        Number of packets transmitted across the net every simulation
        time-step.
    keyspace : :py:class:`rig.bitfield.BitField`
        32-bit bitfield instance that can be used to derive the routing key and
        mask for the net.
    """
    def __init__(self, source, sinks, weight, keyspace):
        """Create a new net.

        See :py:meth:`~rig.netlist.Net.__init__`.

        Parameters
        ----------
        keyspace : :py:class:`rig.bitfield.BitField`
            32-bit bitfield instance that can be used to derive the routing key
            and mask for the net.
        """
        # Assert that the keyspace is 32-bits long
        if keyspace.length != 32:
            raise ValueError(
                "keyspace: Must be 32-bits long, not {}".format(
                    keyspace.length)
            )
        super(Net, self).__init__(source, sinks, weight)
        self.keyspace = keyspace

    @property
    def as_rig_primitive(self):
        """Return a new :py:class:`rig.netlist.Net` representing this Net."""
        return rig.netlist.Net(self.source, self.sinks, self.weight)


class Vertex(object):
    """Represents a nominal unit of computation (a single instance or many
    instances of an application running on a SpiNNaker machine) or an external
    device that is connected to the SpiNNaker network.

    Attributes
    ----------
    application : str or None
        Path to application which should be loaded onto SpiNNaker to simulate
        this vertex, or None if no application is required.
    constraints : [constraint, ...]
        The :py:mod:`~rig.place_and_route.constraints` which should be applied
        to the placement and routing related to the vertex.
    resource : {resource: usage, ...}
        Mapping from resource type to the consumption of that resource, in
        whatever is an appropriate unit.
    cluster : int or None
        Index of the cluster the vertex is a part of.
    """
    def __init__(self, application=None, resources=dict(), constraints=list()):
        """Create a new Vertex.
        """
        self.application = application
        self.constraints = list(constraints)
        self.resources = dict(resources)
        self.cluster = None


class VertexSlice(Vertex):
    """Represents a portion of a nominal unit of computation.

    Attributes
    ----------
    application : str or None
        Path to application which should be loaded onto SpiNNaker to simulate
        this vertex, or None if no application is required.
    constraints : [constraint, ...]
        The :py:mod:`~rig.place_and_route.constraints` which should be applied
        to the placement and routing related to the vertex.
    resource : {resource: usage, ...}
        Mapping from resource type to the consumption of that resource, in
        whatever is an appropriate unit.
    slice : :py:class:`slice`
        Slice of the unit of computation which is represented by this vertex
        slice.
    """
    def __init__(self, slice, application=None, resources=dict(),
                 constraints=list()):
        super(VertexSlice, self).__init__(application, resources, constraints)
        self.slice = slice


class Netlist(object):
    """A netlist represents a set of executables to run on a SpiNNaker machine
    and their communication links.

    Attributes
    ----------
    nets : [:py:class:`~.Net`, ...]
        List of nets (communication).
    vertices : [:py:class:`~.Vertex` or :py:class:`~.VertexSlice`, ...]
        List of vertex objects (executables).
    keyspaces : :py:class:`~nengo_spinnaker.utils.keyspaces.KeyspaceContainer`
        Object containing keyspaces for nets.
    groups : [{:py:class:`~.Vertex`, ...}, ...]
        List of groups of vertices.
    load_functions : [`fn(netlist, controller)`, ...]
        List of functions which will be called to load the model to a SpiNNaker
        machine.  Each must accept a netlist and a controller.
    before_simulation_functions : [`fn(netlist, simulator, n_steps)`, ...]
        List of functions which will be called to prepare the executables for a
        number of simulation steps.  Each must accept a netlist, the simulator
        and a number of simulation steps.
    after_simulation_functions : [`fn(netlist, simulator, n_steps)`, ...]
        List of functions which will be called to clean the executables after a
        number of simulation steps.  Each must accept a netlist, the simulator
        and a number of simulation steps.
    placements : {vertex: (x, y), ...}
        Map from vertices to the chips on which they are placed.
    allocations : {vertex: {resource: allocation, ...}, ...}
        Map of vertices to the resources they have been assigned.
    routes : {net: routing tree, ...}
        Map of nets to the routes through the machine to which they correspond.
    vertices_memory : {vertex: filelike, ...}
        Map of vertices to file-like views of the SDRAM they have been
        allocated.
    """
    def __init__(self, nets, vertices, keyspaces, groups,
                 load_functions=list(), before_simulation_functions=list(),
                 after_simulation_functions=list()):
        # Store given parameters
        self.nets = list(nets)
        self.vertices = list(vertices)
        self.keyspaces = keyspaces
        self.groups = list(groups)
        self.load_functions = list(load_functions)
        self.before_simulation_functions = list(before_simulation_functions)
        self.after_simulation_functions = list(after_simulation_functions)

        # Create containers for the attributes that are filled in by place and
        # route.
        self.placements = dict()
        self.allocations = dict()
        self.routes = dict()
        self.vertices_memory = dict()

    def as_rig_arguments(self):
        """Construct arguments for Rig from the Netlist."""
        vertices_resources = {v: v.resources for v in self.vertices}
        nets = [net.as_rig_primitive for net in self.nets]
        constraints = list(flatten(v.constraints for v in self.vertices))
        constraints.append(ReserveResourceConstraint(Cores, slice(0, 1)))

        return {"vertices_resources": vertices_resources,
                "nets": nets,
                "constraints": constraints
                }

    def place_and_route(self, machine,
                        place=place_and_route.place,
                        place_kwargs={},
                        allocate=place_and_route.allocate,
                        allocate_kwargs={},
                        route=place_and_route.route,
                        route_kwargs={}):
        """Place and route the netlist onto the given SpiNNaker machine.

        Parameters
        ----------
        machine : :py:class:`~rig.machine.Machine`
            Machine onto which the netlist should be placed and routed.

        Other Parameters
        ----------------
        place : function
            Placement function. Must support the interface defined by Rig.
        place_kwargs : dict
            Keyword arguments for the placement method.
        allocate : function
            Resource allocation function. Must support the interface defined by
            Rig.
        allocate_kwargs : dict
            Keyword arguments for the allocation function.
        route : function
            Router function. Must support the interface defined by Rig.
        route_kwargs : dict
            Keyword arguments for the router function.
        """
        # Build a map of vertices to the resources they require, get a list of
        # constraints.
        args = self.as_rig_arguments()
        vertices_resources = args["vertices_resources"]
        constraints = args["constraints"]

        # Perform placement and allocation
        self.placements = place(vertices_resources, self.nets,
                                machine, constraints, **place_kwargs)
        self.allocations = allocate(vertices_resources, self.nets, machine,
                                    constraints, self.placements,
                                    **allocate_kwargs)

        # Identify clusters and modify keyspaces and vertices appropriately
        identify_clusters(self.placements, self.nets, self.groups)

        # Fix all keyspaces
        self.keyspaces.assign_fields()

        # Finally, route all nets
        self.routes = route(vertices_resources, self.nets, machine,
                            constraints, self.placements, self.allocations,
                            **route_kwargs)

    def load_application(self, controller):
        """Load the netlist to a SpiNNaker machine.

        Parameters
        ----------
        controller : :py:class:`~rig.machine_control.MachineController`
            Controller to use to communicate with the machine.
        """
        # Build and load the routing tables, first by building a mapping from
        # nets to keys and masks.
        logger.debug("Loading routing tables")
        net_keys = {n: (n.keyspace.get_value(tag=self.keyspaces.routing_tag),
                        n.keyspace.get_mask(tag=self.keyspaces.routing_tag))
                    for n in self.nets}
        routing_tables = build_routing_tables(self.routes, net_keys)
        controller.load_routing_tables(routing_tables)

        # Assign memory to each vertex as required
        logger.debug("Assigning application memory")
        self.vertices_memory = sdram_alloc_for_vertices(
            controller, self.placements, self.allocations
        )

        # Inform the vertices of where that chunk of memory is
        for vertex, memory in iteritems(self.vertices_memory):
            x, y = self.placements[vertex]
            p = self.allocations[vertex][Cores].start
            controller.write_vcpu_struct_field(
                "user0", memory.address, x, y, p)

        # Call each loading function in turn
        logger.debug("Loading data")
        for fn in self.load_functions:
            fn(self, controller)

        # Load the applications onto the machine
        logger.debug("Loading application executables")
        vertices_applications = {v: v.application for v in self.vertices
                                 if v.application is not None}
        application_map = build_application_map(
            vertices_applications, self.placements, self.allocations
        )
        controller.load_application(application_map)

    def before_simulation(self, simulator, n_steps):
        """Prepare the objects in the netlist for a simulation of a given
        number of steps.
        """
        # Write into memory the duration of the simulation
        for vertex in self.vertices:
            x, y = self.placements[vertex]
            p = self.allocations[vertex][Cores].start
            simulator.controller.write_vcpu_struct_field("user1", n_steps,
                                                         x, y, p)

        # Call all the "before simulation" functions
        for fn in self.before_simulation_functions:
            fn(self, simulator, n_steps)

    def after_simulation(self, simulator, n_steps):
        """Retrieve data from the objects in the netlist after a simulation of
        a given number of steps.
        """
        for fn in self.after_simulation_functions:
            fn(self, simulator, n_steps)
