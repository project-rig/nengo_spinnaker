import logging
from rig import place_and_route  # noqa : F401

from rig.place_and_route.utils import (build_machine,
                                       build_core_constraints,
                                       build_application_map)
from rig.routing_table import (build_routing_table_target_lengths,
                               routing_tree_to_tables,
                               minimise_tables)
from rig.place_and_route import Cores
from rig.machine_control.utils import sdram_alloc_for_vertices
from six import iteritems

from nengo_spinnaker.netlist import utils

logger = logging.getLogger(__name__)


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
    constraints : [contraint, ...]
        List of additional constraints.
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
    def __init__(self, nets, vertices, keyspaces, groups, constraints=list(),
                 load_functions=list(), before_simulation_functions=list(),
                 after_simulation_functions=list()):
        # Store given parameters
        self.nets = list(nets)
        self.vertices = list(vertices)
        self.keyspaces = keyspaces
        self.groups = list(groups)
        self.constraints = list(constraints)
        self.load_functions = list(load_functions)
        self.before_simulation_functions = list(before_simulation_functions)
        self.after_simulation_functions = list(after_simulation_functions)

        # Create containers for the attributes that are filled in by place and
        # route.
        self.placements = dict()
        self.allocations = dict()
        self.net_keyspaces = dict()
        self.routes = dict()
        self.vertices_memory = dict()

    def place_and_route(self, system_info,
                        place=place_and_route.place,
                        place_kwargs={},
                        allocate=place_and_route.allocate,
                        allocate_kwargs={},
                        route=place_and_route.route,
                        route_kwargs={}):
        """Place and route the netlist onto the given SpiNNaker machine.

        Parameters
        ----------
        system_info : \
                :py:class:`~rig.machine_control.MachineController.SystemInfo`
            Describes the system onto which the netlist should be placed and
            routed.

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
        # Generate a Machine and set of core-reserving constraints to prevent
        # the use of non-idle cores.
        machine = build_machine(system_info)
        core_constraints = build_core_constraints(system_info)
        constraints = self.constraints + core_constraints

        # Build a map of vertices to the resources they require, get a list of
        # constraints.
        vertices_resources = {v: v.resources for v in self.vertices}

        # Perform placement and allocation
        place_nets = list(utils.get_nets_for_placement(self.nets))
        self.placements = place(vertices_resources, place_nets, machine,
                                constraints, **place_kwargs)
        self.allocations = allocate(vertices_resources, place_nets, machine,
                                    constraints, self.placements,
                                    **allocate_kwargs)

        # Identify clusters and modify vertices appropriately
        utils.identify_clusters(self.groups, self.placements)

        # Get the nets for routing
        (route_nets,
         vertices_resources,  # Can safely overwrite the resource dictionary
         extended_placements,
         extended_allocations,
         derived_nets) = utils.get_nets_for_routing(
            vertices_resources, self.nets, self.placements, self.allocations)

        # Get a map from the nets we will route with to keyspaces
        self.net_keyspaces = utils.get_net_keyspaces(self.placements,
                                                     derived_nets)

        # Fix all keyspaces
        self.keyspaces.assign_fields()

        # Finally, route all nets using the extended resource dictionary,
        # placements and allocations.
        self.routes = route(vertices_resources, route_nets, machine,
                            constraints, extended_placements,
                            extended_allocations, **route_kwargs)

    def load_application(self, controller, system_info):
        """Load the netlist to a SpiNNaker machine.

        Parameters
        ----------
        controller : :py:class:`~rig.machine_control.MachineController`
            Controller to use to communicate with the machine.
        """
        # Build and load the routing tables, first by building a mapping from
        # nets to keys and masks.
        logger.debug("Loading routing tables")
        net_keys = {n: (ks.get_value(tag=self.keyspaces.routing_tag),
                        ks.get_mask(tag=self.keyspaces.routing_tag))
                    for n, ks in iteritems(self.net_keyspaces)}

        routing_tables = routing_tree_to_tables(self.routes, net_keys)
        target_lengths = build_routing_table_target_lengths(system_info)
        routing_tables = minimise_tables(routing_tables, target_lengths)

        controller.load_routing_tables(routing_tables)

        # Assign memory to each vertex as required
        logger.debug("Assigning application memory")
        self.vertices_memory = sdram_alloc_for_vertices(
            controller, self.placements, self.allocations
        )

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
