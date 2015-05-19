import collections
from rig.machine import Cores, SDRAM
import six
import struct

from nengo_spinnaker.builder.builder import netlistspec, InputPort
from nengo_spinnaker.netlist import Vertex
from nengo_spinnaker.regions import Region
from nengo_spinnaker.regions.filters import make_filter_regions
from nengo_spinnaker.regions import utils as region_utils
from nengo_spinnaker.utils.application import get_application


class SDPTransmitter(object):
    """An operator which receives multicast packets, performs filtering and
    transmits the filtered vector as an SDP packet.
    """
    def __init__(self):
        # Store a map of Nodes to vertices and vertices to regions
        self.nodes_vertices = dict()
        self._sys_regions = dict()  # Vertex to system region
        self._filter_regions = dict()
        self._routing_regions = dict()

    def make_vertices(self, model, *args, **kwargs):
        """Create vertices that will simulate the SDPTransmitter."""
        # Group the incoming signals and connections by the post-object.
        nodes_sigs_conns = collections.defaultdict(dict)
        in_ = model.get_signals_connections_from_object(self)
        for signal, connections in six.iteritems(in_[InputPort.standard]):
            for conn in connections:
                nodes_sigs_conns[conn.post_obj][signal].append(conn)

        # For each Node create a vertex
        for node, signals_connections in six.iteritems(nodes_sigs_conns):
            # Build the system region
            sys_region = SystemRegion(model.machine_timestep, node.size_in, 1)

            # Build the filter regions
            filter_region, routing_region = make_filter_regions(
                signals_connections, model.dt, True,
                model.keyspaces.filter_routing_tag
            )

            # Get the resources
            resources = {
                Cores: 1,
                SDRAM: region_utils.sizeof_regions(
                    [sys_region, filter_region, routing_region], None
                )
            }

            # Create the vertex
            v = self.nodes_vertices[conn] = Vertex(get_application("tx"),
                                                   resources)
            self._sys_regions[v] = sys_region
            self._filter_regions[v] = filter_region
            self._routing_regions[v] = routing_region

        # Return the netlist specification
        return netlistspec(list(self.nodes_vertices.values()),
                           load_function=self.load_to_machine)

    def load_to_machine(self, netlist, controller):
        """Load data to the machine."""
        for vx in six.itervalues(self.nodes_vertices):
            # Grab the regions for the vertex
            sys_region = self._sys_regions[vx]
            filter_region = self._filter_regions[vx]
            routing_region = self._routing_regions[vx]

            # Get the memory
            sys_mem, filter_mem, routing_mem = \
                region_utils.create_app_ptr_and_region_files(
                    netlist.vertices_memory[vx],
                    [sys_region, filter_region, routing_region],
                    None
                )

            # Write the regions into memory
            sys_region.write_region_to_file(sys_mem)
            filter_region.write_subregion_to_file(filter_mem)
            routing_region.write_subregion_to_file(routing_mem)


class SystemRegion(Region):
    """System region for SDP Tx."""
    def __init__(self, machine_timestep, size_in, delay):
        self.machine_timestep = machine_timestep
        self.size_in = size_in
        self.transmission_delay = delay

    def sizeof(self, *args, **kwargs):
        return 12

    def write_region_to_file(self, fp, *args, **kwargs):
        """Write the region to file."""
        fp.write(struct.pack("<3I", self.size_in, self.machine_timestep,
                             self.transmission_delay))
