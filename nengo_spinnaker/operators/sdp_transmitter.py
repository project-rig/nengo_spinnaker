from rig.place_and_route import Cores, SDRAM
import hashlib
import struct

from nengo_spinnaker.builder.model import InputPort
from nengo_spinnaker.builder.netlist import netlistspec
from nengo_spinnaker.netlist import Vertex
from nengo_spinnaker.regions import Region
from nengo_spinnaker.regions.filters import make_filter_regions
from nengo_spinnaker.regions import utils as region_utils
from nengo_spinnaker.utils.application import get_application


class SDPTransmitter(object):
    """An operator which receives multicast packets, performs filtering and
    transmits the filtered vector as an SDP packet.
    """
    def __init__(self, size_in, iptag, label):
        self.size_in = size_in
        self.label = label
        self.iptag = 1 if iptag is None else iptag
        self._vertex = None
        self._sys_region = None
        self._filter_region = None
        self._routing_region = None

    def make_vertices(self, model, *args, **kwargs):
        """Create vertices that will simulate the SDPTransmitter."""
        # Build the system region
        self._sys_region = SystemRegion(model.machine_timestep,
                                        self.size_in, 1, self.iptag, self.label)

        # Build the filter regions
        in_sigs = model.get_signals_to_object(self)[InputPort.standard]
        self._filter_region, self._routing_region = make_filter_regions(
            in_sigs, model.dt, True, model.keyspaces.filter_routing_tag)

        # Get the resources
        resources = {
            Cores: 1,
            SDRAM: region_utils.sizeof_regions(
                [self._sys_region, self._filter_region, self._routing_region],
                None
            )
        }

        # Create the vertex
        self._vertex = Vertex(get_application("tx"), resources)

        # Return the netlist specification
        return netlistspec(self._vertex,
                           load_function=self.load_to_machine)

    def load_to_machine(self, netlist, controller):
        """Load data to the machine."""
        # Get the memory
        sys_mem, filter_mem, routing_mem = \
            region_utils.create_app_ptr_and_region_files(
                netlist.vertices_memory[self._vertex],
                [self._sys_region,
                 self._filter_region,
                 self._routing_region],
                None
            )

        # Write the regions into memory
        self._sys_region.write_region_to_file(sys_mem)
        self._filter_region.write_subregion_to_file(filter_mem)
        self._routing_region.write_subregion_to_file(routing_mem)


class SystemRegion(Region):
    """System region for SDP Tx."""
    def __init__(self, machine_timestep, size_in, delay, iptag, label):
        self.machine_timestep = machine_timestep
        self.size_in = size_in
        self.transmission_delay = delay
        self.iptag = iptag
        self.label_hash = (0 if label is None
                           else int(hashlib.md5(label).hexdigest()[:8], 16))

    def sizeof(self, *args, **kwargs):
        return 20

    def write_region_to_file(self, fp, *args, **kwargs):
        """Write the region to file."""
        fp.write(struct.pack("<5I", self.size_in, self.machine_timestep,
                             self.transmission_delay, self.iptag,
                             self.label_hash))
