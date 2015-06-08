import numpy as np
from rig.machine import Cores, SDRAM
import struct

from nengo_spinnaker.builder.builder import InputPort, netlistspec
from nengo_spinnaker import regions
from nengo_spinnaker.regions.filters import make_filter_regions
from nengo_spinnaker.netlist import Vertex
from nengo_spinnaker.utils.application import get_application
from nengo_spinnaker.utils.type_casts import fix_to_np


class ValueSink(object):
    """Operator which receives and stores values across the SpiNNaker multicast
    network.

    Attributes
    ----------
    probe : Probe
        Nengo probe that is updated by this operator.
    size_in : int
        Number of packets to receive and store per timestep.
    sample_every : int
        Number of machine timesteps between taking samples.
    """
    def __init__(self, probe, dt):
        self.probe = probe
        self.size_in = probe.size_in

        # Compute the sample period
        if probe.sample_every is None:
            self.sample_every = 1
        else:
            self.sample_every = int(np.round(probe.sample_every / dt))

    def make_vertices(self, model, n_steps):  # TODO remove n_steps
        """Construct the data which can be loaded into the memory of a
        SpiNNaker machine.
        """
        # Extract all the filters from the incoming connections to build the
        # filter regions.
        signals_conns = \
            model.get_signals_connections_to_object(self)[InputPort.standard]
        self.filter_region, self.filter_routing_region = make_filter_regions(
            signals_conns, model.dt, True, model.keyspaces.filter_routing_tag)

        # Use a matrix region to record into (slightly unpleasant)
        self.recording_region = regions.MatrixRegion(
            np.zeros((self.size_in, n_steps), dtype=np.uint32)
        )

        # This isn't partitioned, so we just compute the SDRAM requirement and
        # return a new vertex.
        self.system_region = SystemRegion(model.machine_timestep, self.size_in)

        self.regions = [None] * 15
        self.regions[0] = self.system_region
        self.regions[1] = self.filter_region
        self.regions[2] = self.filter_routing_region
        self.regions[14] = self.recording_region  # **YUCK**
        resources = {
            Cores: 1,
            SDRAM: regions.utils.sizeof_regions(self.regions, None)
        }

        self.vertex = Vertex(get_application("value_sink"), resources)

        # Return the spec
        return netlistspec(self.vertex, self.load_to_machine,
                           after_simulation_function=self.after_simulation)

    def load_to_machine(self, netlist, controller):
        """Load the ensemble data into memory."""
        # Assign SDRAM for each memory region and create the application
        # pointer table.
        region_memory = regions.utils.create_app_ptr_and_region_files(
            netlist.vertices_memory[self.vertex], self.regions, None)

        # Write in each region
        for region, mem in zip(self.regions[:3], region_memory):
            if region is not None:
                region.write_subregion_to_file(mem, slice(None))

        # Store the location of the recording region
        self.recording_region_mem = region_memory[14]

    def after_simulation(self, netlist, simulator, n_steps):
        """Retrieve data from a simulation."""
        self.recording_region_mem.seek(0)
        recorded_data = fix_to_np(np.frombuffer(
            self.recording_region_mem.read(n_steps * self.size_in * 4),
            dtype=np.int32)).reshape(n_steps, self.size_in)

        if self.probe not in simulator.data:
            simulator.data[self.probe] = recorded_data
        else:
            full_data = np.vstack([simulator.data[self.probe], recorded_data])
            simulator.data[self.probe] = full_data


class SystemRegion(regions.Region):
    """System region for a value sink."""
    def __init__(self, timestep, size_in):
        self.timestep = timestep
        self.size_in = size_in

    def sizeof(self, *args):
        return 8  # 2 words

    def write_subregion_to_file(self, fp, *args):
        fp.write(struct.pack("<2I", self.timestep, self.size_in))
