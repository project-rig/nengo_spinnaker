import enum
import numpy as np
from rig.place_and_route import Cores, SDRAM
import struct

from nengo_spinnaker.builder.model import InputPort
from nengo_spinnaker.builder.netlist import netlistspec
from nengo_spinnaker import regions
from nengo_spinnaker.regions.utils import Args, sizeof_regions_named
from nengo_spinnaker.regions.filters import make_filter_regions
from nengo_spinnaker.netlist import Vertex
from nengo_spinnaker.partition import divide_slice
from nengo_spinnaker.utils.application import get_application

from ..builder.connection import (EnsembleTransmissionParameters,
                                  PassthroughNodeTransmissionParameters)


class Regions(enum.IntEnum):
    """Region names, corresponding to those used in `value_sink.c`"""
    system = 1
    filters = 2
    filter_routing = 3
    recording = 15


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
    def __init__(self, probe, dt, max_width=16):
        self.probe = probe
        self.size_in = probe.size_in
        self.max_width = max_width

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
        signals_conns = model.get_signals_to_object(self)[InputPort.standard]
        filter_region, filter_routing_region = make_filter_regions(
            signals_conns, model.dt, True, model.keyspaces.filter_routing_tag)

        # Make sufficient vertices to ensure that each has a size_in of less
        # than max_width.
        n_vertices = (
            (self.size_in // self.max_width) +
            (1 if self.size_in % self.max_width else 0)
        )
        self.vertices = tuple(
            ValueSinkVertex(model.machine_timestep, n_steps, sl, filter_region,
                            filter_routing_region) for sl in
            divide_slice(slice(0, self.size_in), n_vertices)
        )

        # Return the spec
        return netlistspec(self.vertices, self.load_to_machine,
                           after_simulation_function=self.after_simulation)

    def load_to_machine(self, netlist, controller):
        """Load the ensemble data into memory."""
        # Load each vertex in turn
        for v in self.vertices:
            v.load_to_machine(netlist)

    def after_simulation(self, netlist, simulator, n_steps):
        """Retrieve data from a simulation."""
        # Create an array into which to read probed values
        data = np.zeros((n_steps, self.size_in), dtype=np.float)

        # Read in the recorded results
        for v in self.vertices:
            data[:, v.input_slice] = v.read_recording(n_steps)

        # Apply the sampling
        data = data[::self.sample_every]

        # Store the probe data in the simulator
        if self.probe in simulator.data:
            # Include any existing probed data
            data = np.vstack((simulator.data[self.probe], data))
        simulator.data[self.probe] = data


class ValueSinkVertex(Vertex):
    def __init__(self, timestep, n_steps, input_slice,
                 filter_region, filter_routing_region):
        """Create a new vertex for a portion of a value sink."""
        self.input_slice = input_slice

        # Store the pre-existing regions and create new regions
        self.regions = {
            Regions.system: SystemRegion(timestep, input_slice),
            Regions.filters: filter_region,
            Regions.filter_routing: filter_routing_region,
            Regions.recording: regions.WordRecordingRegion(n_steps),
        }

        # Store region arguments
        w = input_slice.stop - input_slice.start
        self.region_arguments = {
            Regions.system: Args(),
            Regions.filters: Args(filter_width=w),
            Regions.filter_routing: Args(),
            Regions.recording: Args(input_slice),
        }

        # Determine resources usage
        resources = {
            Cores: 1,
            SDRAM: sizeof_regions_named(self.regions, self.region_arguments)
        }
        super(ValueSinkVertex, self).__init__(get_application("value_sink"),
                                              resources)

    def accepts_signal(self, signal_params, transmission_params):
        """Choose whether to receive this signal or not."""
        if isinstance(transmission_params, EnsembleTransmissionParameters):
            # If the connection is from an ensemble only return true if the
            # decoders contain non-zero values in the input dimensions we care
            # about.
            return np.any(transmission_params.decoders[self.input_slice, :])
        elif isinstance(transmission_params,
                        PassthroughNodeTransmissionParameters):
            # If the connection is from a Node of some variety then only return
            # true if the transform contains non-zero values in the rows which
            # relate to the subspace we receive input in.
            return np.any(transmission_params.transform[self.input_slice])

        # We don't know how to interpret the transmission parameters
        raise NotImplementedError

    def load_to_machine(self, netlist):
        # Get a block of memory for each of the regions
        self.region_memory = \
            regions.utils.create_app_ptr_and_region_files_named(
                netlist.vertices_memory[self], self.regions,
                self.region_arguments
            )

        # Write the regions into memory
        for key in Regions:
            # Get the arguments
            args, kwargs = self.region_arguments[key]

            # Get the region
            self.regions[key].write_subregion_to_file(
                self.region_memory[key], *args, **kwargs
            )

    def read_recording(self, n_steps):
        """Read back the recorded values."""
        # Grab the block of memory and seek to the start
        mem = self.region_memory[Regions.recording]
        mem.seek(0)

        # Perform the read
        return self.regions[Regions.recording].to_array(
            mem, self.input_slice, n_steps
        )


class SystemRegion(regions.Region):
    """System region for a value sink."""
    def __init__(self, timestep, input_slice):
        self.timestep = timestep
        self.input_slice = input_slice

    def sizeof(self, *args):
        return 12  # 3 words

    def write_subregion_to_file(self, fp, *args):
        size_in = self.input_slice.stop - self.input_slice.start
        fp.write(struct.pack("<3I", self.timestep,
                             size_in, self.input_slice.start))
