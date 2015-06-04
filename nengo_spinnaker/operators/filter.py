import numpy as np
from rig.machine import Cores, SDRAM
from six import iteritems
import struct

from nengo.utils.builder import full_transform

from nengo_spinnaker.builder.builder import InputPort, netlistspec, OutputPort
from nengo_spinnaker.regions.filters import make_filter_regions
from .. import regions
from nengo_spinnaker.netlist import Vertex
from nengo_spinnaker.utils.application import get_application
from nengo_spinnaker.utils.type_casts import np_to_fix


class Filter(object):
    """Operator which receives values, performs filter, applies and linear
    transform and then forwards the values.
    """
    def __init__(self, size_in, transmission_delay=1, interpacket_pause=1):
        """Create a new Filter.

        Parameters
        ----------
        size_in : int
            Width of the filter (length of any incoming signals).
        transmission_delay : int
            Number of machine timesteps to wait between transmitting packets.
        interpacket_pause : int
            Number of microseconds to leave between transmitting packets.
        """
        self.size_in = size_in
        self.transmission_delay = transmission_delay
        self.interpacket_pause = interpacket_pause

        # Internal objects
        self.vertex = None
        self.system_region = None
        self.filters_region = None
        self.routing_region = None
        self.output_keys_region = None
        self.transform_region = None
        self.regions = list()

    def make_vertices(self, model, n_steps):
        """Make a vertex for the operator."""
        # We don't partition Filters, so create the vertex and return it
        # Get the incoming filters
        incoming = model.get_signals_connections_to_object(self)
        self.filters_region, self.routing_region = make_filter_regions(
            incoming[InputPort.standard], model.dt, True,
            model.keyspaces.filter_routing_tag, width=self.size_in
        )

        # Create a combined output transform and set of keys
        outgoing = model.get_signals_connections_from_object(self)
        transform, output_keys = get_transforms_and_keys(
            outgoing[OutputPort.standard])

        size_out = len(output_keys)

        self.transform_region = regions.MatrixRegion(np_to_fix(transform))
        self.output_keys_region = regions.KeyspacesRegion(
            output_keys, fields=[regions.KeyField({'cluster': 'cluster'})]
        )

        # Create a system region
        self.system_region = SystemRegion(self.size_in, size_out,
                                          model.machine_timestep,
                                          self.transmission_delay,
                                          self.interpacket_pause)

        # Calculate the resources for the vertex
        self.regions = [
            self.system_region,
            self.output_keys_region,
            self.filters_region,
            self.routing_region,
            self.transform_region,
        ]
        resources = {
            Cores: 1,
            SDRAM: regions.utils.sizeof_regions(self.regions, slice(None)),
        }
        self.vertex = Vertex(get_application("filter"), resources)

        # Return the spec
        return netlistspec(self.vertex, self.load_to_machine)

    def load_to_machine(self, netlist, controller):
        """Load data to the machine."""
        # Get the memory
        region_mem = regions.utils.create_app_ptr_and_region_files(
            netlist.vertices_memory[self.vertex], self.regions, None)

        # Write the regions into memory
        for region, mem in zip(self.regions, region_mem):
            if region is not self.output_keys_region:
                region.write_subregion_to_file(mem, slice(None))
            else:
                region.write_subregion_to_file(mem, slice(None), cluster=0)


def get_transforms_and_keys(signals_connections):
    """Get a combined transform matrix and a list of keys to use to transmit
    elements transformed with the matrix.
    """
    transforms = list()
    keys = list()

    for signal, connections in iteritems(signals_connections):
        if len(connections) == 1:
            # Use the transform from the connection
            transform = full_transform(connections[0], allow_scalars=False)
        elif len(connections) == 0:
            # Use the identity matrix
            raise NotImplementedError
        else:
            # Can't do this
            raise NotImplementedError("Filters cannot transmit multiple "
                                      "Connections using the same signal.")

        if signal.latching:
            # If the signal is latching then we use the transform exactly as it
            # is.
            keep = np.array([True for _ in range(transform.shape[0])])
        else:
            # If the signal isn't latching then we remove rows which would
            # result in zero packets.
            keep = np.any(transform != 0.0, axis=1)

        transforms.append(transform[keep])
        for i, k in zip(range(transform.shape[0]), keep):
            if k:
                keys.append(signal.keyspace(index=i))

    # Combine all the transforms
    if len(transforms) > 0:
        transform = np.vstack(transforms)
    else:
        transform = np.array([[]])
    return transform, keys


class SystemRegion(regions.Region):
    def __init__(self, size_in, size_out, machine_timestep, transmission_delay,
                 interpacket_pause):
        self.size_in = size_in
        self.size_out = size_out
        self.machine_timestep = machine_timestep
        self.transmission_delay = transmission_delay
        self.interpacket_pause = interpacket_pause

    def sizeof(self, *args):
        """Return the size of the region in bytes."""
        return 5 * 4

    def write_subregion_to_file(self, fp, *args, **kwargs):
        """Write the region to file."""
        fp.write(struct.pack(
            "<5I", self.size_in, self.size_out, self.machine_timestep,
            self.transmission_delay, self.interpacket_pause
        ))
