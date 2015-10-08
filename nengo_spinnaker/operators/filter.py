import collections
import numpy as np
from rig.place_and_route import Cores, SDRAM
from rig.place_and_route.constraints import SameChipConstraint
from six import iteritems
import struct

from nengo_spinnaker.builder.model import InputPort, OutputPort
from nengo_spinnaker import regions
from nengo_spinnaker.builder.netlist import netlistspec
from nengo_spinnaker.regions.filters import make_filter_regions
from nengo_spinnaker.netlist import Vertex
from nengo_spinnaker.utils.application import get_application
from nengo_spinnaker.utils.type_casts import np_to_fix

from ..builder.connection import (EnsembleTransmissionParameters,
                                  PassthroughNodeTransmissionParameters)

from nengo_spinnaker.partition import divide_slice


class Filter(object):
    """Operator which receives values, performs filter, applies and linear
    transform and then forwards the values.
    """
    def __init__(self, size_in, n_cores_per_chip=None, n_chips=None):
        """Create a new parallel Filter.

        Parameters
        ----------
        size_in : int
            Width of the filter (length of any incoming signals).
        n_cores_per_chip : int or None
            Number of cores to use per chip to simulate the filter. If None
            then a sensible value is chosen.
        n_chips : int or None
            Number of chips to use. If None then a sensible value is chosen.
        """
        self.size_in = size_in
        self.n_cores_per_chip = n_cores_per_chip
        self.n_chips = n_chips

        # Internal objects
        self.vertices = list()
        self.system_region = None
        self.filters_region = None
        self.routing_region = None
        self.output_keys_region = None
        self.transform_region = None

    def make_vertices(self, model, n_steps):
        """Make vertices for the filter."""
        # Get the outgoing transforms and keys
        sigs = model.get_signals_from_object(self)
        if OutputPort.standard in sigs:
            outgoing = sigs[OutputPort.standard]
            transform, output_keys, sigs_pars_slices = \
                get_transforms_and_keys(outgoing)
        else:
            transform = np.array([[]])
            output_keys = list()
            sigs_pars_slices = list()

        size_out = len(output_keys)

        # Calculate how many cores and chips to use.
        if self.n_cores_per_chip is None or self.n_chips is None:
            # The number of cores is largely a function of the input size, we
            # try to ensure that each core is receiving a max of 32 packets per
            # timestep.
            n_cores_per_chip = int(min(16, np.ceil(self.size_in / 32.0)))

            # The number of chips is now determined by the size in (columns in
            # the transform matrix), the size out (rows in the transform
            # matrix) and the number of cores per chip.
            n_chips = self.n_chips or 1
            n_cores = n_chips * n_cores_per_chip

            while True:
                rows_per_core = int(np.ceil(float(size_out) /
                                            (n_cores * n_chips)))
                load_per_core = rows_per_core * self.size_in

                # The 8,000 limits the number of columns in each row that we
                # need to process. This is a heuristic.
                if load_per_core <= 8000 or n_chips > 9:
                    # The load per core is acceptable or we're using way too
                    # many chips
                    break

                if n_cores < 16:
                    # Increase the number of cores per chip if we can
                    n_cores += 1
                else:
                    # Otherwise increase the number of chips
                    n_chips += 1

            # Store the result
            self.n_cores_per_chip = n_cores
            self.n_chips = n_chips

        # Slice the input space into the given number of subspaces, this is
        # repeated on each chip.
        input_slices = list(divide_slice(slice(0, self.size_in),
                                         self.n_cores_per_chip))

        # Slice the output space into the given number of subspaces, this is
        # sliced across all of the chips.
        output_slices = divide_slice(slice(0, size_out),
                                     self.n_cores_per_chip * self.n_chips)

        # Construct the output keys and transform regions; the output keys and
        # sliced, and the transform is sliced by rows.
        self.output_keys_region = regions.KeyspacesRegion(
            output_keys, fields=[regions.KeyField({'cluster': 'cluster'})],
            partitioned_by_atom=True
        )
        self.transform_region = regions.MatrixRegion(
            np_to_fix(transform),
            sliced_dimension=regions.MatrixPartitioning.rows
        )

        # Construct the system region
        self.system_region = SystemRegion(self.size_in, model.machine_timestep)

        # Get the incoming filters
        incoming = model.get_signals_to_object(self)
        self.filters_region, self.routing_region = make_filter_regions(
            incoming[InputPort.standard], model.dt, True,
            model.keyspaces.filter_routing_tag, width=self.size_in
        )

        # Make the vertices and constraints
        iter_output_slices = iter(output_slices)
        cons = list()  # List of constraints

        # For each chip that we'll be using
        for _ in range(self.n_chips):
            chip_vertices = list()

            # Each core is given an input slice and an output slice.  The same
            # set of input slices is used per chip, but we iterate through the
            # whole list of output slices.
            for in_slice, out_slice in zip(input_slices,
                                           iter_output_slices):
                # Determine the amount of SDRAM required (the 24 additional
                # bytes are for the application pointer table).  We also
                # include this cores contribution to a shared SDRAM vector.
                sdram = (24 + 4*(in_slice.stop - in_slice.start) +
                         self.system_region.sizeof() +
                         self.filters_region.sizeof_padded() +
                         self.routing_region.sizeof_padded() +
                         self.output_keys_region.sizeof_padded(out_slice) +
                         self.transform_region.sizeof_padded(out_slice))

                # Create the vertex and include in the list of vertices
                v = ParallelFilterSlice(in_slice, out_slice,
                                        {Cores: 1, SDRAM: sdram},
                                        sigs_pars_slices)
                chip_vertices.append(v)
                self.vertices.append(v)

            # Create a constraint which will force all of the vertices to exist
            # of the same chip.
            cons.append(SameChipConstraint(chip_vertices))

        # Return the spec
        return netlistspec(self.vertices, self.load_to_machine,
                           constraints=cons)

    def load_to_machine(self, netlist, controller):
        """Load the data to the machine."""
        # Grab the placements for the vertices
        placements = collections.defaultdict(list)
        for v in self.vertices:
            placements[netlist.placements[v]].append(v)

        # For safety, check that there are only as many unique placements as
        # chips we intend to use.
        assert len(placements) == self.n_chips

        # Group the regions in order so that the application pointer table can
        # be constructed.
        _regions = (self.system_region, self.output_keys_region,
                    self.filters_region, self.routing_region,
                    self.transform_region)

        for (x, y), vertices in iteritems(placements):
            # Allocate a block of SDRAM to store the shared input vector, zero
            # this buffer.
            shared_vec = controller.sdram_alloc_as_filelike(self.size_in * 4,
                                                            x=x, y=y)
            self.system_region.shared_vector_address = shared_vec.address
            shared_vec.write(b'\x00' * self.size_in * 4)

            # For each vertex, construct the application pointer table and
            # write in the data from the regions.
            for vertex in vertices:
                # Create the application pointer and associate memory with each
                # region
                region_mem = regions.utils.create_app_ptr_and_region_files(
                    netlist.vertices_memory[vertex], _regions,
                    vertex.out_slice
                )

                # Write in the system region
                self.system_region.write_subregion_to_file(
                    region_mem[0], vertex.in_slice, vertex.out_slice)

                # The output keys
                self.output_keys_region.write_subregion_to_file(
                    region_mem[1], vertex.out_slice, cluster=vertex.cluster)

                # Input filters
                self.filters_region.write_subregion_to_file(region_mem[2])

                # Filter routing
                self.routing_region.write_subregion_to_file(region_mem[3])

                # Transform matrix
                self.transform_region.write_subregion_to_file(
                    region_mem[4], vertex.out_slice)


class ParallelFilterSlice(Vertex):
    """Represents a portion of a parallel filter.

    Attributes
    ----------
    in_slice : :py:class:`slice`
        Slice of the input space managed by this vertex.
    out_slice : :py:class:`slice`
        Slice of the output space managed by this vertex.
    """
    def __init__(self, in_slice, out_slice, resources=dict(),
                 transmission_parameter_slices=list()):
        super(ParallelFilterSlice, self).__init__(get_application("filter"),
                                                  resources)

        # Store the slices
        self.in_slice = in_slice
        self.out_slice = out_slice

        # Store which signal parameter slices we contain
        self.transmission_params = list()
        out_set = set(range(out_slice.start or 0,
                            out_slice.stop or 0,
                            out_slice.step or 1))
        for transmission_params, outs in transmission_parameter_slices:
            # If there is an intersection between the outs and the set of outs
            # we're responsible for then store transmission parameters.
            if out_set & outs:
                self.transmission_params.append(transmission_params)

    def accepts_signal(self, signal_params, transmission_params):
        """Choose whether to receive this signal or not."""
        if isinstance(transmission_params, EnsembleTransmissionParameters):
            # If the connection is from an ensemble only return true if the
            # decoders contain non-zero values in the input dimensions we care
            # about.
            return np.any(transmission_params.decoders[self.in_slice, :])
        elif isinstance(transmission_params,
                        PassthroughNodeTransmissionParameters):
            # If the connection is from a Node of some variety then only return
            # true if the transform contains non-zero values in the rows which
            # relate to the subspace we receive input in.
            return np.any(transmission_params.transform[self.in_slice])

        # We don't know how to interpret the transmission parameters
        raise NotImplementedError

    def transmits_signal(self, signal_params, transmission_params):
        """Choose whether we transmit this signal or not."""
        return transmission_params in self.transmission_params


class SystemRegion(object):
    """The system region of the `filter_parallel` operator.

    Attributes
    ----------
    n_dims : int
        Total size of the input vector
    """
    def __init__(self, n_dims, machine_timestep=1000):
        self.n_dims = n_dims
        self.machine_timestep = machine_timestep
        self.shared_vector_address = None

    def sizeof(self, *args, **kwargs):
        """System region is always 6 words long."""
        return 6 * 4

    sizeof_padded = sizeof

    def write_subregion_to_file(self, fp, in_slice, out_slice):
        """Write the subregion to file accounting for the input and output
        subspaces.
        """
        # The address of the shared vector must be provided
        assert self.shared_vector_address is not None

        # Pack the data
        data = struct.pack("<6I", self.machine_timestep, self.n_dims,
                           in_slice.start, in_slice.stop - in_slice.start,
                           out_slice.stop - out_slice.start,
                           self.shared_vector_address)

        fp.write(data)


def get_transforms_and_keys(signals_connections):
    """Get a combined transform matrix and a list of keys to use to transmit
    elements transformed with the matrix.  This method also returns a list of
    signal parameters, transmission parameters and the slice of the final
    transform matrix that they are associated with.
    """
    transforms = list()
    keys = list()
    slices = list()

    start = end = 0
    for signal, transmission_params in signals_connections:
        # Extract the transform
        transform = transmission_params.transform

        if signal.latching:
            # If the signal is latching then we use the transform exactly as it
            # is.
            keep = np.array([True for _ in range(transform.shape[0])])
        else:
            # If the signal isn't latching then we remove rows which would
            # result in zero packets.
            keep = np.any(transform != 0.0, axis=1)

        transforms.append(transform[keep])
        end += transforms[-1].shape[0]

        slices.append((transmission_params, set(range(start, end))))
        start = end

        for i, k in zip(range(transform.shape[0]), keep):
            if k:
                keys.append(signal.keyspace(index=i))

    # Combine all the transforms
    if len(transforms) > 0:
        transform = np.vstack(transforms)
    else:
        transform = np.array([[]])
    return transform, keys, slices
