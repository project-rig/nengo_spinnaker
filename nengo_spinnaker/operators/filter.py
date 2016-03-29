import enum
import numpy as np
from rig.place_and_route import Cores, SDRAM
import struct

from nengo_spinnaker.builder.netlist import netlistspec
from nengo_spinnaker.builder.model import InputPort, OutputPort
from nengo_spinnaker import regions
from nengo_spinnaker.regions.utils import Args
from nengo_spinnaker.regions.filters import make_filter_regions
from nengo_spinnaker.netlist import Vertex
from nengo_spinnaker.utils.application import get_application
from nengo_spinnaker.utils.collections import flatinsertionlist
from nengo_spinnaker.utils.type_casts import np_to_fix

from ..builder.connection import (EnsembleTransmissionParameters,
                                  PassthroughNodeTransmissionParameters)

from nengo_spinnaker.partition import divide_slice


class Regions(enum.IntEnum):
    """Region names, corresponding to those used in `filter.c`"""
    system = 1
    keys = 2
    input_filters = 3
    input_routing = 4
    transform = 5


class Filter(object):
    """Operator which receives values, performs filtering, applies a linear
    transform and then forwards the resultant vector(s).

    The input and output vector(s) may be sufficiently large that the load
    related to receiving all the packets, filtering the input vector, applying
    the linear transform and transmitting the resultant values may be beyond
    the computational or communication capabilities of a single chip or core.
    The output vectors can be treated as a single large vector which is split
    into smaller vectors by transmitting each component with an appropriate
    key; hence we can consider the entire operation of the filter component as
    computing:

    ..math:: c[z] = \mathbf{A} b[z]

    Where **A** is the linear transform applied by the filter operator,
    :math:`b[z]` is the filtered input vector and :math:`c[z]` is the nominal
    output vector.

    If **A** is of size :math:`m \times n` then *n* determines how many packets
    each processing core (or group of processing cores) needs to receive and
    *m* determines how many packets each processing core (or group of cores)
    needs to transmit. To keep the number of packets received small we perform
    column-wise partition of A such that:

    ..math:: c[z] = \mathbf{A_1} b_1[z] + \mathbf{A_2} b_2[z]

    Where :math:`\mathbf{A_x} b_x[z]` is the product computed by one set of
    processing cores and :math:`c[z]` is the resultant vector as constructed by
    any cores which receive packets from cores implementing the filter
    operator. Note that the sum of products is computed by the receiving cores.
    **A** and `b` are now partitioned such that **A** is of size :math:`m
    \times (\frac{n}{2})` and `b` is of size :math:`\frac{n}{2}`; this reduces
    the number of packets that need to be received by any group of cores
    implementing the filter operator.

    To reduce the number of packets which need to be transmitted by each core
    we partition :math:`A_x` into rows such that:

    ..math::
        c =
        \begin{pmatrix}
          A_{1,1}~b_1 & + & A_{1,2}~b_2\\
          A_{2,1}~b_1 & + & A_{2,2}~b_2
        \end{pmatrix}

    Where, in this example, :math:`A_{x,y}` is of size :math:`\frac{m}{2}
    \times \frac{n}{2}`. Hence both the number of packets received and
    transmitted by each core has been halved, and the number of
    multiply-accumulates performed by each core has been quartered.  This
    reduction in communication and computation in the filter operator is
    achieved at the cost of requiring any operator with input `c` to receive
    twice as many packets as previously (one set of packets for each
    column-wise division) and to perform some additions.
    """
    def __init__(self, size_in, max_cols=128, max_rows=64):
        """Create a new parallel Filter.

        Parameters
        ----------
        size_in : int
            Width of the filter (length of any incoming signals).
        max_cols : int
        max_rows : int
            Maximum number of columns and rows which may be handled by a single
            processing core. The defaults (128 and 64 respectively) result in
            the overall connection matrix being decomposed such that (a) blocks
            are sufficiently small to be stored in DTCM, (b) network traffic is
            reduced.
        """
        # NB: max_rows and max_cols determined by experimentation by AM and
        # some modelling by SBF.
        # Create as many groups as necessary to keep the size in of any group
        # less than max_cols.
        self.size_in = size_in
        n_groups = (size_in // max_cols) + (1 if size_in % max_cols else 0)
        self.groups = tuple(FilterGroup(sl, max_rows) for sl in
                            divide_slice(slice(0, size_in), n_groups))

    def make_vertices(self, model, n_steps):
        """Make vertices for the filter."""
        # Get the complete matrix to be applied by the filter
        out_signals = model.get_signals_from_object(self)

        if OutputPort.standard not in out_signals:
            # If there are no outgoing signals then return no vertices
            return netlistspec(vertices=tuple())
        else:
            transform, keys, output_slices = \
                get_transforms_and_keys(out_signals[OutputPort.standard])

        # Get the filter and filter routing regions
        filter_region, filter_routing_region = make_filter_regions(
            model.get_signals_to_object(self)[InputPort.standard],
            model.dt, True,
            model.keyspaces.filter_routing_tag,
            width=self.size_in
        )

        # Generate the vertices
        vertices = flatinsertionlist()

        for group in self.groups:
            vertices.append(
                group.make_vertices(np_to_fix(transform),
                                    keys, output_slices,
                                    model.machine_timestep,
                                    filter_region,
                                    filter_routing_region)
            )

        # Return the netlist specification
        return netlistspec(vertices=vertices,
                           load_function=self.load_to_machine)

    def load_to_machine(self, netlist, controller):
        """Load the data to the machine."""
        # Get each group to load itself
        for g in self.groups:
            g.load_to_machine(netlist, controller)


class FilterGroup(object):
    """Portion of the columns of the transform applied by a filter, may extend
    across multiple chips.
    """
    def __init__(self, column_slice, max_rows):
        """Create a new group of filter cores.

        Parameters
        ----------
        column_slice : :py:class:`slice`
            Column-wise partition of the overall matrix that is assigned to
            this group of processing cores.
        """
        self.column_slice = column_slice
        self.size_in = column_slice.stop - column_slice.start
        self.max_rows = max_rows

    def make_vertices(self, transform, output_keys, output_slices,
                      machine_timestep, filter_region, filter_routing_region):
        """Partition the transform matrix into groups of rows and assign each
        group of rows to a core for computation.

        If the group needs to be split over multiple chips (i.e., the group is
        larger than 17 cores) then partition the matrix such that any used
        chips are used in their entirety.

        Parameters
        ----------
        transform : ndarray
            The complete (unpartitioned) transform applied by the filter.
        output_keys : [BitField, ...]
            Keys transmitted by filter.
        output_slices : [(TransmissionParameters, set), ...]
            Pairs of transmission parameters and sets containing the row
            indices of the transform matrix corresponding to the transmission
            parameters.
        """
        size_out = transform.shape[0]

        # Build as many vertices as required to keep the number of rows handled
        # by each core below max_rows.
        n_cores = (
            (size_out // self.max_rows) +
            (1 if size_out % self.max_rows else 0)
        )

        # Build the transform region for these cores
        transform_region = regions.MatrixRegion(
            transform[:, self.column_slice],
            sliced_dimension=regions.MatrixPartitioning.rows
        )

        # Build all the vertices
        self.cores = [
            FilterCore(self.column_slice, out_slice,
                       transform_region, output_keys, output_slices,
                       machine_timestep,
                       filter_region, filter_routing_region) for
            out_slice in divide_slice(slice(0, size_out), n_cores)
        ]

        return self.cores

    def load_to_machine(self, netlist, controller):
        """Allocate shared memory on each chip that we're using and load data
        to SDRAM for each core that we have allocated.
        """
        # Get each core to load itself
        for core in self.cores:
            core.load_to_machine(netlist)


class FilterCore(Vertex):
    """Portion of the rows of the transform assigned to a parallel filter
    group, represents the load assigned to a single processing core.
    """
    def __init__(self, column_slice, output_slice,
                 transform_region, output_keys, output_slices,
                 machine_timestep, filter_region, filter_routing_region):
        """Allocate a portion of the overall matrix to a single processing
        core.

        Parameters
        ----------
        column_slice : :py:class:`slice`
            Columns of the transform matrix managed by the group of vertices of
            which we are a member.
        output_slice : :py:class:`slice`
            Slice of the rows of the transform matrix that will be applied by
            this processing core.
        transform_region : MatrixRegion
        output_keys : [BitField, ...]
            Keys transmitted by filter.
        output_slices : [(TransmissionParameters, set), ...]
            Pairs of transmission parameters and sets containing the row
            indices of the transform matrix corresponding to the transmission
            parameters.
        """
        # Check that the output slice is safe
        assert (output_slice.start is not None and
                output_slice.stop is not None and
                (output_slice.step is None or output_slice.step == 1)
                )

        # Store information about the slices of the for which matrix we're
        # responsible.
        self.output_slice = output_slice
        self.column_slice = column_slice

        # Store which signal parameter slices we contain
        self.transmission_params = list()
        out_set = set(range(output_slice.start, output_slice.stop))
        for transmission_params, outs in output_slices:
            # If there is an intersection between the outs and the set of outs
            # we're responsible for then store transmission parameters.
            if out_set & outs:
                self.transmission_params.append(transmission_params)

        # Construct the regions
        self.regions = {
            Regions.system: SystemRegion(column_slice, output_slice,
                                         machine_timestep),
            Regions.transform: transform_region,
            Regions.keys: regions.KeyspacesRegion(
                output_keys,
                fields=[regions.KeyField(dict(cluster="cluster"))],
                partitioned_by_atom=True
            ),
            Regions.input_filters: filter_region,
            Regions.input_routing: filter_routing_region,
        }

        # Construct the region arguments
        w = self.column_slice.stop - self.column_slice.start
        self.region_arguments = {
            Regions.transform: Args(vertex_slice=self.output_slice),
            Regions.keys: Args(vertex_slice=self.output_slice),
            Regions.system: Args(),  # No arguments
            Regions.input_filters: Args(filter_width=w),  # No arguments
            Regions.input_routing: Args(),  # No arguments
        }

        # Determine the resource requirements and find the correct application
        sdram_usage = regions.utils.sizeof_regions_named(
            self.regions, self.region_arguments
        )

        super(FilterCore, self).__init__(
            application=get_application("filter"),
            resources={Cores: 1, SDRAM: sdram_usage}
        )

    def accepts_signal(self, signal_params, transmission_params):
        """Choose whether to receive this signal or not."""
        if isinstance(transmission_params, EnsembleTransmissionParameters):
            # If the connection is from an ensemble only return true if the
            # decoders contain non-zero values in the input dimensions we care
            # about.
            return np.any(transmission_params.decoders[self.column_slice, :])
        elif isinstance(transmission_params,
                        PassthroughNodeTransmissionParameters):
            # If the connection is from a Node of some variety then only return
            # true if the transform contains non-zero values in the rows which
            # relate to the subspace we receive input in.
            return np.any(transmission_params.transform[self.column_slice])

        # We don't know how to interpret the transmission parameters
        raise NotImplementedError

    def transmits_signal(self, signal_params, transmission_params):
        """Choose whether we transmit this signal or not."""
        return transmission_params in self.transmission_params

    def load_to_machine(self, netlist):
        """Write in the data for each region."""
        # Get a block of memory for each of the regions
        self.region_memory = \
            regions.utils.create_app_ptr_and_region_files_named(
                netlist.vertices_memory[self], self.regions,
                self.region_arguments
            )

        # Modify the region arguments
        self.region_arguments[Regions.keys].kwargs.update({
            "cluster": self.cluster})

        # Write the regions into memory
        for key in Regions:
            # Get the arguments
            args, kwargs = self.region_arguments[key]

            # Get the region
            self.regions[key].write_subregion_to_file(
                self.region_memory[key], *args, **kwargs
            )


class SystemRegion(object):
    """The system region of the `filter_parallel` operator.
    """
    def __init__(self, column_slice, output_slice, machine_timestep=1000):
        self.column_slice = column_slice
        self.output_slice = output_slice
        self.machine_timestep = machine_timestep

    def sizeof(self, *args, **kwargs):
        return 4 * 4

    sizeof_padded = sizeof

    def write_subregion_to_file(self, fp):
        """Write the subregion to file accounting for the input and output
        subspaces.
        """
        # Pack the data
        data = struct.pack("<4I",
                           self.machine_timestep,
                           self.column_slice.stop - self.column_slice.start,
                           self.column_slice.start,
                           self.output_slice.stop - self.output_slice.start)
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
