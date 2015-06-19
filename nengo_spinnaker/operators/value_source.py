import collections
import math
import numpy as np
from six import iteritems
from rig.machine import Cores, SDRAM
import struct

from nengo.processes import Process
from nengo.utils import numpy as npext

from nengo_spinnaker.builder.builder import OutputPort, netlistspec
from nengo_spinnaker.netlist import VertexSlice
from nengo_spinnaker import partition_and_cluster as partition
from nengo_spinnaker import regions
from nengo_spinnaker.utils.application import get_application
from nengo_spinnaker.utils.type_casts import np_to_fix


class ValueSource(object):
    """Operator which transmits values from a buffer."""
    def __init__(self, function, size_out, period):
        """Create a new source which evaluates the given function over a period
        of time.
        """
        self.function = function
        self.size_out = size_out
        self.period = period

        # Vertices
        self.system_region = None
        self.keys_region = None
        self.vertices = list()

    def make_vertices(self, model, n_steps):
        """Create the vertices to be simulated on the machine."""
        # Create the system region
        self.system_region = SystemRegion(model.machine_timestep,
                                          self.period, n_steps)

        # Get all the outgoing signals to determine how big the size out is and
        # to build a list of keys.
        sigs_conns = model.get_signals_connections_from_object(self)
        if len(sigs_conns) == 0:
            return netlistspec([])

        keys = list()
        self.conns_transforms = list()
        for sig, conns in iteritems(sigs_conns[OutputPort.standard]):
            assert len(conns) == 1, "Expected a 1:1 mapping"

            # Add the keys for this connection
            conn = conns[0]
            transform, sig_keys = get_transform_keys(model, sig, conn)
            keys.extend(sig_keys)
            self.conns_transforms.append((conn, transform))
        size_out = len(keys)

        # Build the keys region
        self.keys_region = regions.KeyspacesRegion(
            keys, [regions.KeyField({"cluster": "cluster"})],
            partitioned_by_atom=True
        )

        # Create the output region
        self.output_region = regions.MatrixRegion(
            np.zeros((n_steps, size_out)),
            sliced_dimension=regions.MatrixPartitioning.columns
        )

        self.regions = [self.system_region, self.keys_region,
                        self.output_region]

        # Partition by output dimension to create vertices
        transmit_constraint = partition.Constraint(10)
        sdram_constraint = partition.Constraint(8*2**20)  # Max 8MiB
        constraints = {
            transmit_constraint: lambda s: s.stop - s.start,
            sdram_constraint:
                lambda s: regions.utils.sizeof_regions(self.regions, s),
        }
        for sl in partition.partition(slice(0, size_out), constraints):
            # Determine the resources
            resources = {
                Cores: 1,
                SDRAM: regions.utils.sizeof_regions(self.regions, sl),
            }
            vsl = VertexSlice(sl, get_application("value_source"), resources)
            self.vertices.append(vsl)

        # Return the vertices and callback methods
        return netlistspec(self.vertices, self.load_to_machine,
                           self.before_simulation)

    def load_to_machine(self, netlist, controller):
        """Load the values into memory."""
        # For each slice
        self.vertices_region_memory = collections.defaultdict(dict)

        for vertex in self.vertices:
            # Layout the slice of SDRAM we have been given
            region_memory = regions.utils.create_app_ptr_and_region_files(
                netlist.vertices_memory[vertex], self.regions, vertex.slice)

            # Store the location of each region in memory
            for region, mem in zip(self.regions, region_memory):
                self.vertices_region_memory[vertex][region] = mem

            # Write in some of the regions
            self.vertices_region_memory[vertex][self.system_region].seek(0)
            self.system_region.write_subregion_to_file(
                self.vertices_region_memory[vertex][self.system_region],
                vertex.slice
            )
            self.vertices_region_memory[vertex][self.keys_region].seek(0)
            self.keys_region.write_subregion_to_file(
                self.vertices_region_memory[vertex][self.keys_region],
                vertex.slice, cluster=vertex.cluster
            )

    def before_simulation(self, netlist, simulator, n_steps):
        """Generate the values to output for the next set of simulation steps.
        """
        # Write out the system region to deal with the current run-time
        self.system_region.n_steps = n_steps

        # Evaluate the node for this period of time
        if self.period is not None:
            max_n = min(n_steps, self.period / simulator.dt)
        else:
            max_n = n_steps

        ts = np.arange(simulator.steps, simulator.steps + max_n) * simulator.dt
        if callable(self.function):
            values = np.array([self.function(t) for t in ts])
        elif isinstance(self.function, Process):
            values = self.function.run_steps(max_n, d=self.size_out,
                                             dt=simulator.dt)
        else:
            values = np.array([self.function for t in ts])

        # Ensure that the values can be sliced, regardless of how they were
        # generated.
        values = npext.array(values, min_dims=2)

        # Compute the output for each connection
        outputs = []
        for conn, transform in self.conns_transforms:
            output = []

            # For each f(t) for the next set of simulations we calculate the
            # output at the end of the connection.  To do this we first apply
            # the pre-slice, then the function and then the post-slice.
            for v in values:
                # Apply the pre-slice
                v = v[conn.pre_slice]

                # Apply the function on the connection, if there is one.
                if conn.function is not None:
                    v = conn.function(v)

                output.append(np.dot(transform, v.T))
            outputs.append(np.array(output).reshape(n_steps, -1))

        # Combine all of the output values to form a large matrix which we can
        # dump into memory.
        output_matrix = np.hstack(outputs)

        new_output_region = regions.MatrixRegion(
            np_to_fix(output_matrix),
            sliced_dimension=regions.MatrixPartitioning.columns
        )

        # Write the simulation values into memory
        for vertex in self.vertices:
            self.vertices_region_memory[vertex][self.system_region].seek(0)
            self.system_region.write_subregion_to_file(
                self.vertices_region_memory[vertex][self.system_region],
                vertex.slice
            )

            self.vertices_region_memory[vertex][self.output_region].seek(0)
            new_output_region.write_subregion_to_file(
                self.vertices_region_memory[vertex][self.output_region],
                vertex.slice
            )


class SystemRegion(regions.Region):
    """System region for a value source."""
    def __init__(self, timestep, periodic, n_steps):
        # Store all the parameters
        self.timestep = timestep
        self.periodic = periodic
        self.n_steps = n_steps

    def sizeof(self, *args, **kwargs):
        return 4 * 6

    def write_subregion_to_file(self, fp, vertex_slice, **kwargs):
        """Write the region to a file-like."""
        # Determine the size out, frames per block, number of blocks and last
        # block length.
        size_out = vertex_slice.stop - vertex_slice.start
        frames_per_block = int(math.floor(20 * 1024 / (size_out * 4.0)))
        n_blocks = int(math.floor(self.n_steps / frames_per_block))
        last_block_length = self.n_steps % frames_per_block

        fp.write(struct.pack(
            "<6I", self.timestep, size_out, 0x1 if self.periodic else 0x0,
            n_blocks, frames_per_block, last_block_length
        ))


def get_transform_keys(model, sig, conn):
    # Get the transform for the connection from the list of built connections,
    # then remove zeroed rows (should any exist) and derive the list of keys.
    transform = model.params[conn].transform
    keep = np.any(transform != 0.0, axis=1)
    keys = list()

    for i, k in zip(range(transform.shape[0]), keep):
        if k:
            keys.append(sig.keyspace(index=i))

    # Return the transform and the list of keys
    return transform[keep], keys
