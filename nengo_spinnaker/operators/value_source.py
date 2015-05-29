import math
import numpy as np
from six import iteritems
from rig.machine import Cores, SDRAM
import struct

from nengo.processes import Process

from nengo_spinnaker.builder.builder import OutputPort, netlistspec
from nengo_spinnaker.netlist import VertexSlice
from nengo_spinnaker import partition_and_cluster as partition
from nengo_spinnaker import regions
from nengo_spinnaker.utils.application import get_application
from nengo_spinnaker.utils.keyspaces import get_derived_keyspaces
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
        # Evaluate the node for this period of time
        if self.period is not None:
            max_n = min(n_steps, self.period / model.dt)
        else:
            max_n = n_steps

        ts = np.arange(max_n) * model.dt
        if callable(self.function):
            values = np.array([self.function(t) for t in ts])
        elif isinstance(self.function, Process):
            values = self.function.run_steps(max_n, d=self.size_out, 
                                             dt=model.dt)
        else:
            values = np.array([self.function for t in ts])
        
        # Create the system region
        self.system_region = SystemRegion(model.machine_timestep,
                                          self.period, n_steps)

        # Get all the outgoing signals to determine how big the size out is and
        # to build a list of keys.
        keys = list()

        sigs_conns = model.get_signals_connections_from_object(self)

        if len(sigs_conns) == 0:
            return netlistspec([])

        outputs = []
        for sig, conns in iteritems(sigs_conns[OutputPort.standard]):
            assert len(conns) == 1, "Expected a 1:1 mapping"

            # Add the keys for this connection
            conn = conns[0]
            so = conns[0].post_obj.size_in
            keys.extend(list(
                get_derived_keyspaces(sig.keyspace, slice(0, so))
            ))

            # Compute the output for this connection
            output = []
            for v in values:
                if conn.function is not None:
                    v = conn.function(v)
                output.append(np.dot(conn.transform, v.T))
            outputs.append(np.array(output).reshape(n_steps, so))

        size_out = len(keys)
        output_matrix = np.hstack(outputs)
        assert output_matrix.shape == (n_steps, size_out), output_matrix.shape

        # Build the keys region
        self.keys_region = regions.KeyspacesRegion(
            keys, [regions.KeyField({"cluster": "cluster"})],
            partitioned_by_atom=True
        )

        # Create the output region
        self.output_region = regions.MatrixRegion(
            np_to_fix(output_matrix),
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
        return netlistspec(self.vertices, self.load_to_machine)

    def load_to_machine(self, netlist, controller):
        """Load the values into memory."""
        # For each slice
        for vertex in self.vertices:
            # Layout the slice of SDRAM we have been given
            region_memory = regions.utils.create_app_ptr_and_region_files(
                netlist.vertices_memory[vertex], self.regions, vertex.slice)

            # Write in each region
            for region, mem in zip(self.regions, region_memory):
                if region is self.keys_region:
                    self.keys_region.write_subregion_to_file(
                        mem, vertex.slice, cluster=vertex.cluster)
                else:
                    region.write_subregion_to_file(mem, vertex.slice)


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
