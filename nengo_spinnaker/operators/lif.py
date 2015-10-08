"""LIF Ensemble

Takes an intermediate representation of a LIF ensemble and returns a vertex and
appropriate callbacks to load and prepare the ensemble for simulation on
SpiNNaker.  The build method also manages the partitioning of the ensemble into
appropriate sized slices.
"""

import collections
import enum
from nengo.base import ObjView
import numpy as np
from rig.place_and_route import Cores, SDRAM
from rig.place_and_route.constraints import SameChipConstraint
import struct

from nengo_spinnaker.builder.model import InputPort, OutputPort
from nengo_spinnaker.builder.netlist import netlistspec
from nengo_spinnaker.builder.ports import EnsembleInputPort
from nengo_spinnaker.regions.filters import make_filter_regions
from .. import regions
from nengo_spinnaker.netlist import Vertex
from nengo_spinnaker import partition
from nengo_spinnaker.utils.application import get_application
from nengo_spinnaker.utils.config import getconfig
from nengo_spinnaker.utils import type_casts as tp
from nengo_spinnaker.utils import neurons as neuron_utils


class EnsembleRegions(enum.IntEnum):
    """Region names, corresponding to those defined in `ensemble.h`"""
    ensemble = 1  # General ensemble settings
    neuron = 2  # Neuron specific parameters
    encoders = 3  # Encoder matrix (for neurons on core)
    bias = 4  # Biases
    gain = 5  # Gains
    decoders = 6  # Decoder matrix (for all neurons, only some rows)
    keys = 7  # Output keys
    population_length = 8  # Information about the entire cluster
    input_filters = 9
    input_routing = 10
    inhibition_filters = 11
    inhibition_routing = 12
    profiler = 13
    spikes = 14
    voltages = 15


class EnsembleLIF(object):
    """Controller for an ensemble of LIF neurons."""
    def __init__(self, ensemble):
        """Create a new LIF ensemble controller."""
        self.ensemble = ensemble
        self.direct_input = np.zeros(ensemble.size_in)
        self.local_probes = list()

        self.profiled = False
        self.record_spikes = False
        self.record_voltages = False

    def make_vertices(self, model, n_steps):
        """Construct the data which can be loaded into the memory of a
        SpiNNaker machine.
        """
        # Build encoders, gain and bias regions
        params = model.params[self.ensemble]
        ens_regions = dict()

        # Convert the encoders combined with the gain to S1615 before creating
        # the region.
        encoders_with_gain = params.scaled_encoders
        ens_regions[EnsembleRegions.encoders] = regions.MatrixRegion(
            tp.np_to_fix(encoders_with_gain),
            sliced_dimension=regions.MatrixPartitioning.rows)

        # Combine the direct input with the bias before converting to S1615 and
        # creating the region.
        bias_with_di = params.bias + np.dot(encoders_with_gain,
                                            self.direct_input)
        assert bias_with_di.ndim == 1
        ens_regions[EnsembleRegions.bias] = regions.MatrixRegion(
            tp.np_to_fix(bias_with_di),
            sliced_dimension=regions.MatrixPartitioning.rows)

        # Convert the gains to S1615 before creating the region
        ens_regions[EnsembleRegions.gain] = regions.MatrixRegion(
            tp.np_to_fix(params.gain),
            sliced_dimension=regions.MatrixPartitioning.rows)

        # Extract all the filters from the incoming connections
        incoming = model.get_signals_to_object(self)

        (ens_regions[EnsembleRegions.input_filters],
         ens_regions[EnsembleRegions.input_routing]) = make_filter_regions(
            incoming[InputPort.standard], model.dt, True,
            model.keyspaces.filter_routing_tag,
            width=self.ensemble.size_in
        )
        (ens_regions[EnsembleRegions.inhibition_filters],
         ens_regions[EnsembleRegions.inhibition_routing]) = \
            make_filter_regions(
                incoming[EnsembleInputPort.global_inhibition], model.dt, True,
                model.keyspaces.filter_routing_tag, width=1
            )

        # Extract all the decoders for the outgoing connections and build the
        # regions for the decoders and the regions for the output keys.
        outgoing = model.get_signals_from_object(self)
        if OutputPort.standard in outgoing:
            decoders, output_keys = \
                get_decoders_and_keys(outgoing[OutputPort.standard], True)
        else:
            decoders = np.array([])
            output_keys = list()
        size_out = decoders.shape[0]

        ens_regions[EnsembleRegions.decoders] = regions.MatrixRegion(
            tp.np_to_fix(decoders / model.dt),
            sliced_dimension=regions.MatrixPartitioning.rows)
        ens_regions[EnsembleRegions.keys] = regions.KeyspacesRegion(
            output_keys,
            fields=[regions.KeyField({'cluster': 'cluster'})],
            partitioned_by_atom=True
        )

        # The population length region stores information about groups of
        # co-operating cores.
        ens_regions[EnsembleRegions.population_length] = \
            regions.ListRegion("I")

        # The ensemble region contains basic information about the ensemble
        ens_regions[EnsembleRegions.ensemble] = EnsembleRegion(
            model.machine_timestep, self.ensemble.size_in)

        # The neuron region contains information specific to the neuron type
        ens_regions[EnsembleRegions.neuron] = LIFRegion(
            model.dt, self.ensemble.neuron_type.tau_rc,
            self.ensemble.neuron_type.tau_ref
        )

        # Manage profiling
        n_profiler_samples = 0
        self.profiled = getconfig(model.config, self.ensemble, "profile",
                                  False)
        if self.profiled:
            # Try and get number of samples from config
            n_profiler_samples = getconfig(model.config, self.ensemble,
                                           "profile_num_samples")

            # If it's not specified, calculate sensible default
            if n_profiler_samples is None:
                n_profiler_samples = (len(EnsembleSlice.profiler_tag_names) *
                                      n_steps * 2)

        # Create profiler region
        ens_regions[EnsembleRegions.profiler] = regions.Profiler(
            n_profiler_samples)
        ens_regions[EnsembleRegions.ensemble].n_profiler_samples = \
            n_profiler_samples

        # Manage probes
        for probe in self.local_probes:
            if probe.attr in ("output", "spikes"):
                self.record_spikes = True
            elif probe.attr == "voltage":
                self.record_voltages = True
            else:
                raise NotImplementedError(
                    "Cannot probe {} on Ensembles".format(probe.attr)
                )

        # Set the flags
        ens_regions[EnsembleRegions.ensemble].record_spikes = \
            self.record_spikes
        ens_regions[EnsembleRegions.ensemble].record_voltages = \
            self.record_voltages

        # Create the probe recording regions
        ens_regions[EnsembleRegions.spikes] = regions.SpikeRecordingRegion(
            n_steps if self.record_spikes else 0)
        ens_regions[EnsembleRegions.voltages] = regions.VoltageRecordingRegion(
            n_steps if self.record_voltages else 0)

        # Create constraints against which to partition, initially assume that
        # we can devote 16 cores to every problem.
        sdram_constraint = partition.Constraint(128 * 2**20,
                                                0.9)  # 90% of 128MiB
        dtcm_constraint = partition.Constraint(16 * 64 * 2**10,
                                               0.9)  # 90% of 16 cores DTCM

        # The number of cycles available is 200MHz * the machine timestep; or
        # 200 * the machine timestep in microseconds.
        cycles = 200 * model.machine_timestep
        cpu_constraint = partition.Constraint(cycles * 16,
                                              0.8)  # 80% of 16 cores compute

        # Form the constraints dictionary
        def _make_constraint(f, size_in, size_out, **kwargs):
            """Wrap a usage computation method to work with the partitioner."""
            def f_(vertex_slice):
                # Calculate the number of neurons
                n_neurons = vertex_slice.stop - vertex_slice.start

                # Call the original method
                return f(size_in, size_out, n_neurons, **kwargs)
            return f_

        partition_constraints = {
            sdram_constraint: _make_constraint(_lif_sdram_usage,
                                               self.ensemble.size_in,
                                               size_out),
            dtcm_constraint: _make_constraint(_lif_dtcm_usage,
                                              self.ensemble.size_in, size_out),
            cpu_constraint: _make_constraint(_lif_cpu_usage,
                                             self.ensemble.size_in, size_out),
        }

        # Partition the ensemble to create clusters of co-operating cores
        self.clusters = list()
        vertices = list()
        constraints = list()
        for sl in partition.partition(slice(0, self.ensemble.n_neurons),
                                      partition_constraints):
            # For each slice we create a cluster of co-operating cores.  We
            # instantiate the cluster and then ask it to produce vertices which
            # will be added to the netlist.
            cluster = EnsembleCluster(sl, self.ensemble.size_in, size_out,
                                      ens_regions)
            self.clusters.append(cluster)

            # Get the vertices for the cluster
            cluster_vertices = cluster.make_vertices(cycles)
            vertices.extend(cluster_vertices)

            # Create a constraint which forces these vertices to be present on
            # the same chip
            constraints.append(SameChipConstraint(cluster_vertices))

        # Return the vertices and callback methods
        return netlistspec(vertices, self.load_to_machine,
                           after_simulation_function=self.after_simulation,
                           constraints=constraints)

    def load_to_machine(self, netlist, controller):
        """Load the ensemble data into memory."""
        # Delegate the task of loading to the machine
        for cluster in self.clusters:
            cluster.load_to_machine(netlist, controller)

    def after_simulation(self, netlist, simulator, n_steps):
        # If profiling is enabled then get the profiler data
        if self.profiled:
            # Get all the profiler data, this will be dictionary mapping
            # (neurons.start, neurons.stop) to the data returned by the
            # profiler.
            simulator.profiler_data[self.ensemble] = dict()
            for cl in self.clusters:
                simulator.profiler_data[self.ensemble].update(dict(
                    cl.get_profiler_data()
                ))

        # Retrieve probe data
        # If spikes were recorded then get the spikes
        if self.record_spikes:
            # Create an empty matrix of the correct size
            spikes = np.zeros((n_steps, self.ensemble.n_neurons),
                              dtype=np.bool)

            # For each cluster read back the spike data
            for cl in self.clusters:
                # For each neuron slice copy in the spike data
                for neurons, data in cl.get_spike_data(n_steps):
                    spikes[:, neurons] = data

            # Recast the data as floats
            spike_vals = np.zeros((n_steps, self.ensemble.n_neurons))
            spike_vals[spikes] = 1.0 / simulator.dt

        # If voltages were recorded then get the voltages
        if self.record_voltages:
            # Create an empty matrix of the correct size
            voltages = np.zeros((n_steps, self.ensemble.n_neurons))

            # For each cluster read back the voltage data
            for cl in self.clusters:
                # For each neuron slice copy in the voltage data
                for neurons, data in cl.get_voltage_data(n_steps):
                    voltages[:, neurons] = data

        # Store the data associated with probes
        for p in self.local_probes:
            # Get the neuron slice applied by the probe
            neuron_slice = slice(None)
            if isinstance(p.target, ObjView):
                neuron_slice = p.target.slice

            # Get the temporal slicing applied by the probe
            sample_every = 1
            if p.sample_every is not None:
                sample_every = int(p.sample_every / simulator.dt)

            # Get the probe data
            if p.attr in ("output", "spikes"):
                # Spike data
                probe_data = spike_vals[::sample_every, neuron_slice]
            elif p.attr == "voltage":
                # Voltage data
                probe_data = voltages[::sample_every, neuron_slice]

            # Store the probe data
            if p in simulator.data:
                # Append the new probe data to the existing probe data
                probe_data = np.vstack((simulator.data[p], probe_data))
            simulator.data[p] = probe_data


class EnsembleCluster(object):
    def __init__(self, neuron_slice, size_in, size_out, regions):
        """Create a new cluster of collaborating cores."""
        self.neuron_slice = neuron_slice
        self.regions = regions
        self.neuron_slices = list()
        self.vertices = list()
        self.size_in = size_in
        self.size_out = size_out

    def make_vertices(self, cycles):
        """Partition the neurons onto multiple cores."""
        # Make reduced constraints to partition against, we don't partition
        # against SDRAM as we're already sure that there is sufficient SDRAM
        # (and if there isn't we can't possibly fit all the vertices on a
        # single chip).
        dtcm_constraint = partition.Constraint(64 * 2**10, 0.9)  # 90% of DTCM
        cpu_constraint = partition.Constraint(cycles, 0.8)  # 80% of compute

        # Get the number of neurons in this cluster
        n_neurons = self.neuron_slice.stop - self.neuron_slice.start

        # Form the constraints dictionary
        def _make_constraint(f, size_in, **kwargs):
            """Wrap a usage computation method to work with the partitioner."""
            def f_(neuron_slice, output_slice):
                # Calculate the number of neurons
                n_neurons = neuron_slice.stop - neuron_slice.start

                # Calculate the number of outgoing dimensions
                size_out = output_slice.stop - output_slice.start

                # Call the original method
                return f(size_in, size_out, n_neurons, **kwargs)
            return f_

        constraints = {
            dtcm_constraint: _make_constraint(_lif_dtcm_usage, self.size_in,
                                              n_neurons_in_cluster=n_neurons),
            cpu_constraint: _make_constraint(_lif_cpu_usage, self.size_in,
                                             n_neurons_in_cluster=n_neurons),
        }

        # Partition the slice of neurons that we have
        self.neuron_slices = list()
        output_slices = list()
        for neurons, outputs in partition.partition_multiple(
                (self.neuron_slice, slice(self.size_out)), constraints):
            self.neuron_slices.append(neurons)
            output_slices.append(outputs)

        n_slices = len(self.neuron_slices)
        assert n_slices <= 16  # Too many cores in the cluster

        # Also partition the input space
        input_slices = partition.divide_slice(slice(0, self.size_in),
                                              n_slices)

        # Zip these together to create the vertices
        all_slices = zip(input_slices, output_slices)
        for i, (in_slice, out_slice) in enumerate(all_slices):
            # Create the vertex
            vertex = EnsembleSlice(i, self.neuron_slices, in_slice,
                                   out_slice, self.regions)

            # Add to the list of vertices
            self.vertices.append(vertex)

        # Return all the vertices
        return self.vertices

    def load_to_machine(self, netlist, controller):
        """Load the ensemble data into memory."""
        # Get the chip that we're placed on
        placements = set(netlist.placements[v] for v in self.vertices)
        assert len(placements) == 1  # Missing constraint?
        x, y = placements.pop()

        # Allocate some shared memory for the cluster
        with controller(x=x, y=y):
            # Get the shared input vector memory
            shared_input_vector = controller.sdram_alloc(self.size_in*4,
                                                         clear=True)

            # Get the shared spike vector memory
            spike_bytes = neuron_utils.get_bytes_for_unpacked_spike_vector(
                self.neuron_slices)
            shared_spikes_vector = controller.sdram_alloc(spike_bytes,
                                                          clear=True)

            # Get the input and spikes synchronisation semaphores (take 2 more
            # bytes than we need so that we remain word-aligned)
            sema_input = controller.sdram_alloc(4, clear=True)

            if 0x60000000 <= sema_input < 0x70000000:
                # If the memory address is in the buffered range of addresses
                # then move it into the unbuffered range.
                sema_input += 0x10000000

            sema_spikes = sema_input + 1  # 2nd byte

        # Load each slice in turn, passing references to the shared memory
        for vertex in self.vertices:
            vertex.load_to_machine(
                netlist, shared_input_vector, shared_spikes_vector,
                sema_input, sema_spikes
            )

    def get_profiler_data(self):
        """Retrieve the profiler data from the simulation."""
        for vertex in self.vertices:
            # Construct a key for the vertex
            key = (vertex.neuron_slice.start,
                   vertex.neuron_slice.stop)

            # Get the data and yield a new entry
            yield key, vertex.get_profiler_data()

    def get_spike_data(self, n_steps):
        """Retrieve the spike data from the simulation."""
        for vertex in self.vertices:
            # Get the data and yield a new entry
            yield vertex.neuron_slice, vertex.get_spike_data(n_steps)

    def get_voltage_data(self, n_steps):
        """Retrieve the voltage data from the simulation."""
        for vertex in self.vertices:
            # Get the data and yield a new entry
            yield vertex.neuron_slice, vertex.get_voltage_data(n_steps)


class EnsembleSlice(Vertex):
    """Represents a single instance of the Ensemble APLX."""

    # Tag names, corresponding to those defined in `ensemble.h`
    profiler_tag_names = {
        0:  "Input filter",
        1:  "Neuron update",
        2:  "Decode and transmit output",
    }

    def __init__(self, vertex_index, cluster_slices, input_slice, output_slice,
                 ens_regions):
        """Create a new slice of an Ensemble.

        Parameters
        ----------
        vertex_index : int
            Index of this vertex within the cluster.
        cluster_slices : [slice, ...]
            List of slices
        input_slice : slice
            Slice of the input space to be managed by this instance.
        output_slice : slice
            Slice of the output space to be managed by this instance.
        """
        # Store the parameters
        self.input_slice = input_slice
        self.output_slice = output_slice
        self.regions = ens_regions

        # Get the specific neural slice we care about and information regarding
        # the rest of the vertices in this cluster.
        self.vertex_index = vertex_index
        self.neuron_slice = cluster_slices[vertex_index]
        self.n_vertices_in_cluster = len(cluster_slices)
        self.n_neurons_in_cluster = (cluster_slices[-1].stop -
                                     cluster_slices[0].start)

        # Get the basic arguments for the regions that we'll be storing
        self.region_arguments = _get_basic_region_arguments(
            self.neuron_slice, self.output_slice, cluster_slices
        )

        # Add some other arguments for the ensemble region
        self.region_arguments[EnsembleRegions.ensemble].kwargs.update({
            "population_id": vertex_index,
            "input_slice": input_slice,
            "neuron_slice": self.neuron_slice,
            "output_slice": output_slice,
        })

        # Compute the SDRAM usage
        sdram_usage = regions.utils.sizeof_regions_named(
            self.regions, self.region_arguments)

        # Prepare the vertex
        application = "ensemble"

        if ens_regions[EnsembleRegions.profiler].n_samples > 0:
            # If profiling then use the profiled version of the application
            application += "_profiled"

        super(EnsembleSlice, self).__init__(get_application(application),
                                            {Cores: 1, SDRAM: sdram_usage})

    def load_to_machine(self, netlist, shared_input_vector,
                        shared_spike_vector, sema_input, sema_spikes):
        """Load the application data into memory."""
        # Get a block of memory for each of the regions
        self.region_memory = \
            regions.utils.create_app_ptr_and_region_files_named(
                netlist.vertices_memory[self], self.regions,
                self.region_arguments
            )

        # Add some arguments to the ensemble region
        for kwarg, val in (("shared_input_vector", shared_input_vector),
                           ("shared_spike_vector", shared_spike_vector),
                           ("sema_input", sema_input),
                           ("sema_spikes", sema_spikes)):
            self.region_arguments[EnsembleRegions.ensemble].kwargs[kwarg] = val

        # Modify the keyword arguments to the keys region to include the
        # cluster index.
        self.region_arguments[EnsembleRegions.keys].kwargs["cluster"] = \
            self.cluster

        # Write each region into memory
        for key in EnsembleRegions:
            # Get the arguments and the memory
            args, kwargs = self.region_arguments[key]
            mem = self.region_memory[key]

            # Get the region
            region = self.regions[key]

            # Perform the write
            region.write_subregion_to_file(mem, *args, **kwargs)

    def get_profiler_data(self):
        """Retrieve profiler data from the simulation."""
        # Get the profiler output memory block
        mem = self.region_memory[EnsembleRegions.profiler]
        mem.seek(0)

        # Read profiler data from memory and put somewhere accessible
        profiler = self.regions[EnsembleRegions.profiler]
        return profiler.read_from_mem(mem, self.profiler_tag_names)

    def get_probe_data(self, region_name, n_steps):
        """Retrieve probed data from the simulation."""
        # Get the memory block
        mem = self.region_memory[region_name]
        mem.seek(0)

        # Read the data from memory
        region = self.regions[region_name]
        return region.to_array(mem, self.neuron_slice, n_steps)

    def get_spike_data(self, n_steps):
        """Retrieve spike data from the simulation."""
        return self.get_probe_data(EnsembleRegions.spikes, n_steps)

    def get_voltage_data(self, n_steps):
        """Retrieve voltage data from the simulation."""
        return self.get_probe_data(EnsembleRegions.voltages, n_steps)


class EnsembleRegion(regions.Region):
    """Region relevant to all ensembles.

    Python representation of `ensemble_parameters_t`.
    """
    def __init__(self, machine_timestep, size_in, n_profiler_samples=0,
                 record_spikes=False, record_voltages=False):
        self.machine_timestep = machine_timestep
        self.size_in = size_in
        self.n_profiler_samples = n_profiler_samples
        self.record_spikes = record_spikes
        self.record_voltages = record_voltages

    def sizeof(self, *args, **kwargs):
        # Always 15 words
        return 15*4

    def write_subregion_to_file(self, fp, n_populations, population_id,
                                n_neurons_in_population, input_slice,
                                neuron_slice, output_slice,
                                shared_input_vector, shared_spike_vector,
                                sema_input, sema_spikes):
        """Write the region to a file-like.

        Parameters
        ----------
        n_populations : int
            Number of populations with which this executable will collaborate.
        n_neurons_in_population : int
            Number of neurons that are in the shared population.
        population_id : int
            Index of this executable within this group of populations.
        input_slice : slice
            Slice of the input space that this executable will handle.
        neuron_slice : slice
            Portion of the neurons that this executable will handle.
        output_slice : slice
            Slice of the decoded space produced by this ensemble.
        shared_input_vector : int
            Address of SDRAM used to combine input values.
        shared_spike_vector : int
            Address in SDRAM used to combine spike vectors.
        sema_input : int
            Address of a semaphore in shared memory for synchronising reading
            of input vectors.
        sema_spikes : int
            Address of a semaphore in shared memory for synchronising reading
            of spike vectors.
        """
        # Prepare all data for packing
        n_neurons = neuron_slice.stop - neuron_slice.start
        is_offset = input_slice.start
        is_n_dims = input_slice.stop - input_slice.start
        n_decoder_rows = output_slice.stop - output_slice.start

        # Add the flags
        flags = 0x0
        for i, predicate in enumerate((self.record_spikes,
                                       self.record_voltages)):
            if predicate:
                flags |= 1 << i

        # Pack and write the data
        fp.write(struct.pack(
            "<15I",
            self.machine_timestep,
            n_neurons,
            self.size_in,
            n_neurons_in_population,
            n_populations,
            population_id,
            is_offset,
            is_n_dims,
            n_decoder_rows,
            self.n_profiler_samples,
            flags,
            shared_input_vector,
            shared_spike_vector,
            sema_input,
            sema_spikes
        ))


class LIFRegion(regions.Region):
    """Region containing parameters specific to LIF neurons.

    This is the Python representation of `lif_parameters_t`.
    """
    def __init__(self, dt, tau_rc, tau_ref):
        self.dt = dt
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

    def sizeof(self, *args, **kwargs):
        """Get the size of the region in bytes."""
        return 2*4  # 2 words

    def write_subregion_to_file(self, fp):
        """Write the region to the file-like object."""
        # The value -e^(-dt / tau_rc) is precomputed and is scaled down ever so
        # slightly to account for the effects of fixed point.  The result is
        # that the tuning curves of SpiNNaker neurons are usually within 5Hz of
        # the ideal curve and the tuning curve of reference Nengo neurons.  The
        # fudge factor applied (i.e. 1.0*2^-11) was determined by running the
        # tuning curve test in "regression-tests/test_tuning_curve.py",
        # plotting the results and stopping when the ideal tuning curve was
        # very closely matched by the SpiNNaker tuning curve - further
        # improvement of this factor may be possible.
        fp.write(struct.pack(
            "<2I",
            tp.value_to_fix(
                -np.expm1(-self.dt / self.tau_rc) * (1.0 - 2**-11)
            ),
            int(self.tau_ref // self.dt)
        ))


def get_decoders_and_keys(signals_connections, minimise=False):
    """Get a combined decoder matrix and a list of keys to use to transmit
    elements decoded using the decoders.
    """
    decoders = list()
    keys = list()

    # For each signal with a single connection we save the decoder and generate
    # appropriate keys
    for signal, transmission_params in signals_connections:
        decoder = transmission_params.decoders

        if not minimise:
            keep = np.array([True for _ in range(decoder.shape[0])])
        else:
            # We can reduce the number of packets sent and the memory
            # requirements by removing columns from the decoder matrix which
            # will always result in packets containing zeroes.
            keep = np.any(decoder != 0, axis=1)

        decoders.append(decoder[keep, :])
        for i, k in zip(range(decoder.shape[0]), keep):
            if k:
                keys.append(signal.keyspace(index=i))

    # Stack the decoders
    if len(decoders) > 0:
        decoders = np.vstack(decoders)
    else:
        decoders = np.array([[]])

    # Check we have a key for every row
    assert len(keys) == decoders.shape[0]

    return decoders, keys


class Args(collections.namedtuple("Args", "args, kwargs")):
    def __new__(cls, *args, **kwargs):
        return super(Args, cls).__new__(cls, args, kwargs)


def _get_basic_region_arguments(neuron_slice, output_slice, cluster_slices):
    """Get the initial arguments for LIF regions."""
    # By default there are no arguments at all
    region_arguments = collections.defaultdict(Args)

    # Regions sliced by neuron
    for r in (EnsembleRegions.encoders,
              EnsembleRegions.bias,
              EnsembleRegions.gain,
              EnsembleRegions.spikes,
              EnsembleRegions.voltages):
        region_arguments[r] = Args(neuron_slice)

    # Regions sliced by output
    for r in [EnsembleRegions.decoders, EnsembleRegions.keys]:
        region_arguments[r] = Args(output_slice)

    # Population lengths
    pop_lengths = [p.stop - p.start for p in cluster_slices]
    region_arguments[EnsembleRegions.population_length] = Args(pop_lengths)

    # Ensemble region arguments
    region_arguments[EnsembleRegions.ensemble].kwargs.update({
        "n_populations": len(pop_lengths),
        "n_neurons_in_population": sum(pop_lengths),
    })

    return region_arguments


def _lif_sdram_usage(size_in, size_out, n_neurons):
    """Approximation of SDRAM usage."""
    # Per neuron cost = encoders + decoders + gain + bias
    size = n_neurons * (size_in + size_out + 2) + size_out
    return size * 4


def _lif_dtcm_usage(size_in, size_out, n_neurons, n_neurons_in_cluster=None):
    """Approximation of DTCM usage."""
    # Assume no clustering if n_neurons_in_cluster is None
    if n_neurons_in_cluster is None:
        n_neurons_in_cluster = n_neurons

    # Per neuron cost = encoders + gain + bias + voltage + refractory counter
    # Per neuron in cluster cost = decoders
    size = (n_neurons * (size_in + 3) + size_out + size_in +
            n_neurons // 2) + n_neurons_in_cluster * size_out

    return size * 4


def _lif_cpu_usage(size_in, size_out, n_neurons, n_neurons_in_cluster=None):
    """Approximation of compute cost."""
    # Assume no clustering if n_neurons_in_cluster is None
    if n_neurons_in_cluster is None:
        n_neurons_in_cluster = n_neurons

    input_filter_cost = 40 * size_in + 131
    encoder_and_neuron_cost = (10 * size_in + 59) * n_neurons
    decoder_cost = (3 * n_neurons_in_cluster + 234) * size_out

    return input_filter_cost + encoder_and_neuron_cost + decoder_cost
