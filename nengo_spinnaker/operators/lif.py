"""LIF Ensemble

Takes an intermediate representation of a LIF ensemble and returns a vertex and
appropriate callbacks to load and prepare the ensemble for simulation on
SpiNNaker.  The build method also manages the partitioning of the ensemble into
appropriate sized slices.
"""

import collections
import enum
import math
from nengo.base import ObjView
from nengo.connection import LearningRule
from nengo.learning_rules import PES, Voja
import numpy as np
from rig.place_and_route import Cores, SDRAM
from rig.place_and_route.constraints import SameChipConstraint
from six import iteritems
import struct

from nengo_spinnaker.builder.model import InputPort, OutputPort
from nengo_spinnaker.builder.netlist import netlistspec
from nengo_spinnaker.builder.ports import EnsembleInputPort, EnsembleOutputPort
from nengo_spinnaker.regions.filters import (FilterRegion, FilterRoutingRegion,
                                             add_filters, make_filter_regions)
from nengo_spinnaker.regions.utils import Args
from .. import regions
from nengo_spinnaker.netlist import Vertex
from nengo_spinnaker import partition
from nengo_spinnaker.utils.application import get_application
from nengo_spinnaker.utils.config import getconfig
from nengo_spinnaker.utils import type_casts as tp
from nengo_spinnaker.utils import neurons as neuron_utils


class Regions(enum.IntEnum):
    """Region names, corresponding to those defined in `ensemble.h`"""
    ensemble = 1  # General ensemble settings
    neuron = 2  # Neuron specific parameters
    encoders = 3  # Encoder matrix (for neurons on core)
    bias = 4  # Biases
    gain = 5  # Gains
    decoders = 6  # Decoder matrix (for all neurons, only some rows)
    learnt_decoders = 7  # Learnt decoder matrix (for all neurons, all rows)
    keys = 8  # Output keys
    learnt_keys = 9  # Learnt output keys
    population_length = 10  # Information about the entire cluster
    input_filters = 11
    input_routing = 12
    inhibition_filters = 13
    inhibition_routing = 14
    modulatory_filters = 15
    modulatory_routing = 16
    learnt_encoder_filters = 17
    learnt_encoder_routing = 18
    pes = 19
    voja = 20
    filtered_activity = 21
    profiler = 22
    spike_recording = 23
    voltage_recording = 24
    encoder_recording = 25


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
        self.record_encoders = False

    def make_vertices(self, model, n_steps):
        """Construct the data which can be loaded into the memory of a
        SpiNNaker machine.
        """
        # Build encoders, gain and bias regions
        params = model.params[self.ensemble]
        ens_regions = dict()

        # Extract all the filters from the incoming connections
        incoming = model.get_signals_to_object(self)

        # Filter out incoming modulatory connections
        incoming_modulatory = {port: signal
                               for (port, signal) in iteritems(incoming)
                               if isinstance(port, LearningRule)}

        (ens_regions[Regions.input_filters],
         ens_regions[Regions.input_routing]) = make_filter_regions(
            incoming[InputPort.standard], model.dt, True,
            model.keyspaces.filter_routing_tag,
            width=self.ensemble.size_in
        )
        (ens_regions[Regions.inhibition_filters],
         ens_regions[Regions.inhibition_routing]) = \
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

        ens_regions[Regions.decoders] = regions.MatrixRegion(
            tp.np_to_fix(decoders / model.dt),
            sliced_dimension=regions.MatrixPartitioning.rows)

        ens_regions[Regions.keys] = regions.KeyspacesRegion(
            output_keys,
            fields=[regions.KeyField({'cluster': 'cluster'})],
            partitioned_by_atom=True
        )

        # Extract pre-scaled encoders from parameters
        encoders_with_gain = params.scaled_encoders

        # Create filtered activity region
        ens_regions[Regions.filtered_activity] =\
            FilteredActivityRegion(model.dt)

        # Create, initially empty, PES region
        ens_regions[Regions.pes] = PESRegion(self.ensemble.n_neurons)

        # Loop through outgoing learnt connections
        mod_filters = list()
        mod_keyspace_routes = list()
        learnt_decoders = np.array([])
        learnt_output_keys = list()
        for sig, t_params in outgoing[EnsembleOutputPort.learnt]:
            l_rule = t_params.learning_rule
            l_rule_type = t_params.learning_rule.learning_rule_type

            # If this learning rule is PES
            if isinstance(l_rule_type, PES):
                # If there is a modulatory connection
                # associated with the learning rule
                if l_rule in incoming_modulatory:
                    e = incoming_modulatory[l_rule]

                    # Cache the index of the input filter containing the
                    # error signal and the starting offset into the learnt
                    # decoder where the learning rule should operate
                    error_filter_index = len(mod_filters)
                    decoder_start = learnt_decoders.shape[0]

                    # Get new decoders and output keys for learnt connection
                    rule_decoders, rule_output_keys = \
                        get_decoders_and_keys([(sig, t_params)], False)

                    # If there are no existing decodes, hstacking doesn't
                    # work so set decoders to new learnt decoder matrix
                    if decoder_start == 0:
                        learnt_decoders = rule_decoders
                    # Otherwise, stack learnt decoders
                    # alongside existing matrix
                    else:
                        learnt_decoders = np.vstack(
                            (learnt_decoders, rule_decoders))

                    decoder_stop = learnt_decoders.shape[0]

                    # Also add output keys to list
                    learnt_output_keys.extend(rule_output_keys)

                    # Add error connection to lists
                    # of modulatory filters and routes
                    mod_filters, mod_keyspace_routes = add_filters(
                        mod_filters, mod_keyspace_routes, e, minimise=False)

                    # Either add a new filter to the filtered activity
                    # region or get the index of the existing one
                    # activity_filter_index = \
                    #    ens_regions[Regions.filtered_activity].add_get_filter(
                    #        l_rule_type.pre_tau)
                    activity_filter_index = -1

                    # Add a new learning rule to the PES region
                    # **NOTE** divide learning rate by dt
                    # to account for activity scaling
                    ens_regions[Regions.pes].learning_rules.append(
                        PESLearningRule(
                            learning_rate=l_rule_type.learning_rate / model.dt,
                            error_filter_index=error_filter_index,
                            decoder_start=decoder_start,
                            decoder_stop=decoder_stop,
                            activity_filter_index=activity_filter_index))
                # Otherwise raise an error - PES requires a modulatory signal
                else:
                    raise ValueError(
                        "Ensemble %s has outgoing connection with PES "
                        "learning, but no corresponding modulatory "
                        "connection" % self.ensemble.label
                    )
            else:
                raise NotImplementedError(
                    "SpiNNaker does not support %s learning rule."
                    % l_rule_type
                )

        size_learnt_out = learnt_decoders.shape[0]

        ens_regions[Regions.learnt_decoders] = regions.MatrixRegion(
            tp.np_to_fix(learnt_decoders / model.dt),
            sliced_dimension=regions.MatrixPartitioning.rows)

        ens_regions[Regions.learnt_keys] = regions.KeyspacesRegion(
            learnt_output_keys,
            fields=[regions.KeyField({'cluster': 'cluster'})],
            partitioned_by_atom=True
        )
        # Create, initially empty, Voja region, passing in scaling factor
        # used, with gain, to scale activities to match encoders
        ens_regions[Regions.voja] = VojaRegion(1.0 / self.ensemble.radius)

        # Loop through incoming learnt connections
        learnt_encoder_filters = list()
        learnt_encoder_routes = list()
        for sig, t_params in incoming[EnsembleInputPort.learnt]:
            # If this learning rule is Voja
            l_rule = t_params.learning_rule
            l_rule_type = t_params.learning_rule.learning_rule_type
            if isinstance(l_rule_type, Voja):
                # If there is a modulatory connection
                # associated with the learning rule
                if l_rule in incoming_modulatory:
                    l = incoming_modulatory[l_rule]

                    # Cache the index of the input filter
                    # containing the learning signal
                    learn_sig_filter_index = len(mod_filters)

                    # Add learning connection to lists
                    # of modulatory filters and routes
                    mod_filters, mod_keyspace_routes = add_filters(
                        mod_filters, mod_keyspace_routes, l, minimise=False)
                # Otherwise, learning is always on so
                # invalidate learning signal index
                else:
                    learn_sig_filter_index = -1

                # Cache the index of the input filter containing
                # the input signal and the offset into the encoder
                # where the learning rule should operate
                decoded_input_filter_index = len(learnt_encoder_filters)
                encoder_offset = encoders_with_gain.shape[1]

                # Create a duplicate copy of the original size_in columns of
                # the encoder matrix for modification by this learning rule
                base_encoders = encoders_with_gain[:, :self.ensemble.size_in]
                encoders_with_gain = np.hstack((encoders_with_gain,
                                                base_encoders))

                # Add learnt connection to list of filters
                # and routes with learnt encoders
                learnt_encoder_filters, learnt_encoder_routes = add_filters(
                    learnt_encoder_filters, learnt_encoder_routes,
                    [(sig, t_params)], minimise=False)

                # Either add a new filter to the filtered activity
                # region or get the index of the existing one
                # activity_filter_index = \
                #    ens_regions[Regions.filtered_activity].add_get_filter(
                #        l_rule_type.post_tau)
                activity_filter_index = -1

                # Add a new learning rule to the Voja region
                # **NOTE** divide learning rate by dt
                # to account for activity scaling
                ens_regions[Regions.voja].learning_rules.append(
                    VojaLearningRule(
                        learning_rate=l_rule_type.learning_rate,
                        learning_signal_filter_index=learn_sig_filter_index,
                        encoder_offset=encoder_offset,
                        decoded_input_filter_index=decoded_input_filter_index,
                        activity_filter_index=activity_filter_index))
            else:
                raise NotImplementedError(
                    "SpiNNaker does not support %s learning rule."
                    % l_rule_type
                )

        # Create encoders region
        ens_regions[Regions.encoders] = regions.MatrixRegion(
            tp.np_to_fix(encoders_with_gain),
            sliced_dimension=regions.MatrixPartitioning.rows)

        # Tile direct input across all encoder copies (used for learning)
        tiled_direct_input = np.tile(
            self.direct_input,
            encoders_with_gain.shape[1] // self.ensemble.size_in)

        # Combine the direct input with the bias before converting to S1615 and
        # creating the region.
        bias_with_di = params.bias + np.dot(encoders_with_gain,
                                            tiled_direct_input)
        assert bias_with_di.ndim == 1
        ens_regions[Regions.bias] = regions.MatrixRegion(
            tp.np_to_fix(bias_with_di),
            sliced_dimension=regions.MatrixPartitioning.rows)

        # Convert the gains to S1615 before creating the region
        ens_regions[Regions.gain] = regions.MatrixRegion(
            tp.np_to_fix(params.gain),
            sliced_dimension=regions.MatrixPartitioning.rows)

        # Create modulatory filter and routing regions
        ens_regions[Regions.modulatory_filters] =\
            FilterRegion(mod_filters, model.dt)
        ens_regions[Regions.modulatory_routing] =\
            FilterRoutingRegion(mod_keyspace_routes,
                                model.keyspaces.filter_routing_tag)

        # Create learnt encoder filter and routing regions
        ens_regions[Regions.learnt_encoder_filters] =\
            FilterRegion(learnt_encoder_filters, model.dt)
        ens_regions[Regions.learnt_encoder_routing] =\
            FilterRoutingRegion(learnt_encoder_routes,
                                model.keyspaces.filter_routing_tag)

        # The population length region stores information about groups of
        # co-operating cores.
        ens_regions[Regions.population_length] = regions.ListRegion("I")

        # The ensemble region contains basic information about the ensemble
        n_learnt_input_signals = len(learnt_encoder_filters)
        ens_regions[Regions.ensemble] = EnsembleRegion(
            model.machine_timestep, self.ensemble.size_in,
            encoders_with_gain.shape[1], n_learnt_input_signals)

        # The neuron region contains information specific to the neuron type
        ens_regions[Regions.neuron] = LIFRegion(
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
        ens_regions[Regions.profiler] = regions.Profiler(n_profiler_samples)
        ens_regions[Regions.ensemble].n_profiler_samples = n_profiler_samples

        # Manage probes
        for probe in self.local_probes:
            if probe.attr in ("output", "spikes"):
                self.record_spikes = True
            elif probe.attr == "voltage":
                self.record_voltages = True
            elif probe.attr == "scaled_encoders":
                self.record_encoders = True
            else:
                raise NotImplementedError(
                    "Cannot probe {} on Ensembles".format(probe.attr)
                )

        # Set the flags
        ens_regions[Regions.ensemble].record_spikes = self.record_spikes
        ens_regions[Regions.ensemble].record_voltages = self.record_voltages
        ens_regions[Regions.ensemble].record_encoders = self.record_encoders

        # Create the probe recording regions
        self.learnt_enc_dims = (encoders_with_gain.shape[1] -
                                self.ensemble.size_in)
        ens_regions[Regions.spike_recording] =\
            regions.SpikeRecordingRegion(n_steps if self.record_spikes
                                         else 0)
        ens_regions[Regions.voltage_recording] =\
            regions.VoltageRecordingRegion(n_steps if self.record_voltages
                                           else 0)
        ens_regions[Regions.encoder_recording] =\
            regions.EncoderRecordingRegion(n_steps if self.record_encoders
                                           else 0, self.learnt_enc_dims)

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
        def _make_constraint(f, size_in, size_out, size_learnt_out, **kwargs):
            """Wrap a usage computation method to work with the partitioner."""
            def f_(vertex_slice):
                # Calculate the number of neurons
                n_neurons = vertex_slice.stop - vertex_slice.start

                # Call the original method
                return f(size_in, size_out, size_learnt_out,
                         n_neurons, **kwargs)
            return f_

        partition_constraints = {
            sdram_constraint: _make_constraint(_lif_sdram_usage,
                                               encoders_with_gain.shape[1],
                                               size_out, size_learnt_out),
            dtcm_constraint: _make_constraint(_lif_dtcm_usage,
                                              encoders_with_gain.shape[1],
                                              size_out, size_learnt_out),
            cpu_constraint: _make_constraint(_lif_cpu_usage,
                                             encoders_with_gain.shape[1],
                                             size_out, size_learnt_out),
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
            cluster = EnsembleCluster(sl, self.ensemble.size_in,
                                      encoders_with_gain.shape[1], size_out,
                                      size_learnt_out, n_learnt_input_signals,
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
                    neurons_stop = neurons.start + data.shape[1]
                    spikes[:, neurons.start:neurons_stop] = data

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

        # If (learnt) encoders were recorded
        if self.record_encoders:
            # Create empty matrix to hold probed data
            encoders = np.empty((
                n_steps,
                self.ensemble.n_neurons,
                self.learnt_enc_dims))

            # For each cluster read back the encoder data
            for cl in self.clusters:
                # For each neuron slice copy in the encoder data
                for neurons, data in cl.get_encoder_data(n_steps):
                    encoders[:, neurons] = data

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

            # Copy desired slice of recorded data into simulator
            if p.attr in ("output", "spikes"):
                # Spike data
                probe_data = spike_vals[::sample_every, neuron_slice]
            elif p.attr == "voltage":
                # Voltage data
                probe_data = voltages[::sample_every, neuron_slice]
            elif p.attr == "scaled_encoders":
                probe_data = encoders[::sample_every, neuron_slice, :]

            # Store the probe data
            if p in simulator.data:
                # Append the new probe data to the existing probe data
                probe_data = np.vstack((simulator.data[p], probe_data))
            simulator.data[p] = probe_data


class EnsembleCluster(object):
    def __init__(self, neuron_slice, size_in, encoder_width, size_out,
                 size_learnt_out, n_learnt_input_signals, regions):
        """Create a new cluster of collaborating cores."""
        self.neuron_slice = neuron_slice
        self.regions = regions
        self.neuron_slices = list()
        self.vertices = list()
        self.size_in = size_in
        self.encoder_width = encoder_width
        self.size_out = size_out
        self.size_learnt_out = size_learnt_out
        self.n_learnt_input_signals = n_learnt_input_signals

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
            def f_(neuron_slice, output_slice, learnt_output_slice):
                # Calculate the number of neurons
                n_neurons = neuron_slice.stop - neuron_slice.start

                # Calculate the number of outgoing dimensions
                size_out = output_slice.stop - output_slice.start
                size_learnt_out = learnt_output_slice.stop -\
                    learnt_output_slice.start

                # Call the original method
                return f(size_in, size_out, size_learnt_out,
                         n_neurons, **kwargs)
            return f_

        constraints = {
            dtcm_constraint: _make_constraint(_lif_dtcm_usage,
                                              self.encoder_width,
                                              n_neurons_in_cluster=n_neurons),
            cpu_constraint: _make_constraint(_lif_cpu_usage,
                                             self.encoder_width,
                                             n_neurons_in_cluster=n_neurons),
        }

        # Partition the slice of neurons that we have
        self.neuron_slices = list()
        output_slices = list()
        learnt_output_slices = list()
        for neurons, outputs, learnt_outputs in partition.partition_multiple(
                (self.neuron_slice, slice(self.size_out),
                 slice(self.size_learnt_out)), constraints):
            self.neuron_slices.append(neurons)
            output_slices.append(outputs)
            learnt_output_slices.append(learnt_outputs)

        n_slices = len(self.neuron_slices)
        assert n_slices <= 16  # Too many cores in the cluster

        # Also partition the input space
        input_slices = partition.divide_slice(slice(0, self.size_in),
                                              n_slices)

        # Zip these together to create the vertices
        all_slices = zip(input_slices, output_slices, learnt_output_slices)
        for i, (inp, output, learnt) in enumerate(all_slices):
            # Create the vertex
            vertex = EnsembleSlice(i, self.neuron_slices, inp, output, learnt,
                                   self.regions)

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
            # Allocate shared input vector memory
            shared_input_vector = controller.sdram_alloc(self.size_in*4,
                                                         clear=True)

            # Allocate shared learnt input vector memory
            shared_learnt_input_vector = [
                controller.sdram_alloc(self.size_in*4, clear=True)
                for _ in range(self.n_learnt_input_signals)]

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
                netlist, shared_input_vector, shared_learnt_input_vector,
                shared_spikes_vector, sema_input, sema_spikes
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

    def get_encoder_data(self, n_steps):
        """Retrieved (learnt) encoder data from the simulation."""
        for vertex in self.vertices:
            yield vertex.neuron_slice, vertex.get_encoder_data(n_steps)


class EnsembleSlice(Vertex):
    """Represents a single instance of the Ensemble APLX."""

    # Tag names, corresponding to those defined in `ensemble.h`
    profiler_tag_names = {
        0:  "Input filter",
        1:  "Neuron update",
        2:  "Decode and transmit output",
        3:  "PES",
        4:  "Voja",
    }

    def __init__(self, vertex_index, cluster_slices, input_slice, output_slice,
                 learnt_output_slice, ens_regions):
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
        learnt_output_slice : slice
            Slice of the learned output space to be managed by this instance.
        """
        # Store the parameters
        self.input_slice = input_slice
        self.output_slice = output_slice
        self.learnt_output_slice = learnt_output_slice
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
            self.neuron_slice, self.output_slice,
            self.learnt_output_slice, cluster_slices
        )

        # Add some other arguments for the ensemble region
        self.region_arguments[Regions.ensemble].kwargs.update({
            "population_id": vertex_index,
            "input_slice": input_slice,
            "neuron_slice": self.neuron_slice,
            "output_slice": output_slice,
            "learnt_output_slice": learnt_output_slice,
        })

        # Add input width parameter to input filter regions that get sliced
        input_width = input_slice.stop - input_slice.start
        self.region_arguments[Regions.input_filters].\
            kwargs["filter_width"] = input_width
        self.region_arguments[Regions.learnt_encoder_filters].\
            kwargs["filter_width"] = input_width

        # Compute the SDRAM usage
        sdram_usage = regions.utils.sizeof_regions_named(self.regions,
                                                         self.region_arguments)

        # Prepare the vertex
        application = "ensemble"

        if ens_regions[Regions.profiler].n_samples > 0:
            # If profiling then use the profiled version of the application
            application += "_profiled"

        super(EnsembleSlice, self).__init__(get_application(application),
                                            {Cores: 1, SDRAM: sdram_usage})

    def load_to_machine(self, netlist, shared_input_vector,
                        shared_learnt_input_vector, shared_spike_vector,
                        sema_input, sema_spikes):
        """Load the application data into memory."""
        # Get a block of memory for each of the regions
        self.region_memory = \
            regions.utils.create_app_ptr_and_region_files_named(
                netlist.vertices_memory[self], self.regions,
                self.region_arguments
            )

        # Add some arguments to the ensemble region
        for kwarg, val in (
                ("shared_input_vector", shared_input_vector),
                ("shared_spike_vector", shared_spike_vector),
                ("shared_learnt_input_vector", shared_learnt_input_vector),
                ("sema_input", sema_input),
                ("sema_spikes", sema_spikes)):
            self.region_arguments[Regions.ensemble].kwargs[kwarg] = val

        # Modify the keyword arguments to the keys region to include the
        # cluster index.
        self.region_arguments[Regions.keys].kwargs["cluster"] = \
            self.cluster
        self.region_arguments[Regions.learnt_keys].kwargs["cluster"] = \
            self.cluster

        # Write each region into memory
        for key in Regions:
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
        mem = self.region_memory[Regions.profiler]
        mem.seek(0)

        # Read profiler data from memory and put somewhere accessible
        profiler = self.regions[Regions.profiler]
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
        return self.get_probe_data(Regions.spike_recording, n_steps)

    def get_voltage_data(self, n_steps):
        """Retrieve voltage data from the simulation."""
        return self.get_probe_data(Regions.voltage_recording, n_steps)

    def get_encoder_data(self, n_steps):
        """Retrieve (learnt) encoder data from the simulation."""
        return self.get_probe_data(Regions.encoder_recording, n_steps)


class EnsembleRegion(regions.Region):
    """Region relevant to all ensembles.

    Python representation of `ensemble_parameters_t`.
    """
    def __init__(self, machine_timestep, size_in, encoder_width,
                 n_learnt_input_signals, n_profiler_samples=0,
                 record_spikes=False, record_voltages=False,
                 record_encoders=False):
        self.machine_timestep = machine_timestep
        self.size_in = size_in
        self.encoder_width = encoder_width
        self.n_learnt_input_signals = n_learnt_input_signals
        self.n_profiler_samples = n_profiler_samples
        self.record_spikes = record_spikes
        self.record_voltages = record_voltages
        self.record_encoders = record_encoders

    def sizeof(self, *args, **kwargs):
        return (18 + self.n_learnt_input_signals) * 4

    def write_subregion_to_file(self, fp, n_populations, population_id,
                                n_neurons_in_population, input_slice,
                                neuron_slice, output_slice,
                                learnt_output_slice, shared_input_vector,
                                shared_learnt_input_vector,
                                shared_spike_vector, sema_input, sema_spikes):
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
        learnt_output_slice : slice
            Slice of the learnt decoded space produced by this ensemble.
        shared_input_vector : int
            Address of SDRAM used to combine input values.
        shared_learnt_input_vector : list
            List of addresses of SDRAM used to combine learnt input values.
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
        n_learnt_decoder_rows = learnt_output_slice.stop -\
            learnt_output_slice.start

        assert len(shared_learnt_input_vector) == self.n_learnt_input_signals

        # Add the flags
        flags = 0x0
        for i, predicate in enumerate((self.record_spikes,
                                       self.record_voltages,
                                       self.record_encoders)):
            if predicate:
                flags |= 1 << i

        # Pack and write the data
        fp.write(struct.pack(
            "<%uI" % (18 + self.n_learnt_input_signals),
            self.machine_timestep,
            n_neurons,
            self.size_in,
            self.encoder_width,
            n_neurons_in_population,
            n_populations,
            population_id,
            is_offset,
            is_n_dims,
            n_decoder_rows,
            n_learnt_decoder_rows,
            self.n_profiler_samples,
            self.n_learnt_input_signals,
            flags,
            shared_input_vector,
            shared_spike_vector,
            sema_input,
            sema_spikes,
            *shared_learnt_input_vector
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

PESLearningRule = collections.namedtuple(
    "PESLearningRule",
    "learning_rate, error_filter_index, decoder_start, decoder_stop, "
    "activity_filter_index")


class PESRegion(regions.Region):
    """Region representing parameters for PES learning rules.
    """
    def __init__(self, n_neurons):
        self.learning_rules = []
        self.n_neurons = n_neurons

    def sizeof(self, output_slice, learnt_output_slice):
        sliced_rules = self.get_sliced_learning_rules(learnt_output_slice)
        return 4 + (len(sliced_rules) * 24)

    def write_subregion_to_file(self, fp, output_slice, learnt_output_slice):
        # Get number of static decoder rows - used to offset
        # all PES decoder offsets into global decoder array
        n_decoder_rows = output_slice.stop - output_slice.start

        # Filter learning rules that learn decoders within learnt output slice
        sliced_rules = self.get_sliced_learning_rules(learnt_output_slice)

        # Write number of learning rules for slice
        fp.write(struct.pack("<I", len(sliced_rules)))

        # Loop through sliced learning rules
        for l in sliced_rules:
            # Error signal starts either at 1st dimension or the first
            # dimension of decoder that occurs within learnt output slice
            error_start_dim = max(0,
                                  learnt_output_slice.start - l.decoder_start)

            # Error signal end either at last dimension or
            # the last dimension of learnt output slice
            error_end_dim = min(l.decoder_stop - l.decoder_start,
                                learnt_output_slice.stop - l.decoder_start)

            # The row of the global decoder is the learnt row relative to the
            # start of the learnt output slice with the number of static
            # decoder rows added to put it into combined decoder space
            decoder_row = n_decoder_rows +\
                max(0, l.decoder_start - learnt_output_slice.start)

            # Write structure
            fp.write(struct.pack(
                "<i4Ii",
                tp.value_to_fix(l.learning_rate / float(self.n_neurons)),
                l.error_filter_index,
                error_start_dim,
                error_end_dim,
                decoder_row,
                l.activity_filter_index))

    def get_sliced_learning_rules(self, learnt_output_slice):
        return [l for l in self.learning_rules
                if (l.decoder_start < learnt_output_slice.stop and
                    l.decoder_stop > learnt_output_slice.start)]

VojaLearningRule = collections.namedtuple(
    "VojaLearningRule",
    "learning_rate, learning_signal_filter_index, encoder_offset, "
    "decoded_input_filter_index, activity_filter_index")


class VojaRegion(regions.Region):
    """Region representing parameters for PES learning rules.
    """
    def __init__(self, one_over_radius):
        self.learning_rules = []
        self.one_over_radius = one_over_radius

    def sizeof(self):
        return 8 + (len(self.learning_rules) * 20)

    def write_subregion_to_file(self, fp):
        # Write number of learning rules and scaling factor
        fp.write(struct.pack(
            "<2I",
            len(self.learning_rules),
            tp.value_to_fix(self.one_over_radius)
        ))

        # Write learning rules
        for l in self.learning_rules:
            data = struct.pack(
                "<Ii2Ii",
                tp.value_to_fix(l.learning_rate),
                l.learning_signal_filter_index,
                l.encoder_offset,
                l.decoded_input_filter_index,
                l.activity_filter_index
            )
            fp.write(data)


class FilteredActivityRegion(regions.Region):
    def __init__(self, dt):
        self.filter_propogators = []
        self.dt = dt

    def add_get_filter(self, time_constant):
        # If time constant is none or less than dt,
        # a filter is not required so return -1
        if time_constant is None or time_constant < self.dt:
            return -1
        # Otherwise
        else:
            # Calculate propogator
            propogator = math.exp(-float(self.dt) / float(time_constant))

            # Convert to fixed-point
            propogator_fixed = tp.value_to_fix(propogator)

            # If there is already a filter with the same fixed-point
            # propogator in the list, return its index
            if propogator_fixed in self.filter_propogators:
                return self.filter_propogators.index(propogator_fixed)
            # Otherwise add propogator to list and return its index
            else:
                self.filter_propogators.append(propogator_fixed)
                return (len(self.filter_propogators) - 1)

    def sizeof(self):
        return 4 + (8 * len(self.filter_propogators))

    def write_subregion_to_file(self, fp):
        # Write number of learning rules
        fp.write(struct.pack("<I", len(self.filter_propogators)))

        # Write filters
        for f in self.filter_propogators:
            data = struct.pack(
                "<ii",
                f,
                tp.value_to_fix(1.0) - f,
            )
            fp.write(data)


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


def _get_basic_region_arguments(neuron_slice, output_slice,
                                learnt_output_slice, cluster_slices):
    """Get the initial arguments for LIF regions."""
    # By default there are no arguments at all
    region_arguments = collections.defaultdict(Args)

    # Regions sliced by neuron
    for r in (Regions.encoders,
              Regions.bias,
              Regions.gain,
              Regions.spike_recording,
              Regions.voltage_recording,
              Regions.encoder_recording):
        region_arguments[r] = Args(neuron_slice)

    # Regions sliced by output
    for r in (Regions.decoders, Regions.keys):
        region_arguments[r] = Args(output_slice)

    # Regions sliced by learnt output
    for r in (Regions.learnt_decoders, Regions.learnt_keys):
        region_arguments[r] = Args(learnt_output_slice)

    # Population lengths
    pop_lengths = [p.stop - p.start for p in cluster_slices]
    region_arguments[Regions.population_length] = Args(pop_lengths)

    # PES region needs both static and learnt output slices
    region_arguments[Regions.pes] = Args(output_slice, learnt_output_slice)

    # Ensemble region arguments
    region_arguments[Regions.ensemble].kwargs.update({
        "n_populations": len(pop_lengths),
        "n_neurons_in_population": sum(pop_lengths),
    })

    return region_arguments


def _lif_sdram_usage(size_in, size_out, size_learnt_out, n_neurons):
    """Approximation of SDRAM usage."""
    # Per neuron cost = encoders + decoders + gain + bias
    total_size_out = size_out + size_learnt_out
    size = (n_neurons * (size_in + total_size_out + 2)) + total_size_out
    return size * 4


def _lif_dtcm_usage(size_in, size_out, size_learnt_out,
                    n_neurons, n_neurons_in_cluster=None):
    """Approximation of DTCM usage."""
    # Assume no clustering if n_neurons_in_cluster is None
    if n_neurons_in_cluster is None:
        n_neurons_in_cluster = n_neurons

    # Per neuron cost = encoders + gain + bias + voltage + refractory counter
    # Per neuron in cluster cost = decoders
    # **TODO** filters (especially learnt ones need taking into account)
    total_size_out = size_out + size_learnt_out
    size = (n_neurons * (size_in + 3) + total_size_out + (5 * size_in) +
            n_neurons // 2) + (n_neurons_in_cluster * total_size_out)

    return size * 4


def _lif_cpu_usage(size_in, size_out, size_learnt_out, n_neurons,
                   n_neurons_in_cluster=None):
    """Approximation of compute cost."""
    # Assume no clustering if n_neurons_in_cluster is None
    if n_neurons_in_cluster is None:
        n_neurons_in_cluster = n_neurons

    input_filter_cost = 40 * size_in + 131
    encoder_and_neuron_cost = (10 * size_in + 59) * n_neurons
    decoder_cost = (3 * n_neurons_in_cluster + 234) * size_out

    # **TODO** base on profiled data!
    learnt_decoder_cost = (3 * n_neurons_in_cluster + 234) * size_learnt_out

    return input_filter_cost + encoder_and_neuron_cost +\
        decoder_cost + learnt_decoder_cost
