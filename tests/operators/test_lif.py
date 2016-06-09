import itertools
import nengo
import numpy as np
import pytest
import struct
import tempfile

from nengo_spinnaker.operators import lif
from nengo_spinnaker.utils import type_casts as tp


class TestEnsembleLIF(object):
    @pytest.mark.parametrize("size_in", [1, 4, 5])
    def test_init(self, size_in):
        """Test that creating an Ensemble LIF creates an empty list of local
        probes and an empty input vector.
        """
        # Create an ensemble
        ens = nengo.Ensemble(100, size_in, add_to_container=False)

        op = lif.EnsembleLIF(ens)
        assert op.ensemble is ens
        assert np.all(op.direct_input == np.zeros(size_in))
        assert op.local_probes == list()


@pytest.mark.parametrize(
    ("machine_timestep", "size_in", "encoder_width", "n_populations",
     "n_neurons_in_population", "population_id", "n_learnt_input_signals",
     "input_slice", "neuron_slice", "output_slice", "learnt_output_slice",
     "shared_input_vector", "shared_learnt_input_vector",
     "shared_spike_vector", "sema_input", "sema_spikes", "n_profiler_samples"),
    [(1000, 3, 4, 4, 100, 0, 0, slice(0, 3), slice(0, 25), slice(0, 10), slice(0, 0),
      0x6780, [], 0x7800, 0x7804, 0x7808, 0),
     (2000, 1, 2, 4, 101, 4, 1, slice(1, 5), slice(22, 25), slice(3, 10), slice(3, 10),
      0x6780, [0x5000], 0x7800, 0x6000, 0x6004, 6000),
     ])
@pytest.mark.parametrize("record_spikes", (True, False))
@pytest.mark.parametrize("record_voltages", (True, False))
@pytest.mark.parametrize("record_encoders", (True, False))
def test_EnsembleRegion(machine_timestep, size_in, encoder_width,
                        n_populations, n_neurons_in_population, population_id,
                        n_learnt_input_signals,
                        input_slice, neuron_slice, output_slice,
                        learnt_output_slice, shared_input_vector,
                        shared_learnt_input_vector, shared_spike_vector,
                        sema_input, sema_spikes,
                        n_profiler_samples, record_spikes, record_voltages,
                        record_encoders):
    # Create the region
    region = lif.EnsembleRegion(machine_timestep, size_in, encoder_width,
                                n_learnt_input_signals,
                                record_spikes=record_spikes,
                                record_voltages=record_voltages,
                                record_encoders=record_encoders)

    # Update the region
    region.n_profiler_samples = n_profiler_samples

    # Check that the size is reported correctly
    assert region.sizeof() == (18 + n_learnt_input_signals) * 4

    # Check that the region is written out correctly
    fp = tempfile.TemporaryFile()
    region.write_subregion_to_file(
        fp, n_populations, population_id, n_neurons_in_population, input_slice,
        neuron_slice, output_slice, learnt_output_slice, shared_input_vector,
        shared_learnt_input_vector, shared_spike_vector,
        sema_input, sema_spikes
    )

    # Check that the correct amount of data was written
    fp.seek(0)
    data = fp.read()
    assert len(data) == region.sizeof()

    # Compute the expected flags
    flags = 0x0
    if record_spikes:
        flags |= 1 << 0
    if record_voltages:
        flags |= 1 << 1
    if record_encoders:
        flags |= 1 << 2

    # Check that the data was correct
    unpacked = struct.unpack("<%uI" % (18 + n_learnt_input_signals), data)

    assert unpacked[:18] == (
        machine_timestep,
        neuron_slice.stop - neuron_slice.start,
        size_in,
        encoder_width,
        n_neurons_in_population,
        n_populations,
        population_id,
        input_slice.start,
        input_slice.stop - input_slice.start,
        output_slice.stop - output_slice.start,
        learnt_output_slice.stop - learnt_output_slice.start,
        n_profiler_samples,
        n_learnt_input_signals,
        flags,
        shared_input_vector,
        shared_spike_vector,
        sema_input,
        sema_spikes)

    assert list(unpacked[18:18 + n_learnt_input_signals]) == shared_learnt_input_vector


@pytest.mark.parametrize(
    "dt, tau_ref, tau_rc", [(0.001, 0.0, 0.002), (0.01, 0.001, 0.02)])
def test_LIFRegion(dt, tau_rc, tau_ref):
    """Test region specific to LIF neurons."""
    # Create the region
    region = lif.LIFRegion(dt, tau_rc, tau_ref)

    # Check that the size is reported correctly
    assert region.sizeof() == 2*4  # 2 words

    # Check that the data is written out correctly
    # Create the file and write to it
    fp = tempfile.TemporaryFile()
    region.write_subregion_to_file(fp)

    # Read everything back
    fp.seek(0)
    values = fp.read()
    assert len(values) == region.sizeof()

    # Unpack and check that the values written out are reasonable
    dt_over_t_rc, t_ref = struct.unpack("<2I", values)
    assert t_ref == int(tau_ref // dt)
    assert (tp.value_to_fix(-np.expm1(-dt / tau_rc)) * 0.9 < dt_over_t_rc <
            tp.value_to_fix(-np.expm1(-dt / tau_rc)) * 1.1)

@pytest.mark.parametrize(
    "num_learning_rules",
    [0, 1, 2])
def test_VojaRegion(num_learning_rules):
    """Test region specific to Voja learning."""
    # Create the region
    region = lif.VojaRegion(1.0)

    # Add correct number of learning rules
    region.learning_rules.extend(itertools.repeat(
        lif.VojaLearningRule(1e-4, -1, 0, 0, -1),
        num_learning_rules))

    # Check that the size is reported correctly
    assert region.sizeof() == (8 + (num_learning_rules * 20))

    # Check that the data is written out correctly
    # Create the file and write to it
    fp = tempfile.TemporaryFile()
    region.write_subregion_to_file(fp)

    # Read everything back and check size is as sizeof reports
    fp.seek(0)
    values = fp.read()
    assert len(values) == region.sizeof()


@pytest.mark.parametrize(
    "output_slice, learnt_output_slice, learning_rules",
    [(slice(0, 4), slice(0, 2), [(slice(0, 2), slice(0, 2), 4)]),
     (slice(0, 4), slice(0, 2), [(slice(2, 4),)]),
     (slice(0, 4), slice(0, 2), [(slice(1, 2), slice(0, 1), 5)]),
     (slice(0, 4), slice(2, 4), [(slice(1, 3), slice(1, 2), 4)]),
     (slice(0, 4), slice(2, 4), [(slice(1, 5), slice(1, 3), 4)])]
)
def test_PESRegion(output_slice, learnt_output_slice, learning_rules):
    """Test region specific to PES learning."""
    # Build region with list of learning rules with specified slices
    region = lif.PESRegion(100)
    region.learning_rules.extend(
        [lif.PESLearningRule(1e-4, 1, l[0].start, l[0].stop, -1)
         for l in learning_rules])

    # Get region size
    region_size = region.sizeof(output_slice, learnt_output_slice)

    # Check that the data is written out correctly
    # Create the file and write to it
    fp = tempfile.TemporaryFile()
    region.write_subregion_to_file(fp, output_slice, learnt_output_slice)

    # Read everything back
    fp.seek(0)
    values = fp.read()
    assert len(values) == region_size

    # Unpack number of learning_rules
    num_rules_read = struct.unpack("<I", values[:4])[0]

    # Extract and count the learning rules which SHOULD'VE been written
    correct_learning_rules = [l for l in learning_rules if len(l) > 1]
    num_rules_correct = len(correct_learning_rules)

    # Check these match
    assert num_rules_read == num_rules_correct

    # Check on this basis whether size is correct
    assert region_size == (4 + (num_rules_correct * 24))

    # Loop through correct learning rules
    for i, l in enumerate(correct_learning_rules):
        # Unpack
        _, _, error_start, error_stop, decoder_start, _ = struct.unpack_from(
            "<i4Ii", values, 4 + (i * 24))

        # Check these match with correct parameters
        assert error_start == l[1].start
        assert error_stop == l[1].stop
        assert decoder_start == l[2]


@pytest.mark.parametrize(
    "neuron_slice, out_slice, learnt_out_slice, cluster_slices, cluster_lengths",
    [(slice(1, 99), slice(3, 44), slice(0, 0), [slice(0, 1), slice(1, 5)],
      [1, 4]),
     (slice(1, 99), slice(13, 54), slice(13, 54), [slice(0, 1), slice(1, 5), slice(5, 40)],
      [1, 4, 35])]
)
def test_get_basic_region_arguments(neuron_slice, out_slice, learnt_out_slice,
                                    cluster_slices, cluster_lengths):
    # Get the region arguments using these parameters
    region_args = lif._get_basic_region_arguments(neuron_slice, out_slice,
                                                  learnt_out_slice, cluster_slices)

    # For each region assert that the arguments are correct
    assert region_args[lif.Regions.ensemble].kwargs == {
        "n_populations": len(cluster_slices),
        "n_neurons_in_population": sum(cluster_lengths),
    }

    for r in (lif.Regions.neuron,
              lif.Regions.input_filters,
              lif.Regions.input_routing,
              lif.Regions.inhibition_filters,
              lif.Regions.inhibition_routing):
        assert region_args[r] == lif.Args()

    for r in (lif.Regions.encoders,
              lif.Regions.bias,
              lif.Regions.gain,
              lif.Regions.spike_recording,
              lif.Regions.voltage_recording,
              lif.Regions.encoder_recording):
        assert region_args[r] == lif.Args(neuron_slice)

    for r in (lif.Regions.decoders, lif.Regions.keys):
        assert region_args[r] == lif.Args(out_slice)

    for r in (lif.Regions.learnt_decoders, lif.Regions.learnt_keys):
        assert region_args[r] == lif.Args(learnt_out_slice)

    assert region_args[lif.Regions.population_length] == \
        lif.Args(cluster_lengths)
