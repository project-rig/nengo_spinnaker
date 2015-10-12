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
    ("machine_timestep", "size_in", "n_populations", "n_neurons_in_population",
     "population_id", "input_slice", "neuron_slice", "output_slice",
     "shared_input_vector", "shared_spike_vector", "sema_input", "sema_spikes",
     "n_profiler_samples"),
    [(1000, 3, 4, 100, 0, slice(0, 3), slice(0, 25), slice(0, 10),
      0x6780, 0x7800, 0x7804, 0x7808, 0),
     (2000, 1, 2, 101, 4, slice(1, 5), slice(22, 25), slice(3, 10),
      0x6780, 0x7800, 0x6000, 0x6004, 6000),
     ])
@pytest.mark.parametrize("record_spikes", (True, False))
@pytest.mark.parametrize("record_voltages", (True, False))
def test_EnsembleRegion(machine_timestep, size_in, n_populations,
                        n_neurons_in_population, population_id, input_slice,
                        neuron_slice, output_slice, shared_input_vector,
                        shared_spike_vector, sema_input, sema_spikes,
                        n_profiler_samples, record_spikes, record_voltages):
    # Create the region
    region = lif.EnsembleRegion(machine_timestep, size_in,
                                record_spikes=record_spikes,
                                record_voltages=record_voltages)

    # Update the region
    region.n_profiler_samples = n_profiler_samples

    # Check that the size is reported correctly
    assert region.sizeof() == 4*15  # 15 words

    # Check that the region is written out correctly
    fp = tempfile.TemporaryFile()
    region.write_subregion_to_file(
        fp, n_populations, population_id, n_neurons_in_population, input_slice,
        neuron_slice, output_slice, shared_input_vector, shared_spike_vector,
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

    # Check that the data was correct
    assert struct.unpack("<15I", data) == (
        machine_timestep,
        neuron_slice.stop - neuron_slice.start,
        size_in,
        n_neurons_in_population,
        n_populations,
        population_id,
        input_slice.start,
        input_slice.stop - input_slice.start,
        output_slice.stop - output_slice.start,
        n_profiler_samples,
        flags,
        shared_input_vector,
        shared_spike_vector,
        sema_input,
        sema_spikes
    )


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
    "neuron_slice, out_slice, cluster_slices, cluster_lengths",
    [(slice(1, 99), slice(3, 44), [slice(0, 1), slice(1, 5)],
      [1, 4]),
     (slice(1, 99), slice(13, 54), [slice(0, 1), slice(1, 5), slice(5, 40)],
      [1, 4, 35])]
)
def test_get_basic_region_arguments(neuron_slice, out_slice, cluster_slices,
                                    cluster_lengths):
    # Get the region arguments using these parameters
    region_args = lif._get_basic_region_arguments(neuron_slice, out_slice,
                                                  cluster_slices)

    # For each region assert that the arguments are correct
    assert region_args[lif.EnsembleRegions.ensemble].kwargs == {
        "n_populations": len(cluster_slices),
        "n_neurons_in_population": sum(cluster_lengths),
    }

    for r in (lif.EnsembleRegions.neuron,
              lif.EnsembleRegions.input_filters,
              lif.EnsembleRegions.input_routing,
              lif.EnsembleRegions.inhibition_filters,
              lif.EnsembleRegions.inhibition_routing):
        assert region_args[r] == lif.Args()

    for r in (lif.EnsembleRegions.encoders,
              lif.EnsembleRegions.bias,
              lif.EnsembleRegions.gain,
              lif.EnsembleRegions.spikes,
              lif.EnsembleRegions.voltages):
        assert region_args[r] == lif.Args(neuron_slice)

    for r in (lif.EnsembleRegions.decoders, lif.EnsembleRegions.keys):
        assert region_args[r] == lif.Args(out_slice)

    assert region_args[lif.EnsembleRegions.population_length] == \
        lif.Args(cluster_lengths)
