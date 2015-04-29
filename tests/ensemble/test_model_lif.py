import nengo
import pytest
import struct
import tempfile

from rig import type_casts
s1615 = type_casts.float_to_fix(True, 32, 15)

from nengo_spinnaker.ensemble import model_lif as lif

from nengo_spinnaker.annotations import Annotations
from nengo_spinnaker.ensemble.annotations import AnnotatedEnsemble



class TestSystemRegion(object):
    """Test system regions for Ensembles."""
    @pytest.mark.parametrize(
        "machine_timestep, dt, n_neurons, size_in, tau_ref, tau_rc, size_out",
        [(1000, 0.001, 300, 5, 0.0, 0.002, 7),
         (10000, 0.01, 123, 1, 0.001, 0.02, 3),
         ]
    )
    def test_from_annotations(self, machine_timestep, dt, n_neurons, size_in,
                              tau_ref, tau_rc, size_out):
        """Test constructing a system region from annotation parameters."""
        with nengo.Network() as network:
            ens = nengo.Ensemble(n_neurons, size_in)
            ens.neuron_type.tau_ref = tau_ref
            ens.neuron_type.tau_rc = tau_rc

        # Create an empty annotation for the ensemble
        ens_annotation = AnnotatedEnsemble(ens)
        annotations = Annotations(
            {ens: ens_annotation}, {}, [], [], machine_timestep
        )

        # Build the network
        model = nengo.builder.Model(dt=dt)
        model.build(network)

        # Create the system region
        region = lif.SystemRegion.from_annotations(
            ens, ens_annotation, model, annotations, size_out
        )

        # Check parameters were correctly extracted
        assert region.n_input_dimensions == size_in
        assert region.n_output_dimensions == size_out
        assert region.machine_timestep == machine_timestep
        assert region.t_ref == tau_ref
        assert region.t_rc == tau_rc
        assert region.dt == dt
        assert not region.probe_spikes

    def test_from_annotations_probe_spikes(self):
        """Test constructing a system region from annotation when spike probing
        is enabled.
        """
        with nengo.Network() as network:
            ens = nengo.Ensemble(100, 1)
            p_ens = nengo.Probe(ens.neurons)

        # Create an empty annotation for the ensemble
        ens_annotation = AnnotatedEnsemble(ens)
        ens_annotation.local_probes.append(p_ens)
        annotations = Annotations({ens: ens_annotation}, {}, [], [])

        # Build the network
        model = nengo.builder.Model()
        model.build(network)

        # Create the system region
        region = lif.SystemRegion.from_annotations(
            ens, ens_annotation, model, annotations, 5
        )

        # Check parameters were correctly extracted
        print(region.probe_spikes)
        assert region.probe_spikes

    def test_sizeof(self):
        region = lif.SystemRegion(1, 5, 1000, 0.01, 0.02, 0.001, False)
        assert region.sizeof() == 8 * 4  # 8 words

    @pytest.mark.parametrize(
        "vertex_slice, vertex_neurons",
        [(slice(0, 100), 100),
         (slice(100, 120), 20),
         ]
    )
    @pytest.mark.parametrize(
        "machine_timestep, dt, size_in, tau_ref, tau_rc, " \
        "size_out, probe_spikes",
        [(1000, 0.001, 5, 0.0, 0.002, 7, True),
         (10000, 0.01, 1, 0.001, 0.02, 3, False),
         ]
    )
    def test_write_subregion_to_file(self, machine_timestep, dt,
                                     size_in, tau_ref, tau_rc,
                                     size_out, probe_spikes,
                                     vertex_slice, vertex_neurons):
        # Check that the region is correctly written to file
        region = lif.SystemRegion(
            size_in, size_out, machine_timestep, tau_ref, tau_rc,
            dt, probe_spikes
        )

        # Create the file
        fp = tempfile.TemporaryFile()

        # Write to it
        region.write_subregion_to_file(vertex_slice, fp)

        # Read back and check that the values are sane
        fp.seek(0)
        values = fp.read()
        assert len(values) == region.sizeof()

        (n_in, n_out, n_n, m_t, t_ref, dt_over_t_rc, rec_spikes, i_dims) = \
            struct.unpack_from("<8I", values)
        assert n_in == size_in
        assert n_out == size_out
        assert n_n == vertex_neurons
        assert m_t == machine_timestep
        assert t_ref == int(tau_ref // dt)
        assert dt_over_t_rc == s1615(dt / tau_rc)
        assert ((probe_spikes and rec_spikes != 0) or
                (not probe_spikes and rec_spikes == 0))
        assert i_dims == 1
