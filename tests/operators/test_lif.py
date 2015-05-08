import nengo
import numpy as np
import pytest
import struct
import tempfile

from rig import type_casts
s1615 = type_casts.float_to_fix(True, 32, 15)

from nengo_spinnaker.operators import lif


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


class TestSystemRegion(object):
    """Test system regions for Ensembles."""
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
        "machine_timestep, dt, size_in, tau_ref, tau_rc, "
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


class TestPESRegion(object):
    def test_dummy(self):
        """Test the current dummy PES region."""
        region = lif.PESRegion()
        assert region.sizeof() == 4

        # Test writing out
        fp = tempfile.TemporaryFile()
        region.write_region_to_file(fp)
        fp.seek(0)
        assert fp.read() == b'\x00' * 4
