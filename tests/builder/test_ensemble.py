import nengo
import numpy as np
import pytest

from nengo_spinnaker.builder import builder, ensemble


class TestBuildEnsembleLIF(object):
    @pytest.mark.parametrize("n_neurons, size_in", [(100, 1), (300, 4)])
    def test_build_ensemble_lif(self, n_neurons, size_in):
        """Test building LIF ensembles."""
        # Create a Nengo ensemble to build
        ens = nengo.Ensemble(n_neurons, size_in, add_to_container=False)

        # Create a model
        model = builder.Model()
        model.seeds[ens] = 1

        # Build the ensemble
        ensemble.build_ensemble(model, ens)

        # Check that the built ensemble was inserted into the params and that
        # the parameters are (loosely) as expected.
        assert model.params[ens].eval_points is not None
        assert (model.params[ens].encoders.shape ==
                model.params[ens].scaled_encoders.shape ==
                (n_neurons, size_in))
        assert (model.params[ens].intercepts.shape ==
                model.params[ens].max_rates.shape ==
                model.params[ens].gain.shape ==
                model.params[ens].bias.shape == (n_neurons, ))

        # Check that a new object was inserted into the objects dictionary
        assert isinstance(model.object_intermediates[ens],
                          ensemble.EnsembleLIF)

    def test_with_encoders_and_gain_bias(self):
        """Test that the encoders we provide are used (albeit scaled)"""
        # Create a Nengo ensemble to build
        ens = nengo.Ensemble(1, 1, add_to_container=False)
        ens.radius = 10.0
        ens.encoders = np.array([[1.0]])
        ens.gain = np.array([0.5])
        ens.bias = np.array([0.0])

        # Create a model
        model = builder.Model()
        model.seeds[ens] = 1

        # Build the ensemble
        ensemble.build_ensemble(model, ens)

        # Check that parameters are (loosely) as expected.
        assert model.params[ens].encoders == ens.encoders
        assert model.params[ens].gain == ens.gain
        assert model.params[ens].bias == ens.bias
        assert model.params[ens].scaled_encoders == ens.encoders * (0.5 / 10)

    @pytest.mark.xfail
    def test_only_gain(self):
        """Build an ensemble with only gain specified."""
        # Create a Nengo ensemble to build
        ens = nengo.Ensemble(1, 1, add_to_container=False)
        ens.gain = np.array([0.5])

        # Create a model
        model = builder.Model()
        model.seeds[ens] = 1

        # Build the ensemble
        ensemble.build_ensemble(model, ens)

        # Check that parameters are (loosely) as expected.
        assert model.params[ens].gain == ens.gain  # pragma : no cover

    @pytest.mark.xfail
    def test_only_bias(self):
        """Build an ensemble with only bias specified."""
        # Create a Nengo ensemble to build
        ens = nengo.Ensemble(1, 1, add_to_container=False)
        ens.bias = np.array([-0.5])

        # Create a model
        model = builder.Model()
        model.seeds[ens] = 1

        # Build the ensemble
        ensemble.build_ensemble(model, ens)

        # Check that parameters are (loosely) as expected.
        assert model.params[ens].bias == ens.bias  # pragma : no cover


class TestEnsembleLIF(object):
    @pytest.mark.parametrize("size_in", [1, 4, 5])
    def test_init(self, size_in):
        """Test that creating an Ensemble LIF creates an empty list of local
        probes and an empty input vector.
        """
        lif = ensemble.EnsembleLIF(size_in)
        assert np.all(lif.direct_input == np.zeros(size_in))
        assert lif.local_probes == list()
