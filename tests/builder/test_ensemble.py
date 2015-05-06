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

    @pytest.mark.xfail(reason="Unimplemented functionality")
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

    @pytest.mark.xfail(reason="Unimplemented functionality")
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


@pytest.mark.xfail(reason="Unimplemented functionality")
def test_neurons_source():
    """Test that neurons sources are sane."""
    with nengo.Network():
        a = nengo.Ensemble(100, 2)
        b = nengo.Ensemble(100, 4)

        a_b = nengo.Connection(a.neurons, b.neurons, transform=np.eye(100))

    # Create a model with the Ensemble for a in it
    model = builder.Model()
    a_ens = ensemble.EnsembleLIF(a.size_in)
    model.object_intermediates[a] = a_ens

    # Get the source, check that an appropriate target is return
    source = ensemble.get_neurons_source(model, a_b)
    assert source.target.obj is a_ens
    assert source.target.port is ensemble.EnsembleOutputPort.neurons


class TestEnsembleSink(object):
    def test_normal_sink(self):
        """Test that sinks for most connections into Ensembles do nothing
        special.
        """
        # Create a network and standard model
        with nengo.Network():
            a = nengo.Ensemble(100, 2)
            b = nengo.Ensemble(200, 4)

            a_b = nengo.Connection(a, b[1:3])

        # Create a model with the Ensemble for b in it
        model = builder.Model()
        b_ens = ensemble.EnsembleLIF(b.size_in)
        model.object_intermediates[b] = b_ens

        # Get the sink, check that an appropriate target is return
        sink = ensemble.get_ensemble_sink(model, a_b)
        assert sink.target.obj is b_ens
        assert sink.target.port is builder.InputPort.standard

    def test_constant_node_sink_with_slice(self):
        """Test that connections from constant valued Nodes to Ensembles are
        optimised out correctly.
        """
        # Create a network and standard model
        with nengo.Network():
            a = nengo.Node([0.5, 1.0])
            b = nengo.Ensemble(200, 2)

            a_b = nengo.Connection(a[0], b[1])

        # Create a model with the Ensemble for b in it
        model = builder.Model()
        b_ens = ensemble.EnsembleLIF(b.size_in)
        model.object_intermediates[b] = b_ens

        # Check that no sink is created but that the direct input is modified
        assert np.all(b_ens.direct_input == np.zeros(2))
        assert ensemble.get_ensemble_sink(model, a_b) is None
        assert np.all(b_ens.direct_input == [0.0, 0.5])

    def test_constant_node_sink_with_function(self):
        """Test that connections from constant valued Nodes to Ensembles are
        optimised out correctly.
        """
        # Create a network and standard model
        with nengo.Network():
            a = nengo.Node([0.5, 1.0])
            b = nengo.Ensemble(200, 2)

            a_b = nengo.Connection(a, b, function=lambda x: x**2,
                                   transform=[[0.0, -1.0], [-1.0, 0.0]])

        # Create a model with the Ensemble for b in it
        model = builder.Model()
        b_ens = ensemble.EnsembleLIF(b.size_in)
        model.object_intermediates[b] = b_ens

        # Check that no sink is created but that the direct input is modified
        assert np.all(b_ens.direct_input == np.zeros(2))
        assert ensemble.get_ensemble_sink(model, a_b) is None
        assert np.all(b_ens.direct_input == [-1.0, -0.25])


class TestNeuronSinks(object):
    def test_global_inhibition_sink(self):
        """Test that sinks are correctly determined for connections which are
        global inhibition connections.
        """
        # Create a model with a global inhibition connection
        with nengo.Network():
            a = nengo.Ensemble(100, 2)
            b = nengo.Ensemble(200, 4)

            a_b = nengo.Connection(a, b.neurons, transform=[[1.0, 0.5]]*200)

        # Create a model with the Ensemble for b in it
        model = builder.Model()
        b_ens = ensemble.EnsembleLIF(b.size_in)
        model.object_intermediates[b] = b_ens

        # Get the sink, check that an appropriate target is return
        sink = ensemble.get_neurons_sink(model, a_b)
        assert sink.target.obj is b_ens
        assert sink.target.port is ensemble.EnsembleInputPort.global_inhibition

    def test_arbitrary_neuron_sink(self):
        """We have no plan to support arbitrary connections to neurons."""
        with nengo.Network():
            a = nengo.Ensemble(100, 2)
            b = nengo.Ensemble(200, 4)

            a_b = nengo.Connection(a, b.neurons,
                                   transform=[[1.0, 0.5]]*199 + [[0.5, 1.0]])

        # Create a model with the Ensemble for b in it
        model = builder.Model()
        b_ens = ensemble.EnsembleLIF(b.size_in)
        model.object_intermediates[b] = b_ens

        # This should fail
        with pytest.raises(NotImplementedError):
            ensemble.get_neurons_sink(model, a_b)

    def test_neuron_sink(self):
        """Test that standard connections to neurons return an appropriate
        sink.
        """
        with nengo.Network():
            a = nengo.Ensemble(100, 2)
            b = nengo.Ensemble(100, 4)

            a_b = nengo.Connection(a.neurons, b.neurons, transform=np.eye(100))

        # Create a model with the Ensemble for b in it
        model = builder.Model()
        b_ens = ensemble.EnsembleLIF(b.size_in)
        model.object_intermediates[b] = b_ens

        # Get the sink, check that an appropriate target is return
        sink = ensemble.get_neurons_sink(model, a_b)
        assert sink.target.obj is b_ens
        assert sink.target.port is ensemble.EnsembleInputPort.neurons


class TestEnsembleLIF(object):
    @pytest.mark.parametrize("size_in", [1, 4, 5])
    def test_init(self, size_in):
        """Test that creating an Ensemble LIF creates an empty list of local
        probes and an empty input vector.
        """
        lif = ensemble.EnsembleLIF(size_in)
        assert np.all(lif.direct_input == np.zeros(size_in))
        assert lif.local_probes == list()
