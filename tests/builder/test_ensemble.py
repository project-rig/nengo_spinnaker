import mock
import nengo
import numpy as np
import pytest

from nengo_spinnaker.builder import builder, ensemble
from nengo_spinnaker import operators


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
        assert isinstance(model.object_operators[ens],
                          operators.EnsembleLIF)

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
    a_ens = operators.EnsembleLIF(a)
    model.object_operators[a] = a_ens

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
        b_ens = operators.EnsembleLIF(b)
        model.object_operators[b] = b_ens

        # Get the sink, check that an appropriate target is return
        sink = ensemble.get_ensemble_sink(model, a_b)
        assert sink.target.obj is b_ens
        assert sink.target.port is builder.InputPort.standard

    def test_normal_sink_for_passthrough_node(self):
        """Test that sinks for most connections into Ensembles do nothing
        special.
        """
        # Create a network and standard model
        with nengo.Network():
            a = nengo.Node(None, size_in=4)
            b = nengo.Ensemble(200, 4)

            a_b = nengo.Connection(a, b)

        # Create a model with the Ensemble for b in it
        model = builder.Model()
        b_ens = operators.EnsembleLIF(b)
        model.object_operators[b] = b_ens

        # Get the sink, check that an appropriate target is return
        sink = ensemble.get_ensemble_sink(model, a_b)
        assert sink.target.obj is b_ens
        assert sink.target.port is builder.InputPort.standard

    def test_normal_sink_for_process_node(self):
        """Test that sinks for most connections into Ensembles do nothing
        special.
        """
        # Create a network and standard model
        with nengo.Network():
            a = nengo.Node(nengo.processes.WhiteNoise(), size_out=4)
            b = nengo.Ensemble(200, 4)

            a_b = nengo.Connection(a, b)

        # Create a model with the Ensemble for b in it
        model = builder.Model()
        b_ens = operators.EnsembleLIF(b)
        model.object_operators[b] = b_ens

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
        b_ens = operators.EnsembleLIF(b)
        model.object_operators[b] = b_ens

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
        b_ens = operators.EnsembleLIF(b)
        model.object_operators[b] = b_ens

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
        b_ens = operators.EnsembleLIF(b)
        model.object_operators[b] = b_ens

        decs = mock.Mock()
        evals = mock.Mock()
        si = mock.Mock()
        model.params[a_b] = builder.BuiltConnection(decs, evals, a_b.transform,
                                                    si)

        # Get the sink, check that an appropriate target is return
        sink = ensemble.get_neurons_sink(model, a_b)
        assert sink.target.obj is b_ens
        assert sink.target.port is ensemble.EnsembleInputPort.global_inhibition

        assert model.params[a_b].decoders is decs
        assert model.params[a_b].eval_points is evals
        assert model.params[a_b].solver_info is si
        assert np.all(model.params[a_b].transform == np.array([[1.0, 0.5]]))
        assert model.params[a_b].transform.shape == (1, 2)

    def test_arbitrary_neuron_sink(self):
        """We have no plan to support arbitrary connections to neurons."""
        with nengo.Network():
            a = nengo.Ensemble(100, 2)
            b = nengo.Ensemble(200, 4)

            a_b = nengo.Connection(a, b.neurons,
                                   transform=[[1.0, 0.5]]*199 + [[0.5, 1.0]])

        # Create a model with the Ensemble for b in it
        model = builder.Model()
        b_ens = operators.EnsembleLIF(b)
        model.object_operators[b] = b_ens

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
        b_ens = operators.EnsembleLIF(b)
        model.object_operators[b] = b_ens

        # Get the sink, check that an appropriate target is return
        sink = ensemble.get_neurons_sink(model, a_b)
        assert sink.target.obj is b_ens
        assert sink.target.port is ensemble.EnsembleInputPort.neurons


class TestBuildFromEnsembleConnection(object):
    """Test the construction of parameters that describe connections from
    Ensembles.
    """
    def test_standard_build(self):
        """Test relatively standard build."""
        # Create the network
        with nengo.Network():
            a = nengo.Ensemble(200, 3)
            b = nengo.Node(lambda t, x: None, size_in=2)
            a_b = nengo.Connection(a[:2], b, transform=0.5*np.eye(2))

        # Create the model and built the pre-synaptic Ensemble
        model = builder.Model()
        model.rng = np.random
        model.seeds[a] = 1
        model.seeds[a_b] = 2
        ensemble.build_ensemble(model, a)

        # Now build the connection and check that the params seem sensible
        params = ensemble.build_from_ensemble_connection(model, a_b)
        assert params.decoders.shape == (200, 2)
        assert np.all(params.transform == 0.5 * np.eye(2))
        assert np.all(params.eval_points == model.params[a].eval_points)
        assert params.solver_info is not None

    @pytest.mark.xfail(reason="Unimplemented functionality")
    def test_weights_built(self):
        """Test a build using a weights-based solver."""
        # Create the network
        with nengo.Network():
            a = nengo.Ensemble(200, 2)
            b = nengo.Ensemble(400, 2)
            a_b = nengo.Connection(
                a, b, solver=nengo.solvers.Lstsq(weights=True)
            )

        # Create the model and built the pre-synaptic Ensemble
        model = builder.Model()
        model.rng = np.random
        model.seeds[a] = 1
        model.seeds[b] = 2
        model.seeds[a_b] = 3
        ensemble.build_ensemble(model, a)
        ensemble.build_ensemble(model, b)

        # Now build the connection and check that the params seem sensible
        params = ensemble.build_from_ensemble_connection(model, a_b)
        assert params.decoders.shape == (200, 400)


class TestBuildFromNeuronsConnection(object):
    """Test the construction of parameters that describe connections from
    Neurons.
    """
    @pytest.mark.xfail(reason="Unimplemented functionality")
    def test_standard_build(self):
        # Create the network
        with nengo.Network():
            a = nengo.Ensemble(100, 2)
            b = nengo.Ensemble(100, 3)
            a_b = nengo.Connection(a.neurons, b.neurons)

        # Get the connection parameters
        params = ensemble.build_from_neurons_connection(None, a_b)
        assert params.decoders is None
        assert np.all(params.transform == np.eye(100))
        assert params.eval_points is None
        assert params.solver_info is None


class TestProbeEnsemble(object):
    """Test probing ensembles."""
    @pytest.mark.parametrize("with_slice", [False, True])
    def test_probe_output_with_sampling(self, with_slice):
        """Test that probing the output of an Ensemble generates a new
        connection and a new object.
        """
        with nengo.Network() as net:
            a = nengo.Ensemble(100, 3)

            if not with_slice:
                p = nengo.Probe(a, sample_every=0.0023)
            else:
                p = nengo.Probe(a[0:1], sample_every=0.0023)

        # Create an empty model to build the probe into
        model = builder.Model()
        model.build(net)

        # Check that a new connection was added and built
        assert len(model.connections_signals) == 1
        for conn in model.connections_signals.keys():
            assert conn.pre_obj is a
            assert conn.post_obj is p
            assert conn in model.params  # Was it built?

            if with_slice:
                assert conn.pre_slice == p.slice

        # Check that a new object was added to the model
        vs = model.object_operators[p]
        assert isinstance(vs, operators.ValueSink)
        assert vs.probe is p

    def test_probe_output_no_sampling(self):
        """Test that probing the output of an Ensemble generates a new
        connection and a new object.
        """
        with nengo.Network() as net:
            a = nengo.Ensemble(100, 3)
            p = nengo.Probe(a)

        # Create an empty model to build the probe into
        model = builder.Model()
        model.build(net)

        # Check that a new object was added to the model
        vs = model.object_operators[p]
        assert vs.sample_every == 1

    @pytest.mark.xfail(reason="Unimplemented functionality")
    def test_probe_input(self):
        """Test probing the input of an Ensemble."""
        with nengo.Network():
            a = nengo.Ensemble(100, 3)
            p = nengo.Probe(a, "input")

        # Create an empty model to build the probe into
        model = builder.Model()
        model.rng = np.random
        model.seeds[p] = 1

        # Build the probe
        ensemble.build_ensemble_probe(model, p)


class TestProbeNeurons(object):
    """Test probing neurons."""
    def test_probe_spikes(self):
        """Check that probing spikes modifies the local_probes list on the
        operator, but does nothing else.
        """
        with nengo.Network() as net:
            a = nengo.Ensemble(300, 1)
            p = nengo.Probe(a.neurons, "spikes")

        # Create an empty model to build the probe into
        model = builder.Model()
        model.build(net)

        # Assert that we added the probe to the list of local probes and
        # nothing else
        assert model.object_operators[a].local_probes == [p]
        assert len(model.object_operators) == 1
        assert len(model.connections_signals) == 0

    def test_probe_spike_slice(self):
        with nengo.Network() as net:
            a = nengo.Ensemble(300, 1)
            p = nengo.Probe(a.neurons[:100], "spikes")

        # Create an empty model to build the probe into
        model = builder.Model()
        model.build(net)

        # Assert that we added the probe to the list of local probes and
        # nothing else
        assert model.object_operators[a].local_probes == [p]
        assert len(model.object_operators) == 1
        assert len(model.connections_signals) == 0

    def test_probe_voltage(self):
        """Check that probing voltage modifies the local_probes list on the
        operator, but does nothing else.
        """
        with nengo.Network() as net:
            a = nengo.Ensemble(300, 1)
            p = nengo.Probe(a.neurons, "voltage")

        # Create an empty model to build the probe into
        model = builder.Model()
        model.build(net)

        # Assert that we added the probe to the list of local probes and
        # nothing else
        assert model.object_operators[a].local_probes == [p]
        assert len(model.object_operators) == 1
        assert len(model.connections_signals) == 0

    @pytest.mark.xfail(reason="Unimplemented functionality")
    def test_refractory_time(self):
        """Check that probing refractory time modifies the local_probes list on
        the operator, but does nothing else.
        """
        with nengo.Network() as net:
            a = nengo.Ensemble(300, 1)
            p = nengo.Probe(a.neurons, "refractory_time")

        # Create an empty model to build the probe into
        model = builder.Model()
        model.build(net)

        # Assert that we added the probe to the list of local probes and
        # nothing else
        assert model.object_operators[a].local_probes == [p]
        assert len(model.object_operators) == 1
        assert len(model.connections_signals) == 0
