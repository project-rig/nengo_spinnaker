import mock
import nengo
from nengo.utils.builder import full_transform
import numpy as np
import pytest

from nengo_spinnaker.ensemble import annotations as ns_ens
from nengo_spinnaker import annotations as anns


class TestIntermediateEnsemble(object):
    @pytest.mark.parametrize(
        "size_in",
        [1, 4, 9]
    )
    def test_init(self, size_in):
        """Test that the init correctly initialise the direct input and the
        list of local probes.
        """
        with nengo.Network():
            a = nengo.Ensemble(100, size_in)

        # Create the intermediate representation
        o = anns.Annotations.object_builders[nengo.Ensemble](a, None)
        assert o.constraints == list()
        assert np.all(o.direct_input == np.zeros(size_in))
        assert o.local_probes == list()


class TestGetEnsembleSink(object):
    def test_get_sink_standard(self):
        """Test that get_sink_standard just does a look up in the object map
        dictionary and uses InputPort.standard.
        """
        with nengo.Network():
            a = nengo.Ensemble(300, 4)
            b = nengo.Ensemble(300, 2)
            c = nengo.Connection(a[:2], b)

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: mock.Mock(name="ir_b", spec_set=[]),
        }

        irn = anns.Annotations(obj_map, {}, [], [])
        assert (
            ns_ens.get_ensemble_sink(c, irn) ==
            anns.soss(anns.NetAddress(obj_map[b], anns.InputPort.standard))
        )

    def test_get_sink_constant_node(self):
        """Test that if the "pre" object is a constant valued Node that None is
        returned and that the IntermediateEnsemble is modified.
        """
        with nengo.Network():
            a = nengo.Node([2.0, -0.25])
            b = nengo.Ensemble(300, 2)

            c = nengo.Connection(a[0], b[1], transform=5,
                                 function=lambda x: x+2)
            d = nengo.Connection(a[0], b[0])

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: ns_ens.AnnotatedEnsemble(b)
        }

        # We don't return a sink (None means "no connection required")
        irn = anns.Annotations(obj_map, {}, [], [])
        assert ns_ens.get_ensemble_sink(c, irn) is None

        # But the Node values are added into the intermediate representation
        # for the ensemble with the connection transform and function applied.
        assert np.all(obj_map[b].direct_input ==
                      np.dot(full_transform(c, slice_pre=False),
                             c.function(a.output[c.pre_slice])))

        # For the next connection assert that we again don't add a connection
        # and that the direct input is increased.
        assert ns_ens.get_ensemble_sink(d, irn) is None
        assert np.all(obj_map[b].direct_input ==
                      np.dot(full_transform(c, slice_pre=False),
                             c.function(a.output[c.pre_slice])) +
                      np.dot(full_transform(d), a.output))


class TestGetNeuronsSink(object):
    def test_neurons_to_neurons(self):
        """Test that get_neurons_sink correctly returns the Ensemble as the
        object and InputPort.neurons as the port.
        """
        with nengo.Network():
            a = nengo.Ensemble(300, 4)
            b = nengo.Ensemble(300, 2)
            c = nengo.Connection(a.neurons, b.neurons)

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: mock.Mock(name="ir_b", spec_set=[]),
        }

        irn = anns.Annotations(obj_map, {}, [], [])
        irn = anns.Annotations(obj_map, {}, [], [])
        assert (
            ns_ens.get_neurons_sink(c, irn) ==
            anns.soss(anns.NetAddress(obj_map[b], anns.InputPort.neurons))
        )

    @pytest.mark.parametrize(
        "a",  # a is the originator of a connection into some neurons
        [nengo.Node(lambda t: t**2, size_in=0, size_out=1,
                    add_to_container=False),
         nengo.Ensemble(100, 1, add_to_container=False),
         ]
    )
    def test_global_inhibition(self, a):
        """Test that get_neurons_sink correctly returns the target as the
        Ensemble and the port as global_inhibition.
        """
        b = nengo.Ensemble(100, 2, add_to_container=False)
        c = nengo.Connection(a, b.neurons, transform=[[1]]*b.n_neurons,
                             add_to_container=False)

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: mock.Mock(name="ir_b", spec_set=[]),
        }

        irn = anns.Annotations(obj_map, {}, [], [])
        assert (
            ns_ens.get_neurons_sink(c, irn) ==
            anns.soss(
                anns.NetAddress(obj_map[b], anns.InputPort.global_inhibition)
            )
        )

    def test_other(self):
        with nengo.Network():
            a = nengo.Ensemble(300, 1)
            b = nengo.Ensemble(300, 2)
            c = nengo.Connection(a, b.neurons, function=lambda x: [x]*300)

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: mock.Mock(name="ir_b", spec_set=[]),
        }

        irn = anns.Annotations(obj_map, {}, [], [])
        with pytest.raises(NotImplementedError):
            ns_ens.get_neurons_sink(c, irn)


class TestGetNeuronsSource(object):
    """Test getting the source for connections from neurons."""
    def test_first_time(self):
        with nengo.Network():
            a = nengo.Ensemble(100, 2)
            b = nengo.Ensemble(300, 3)
            c = nengo.Connection(a.neurons, b.neurons[:100])

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: mock.Mock(name="ir_b", spec_set=[]),
        }

        irn = anns.Annotations(obj_map, {}, [], [])
        assert (
            ns_ens.get_neurons_source(c, irn) ==
            anns.soss(anns.NetAddress(obj_map[a], anns.OutputPort.neurons),
                      weight=a.n_neurons)
        )

    def test_second_time(self):
        with nengo.Network():
            a = nengo.Ensemble(100, 2)
            b = nengo.Ensemble(300, 3)
            c = nengo.Ensemble(100, 2)

            d = nengo.Connection(a.neurons, b.neurons[:100])
            e = nengo.Connection(a.neurons, c.neurons)

        obj_map = {
            a: mock.Mock(name="ir_a", spec_set=[]),
            b: mock.Mock(name="ir_b", spec_set=[]),
            c: mock.Mock(name="ir_c", spec_set=[]),
        }
        conn_map = {
            d: anns.AnnotatedNet(
                anns.NetAddress(obj_map[a], anns.OutputPort.neurons),
                anns.NetAddress(obj_map[b], anns.InputPort.neurons)
            )
        }

        # Building connection e (having built d) should result in modification
        # of the connection and the use of the same connection (indicated by
        # returning the existing connection).
        irn = anns.Annotations(obj_map, conn_map, [], [])
        assert ns_ens.get_neurons_source(e, irn) == conn_map[d]


def test_get_neurons_probe():
    """Test building probes for Neuron-type objects."""
    with nengo.Network():
        a = nengo.Ensemble(300, 2)
        p = nengo.Probe(a.neurons)

    # Get the IR for the ensemble
    ir_a = ns_ens.AnnotatedEnsemble(a)
    assert ir_a.local_probes == list()

    # Building the probe should just add it to the intermediate representation
    # for `a`'s list of local probes.
    assert (
        ns_ens.get_neurons_probe(
            p, 3345, anns.Annotations({a: ir_a}, {}, (), ())) ==
        (None, [], [])
    )
    assert ir_a.local_probes == [p]
