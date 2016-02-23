import mock
import nengo
import numpy as np
import pytest

from nengo_spinnaker.utils import model as model_utils
from nengo_spinnaker.builder import model

from nengo_spinnaker.builder.ensemble import EnsembleTransmissionParameters
from nengo_spinnaker.builder.node import (
    PassthroughNodeTransmissionParameters, NodeTransmissionParameters
)
from nengo_spinnaker.builder.ports import EnsembleInputPort


def test_get_force_removal_passnodes():
    """Passthrough Nodes in networks of passthrough Nodes that connect to
    neurons need to be marked for removal.
    """
    # Construct a network with three sets of passthrough Nodes
    with nengo.Network() as model:
        # Spurious Node that should be ignored
        nengo.Node(lambda t: t)

        # First set of passthrough Nodes
        # 0 --\
        # 1 --> 3 ---> 4 --> Neurons
        # 2 --/   \--> 5 --> Ensemble
        in_a = nengo.Node(np.zeros(100))
        out_a = nengo.Ensemble(100, 1)

        set_a = list(nengo.Node(size_in=100, label="A{}".format(n))
                     for n in range(6))

        for node in set_a[:3]:
            nengo.Connection(in_a, node)
            nengo.Connection(node, set_a[3])

        for node in set_a[4:]:
            nengo.Connection(set_a[3], node)

        nengo.Connection(set_a[4], out_a.neurons)
        nengo.Connection(set_a[5], out_a, transform=np.zeros((1, 100)))

        # Second set of passthrough Nodes
        # Neurons -> 0 --\
        # Value ---> 1 --> 2 --> Value
        with nengo.Network():
            in_b = nengo.Ensemble(200, 1)
            set_b = list(nengo.Node(size_in=1, label="B{}".format(n))
                         for n in range(3))

            nengo.Connection(in_b.neurons, set_b[0],
                             transform=np.ones((1, 200)))
            nengo.Connection(in_b, set_b[1])

            nengo.Connection(set_b[0], set_b[2])
            nengo.Connection(set_b[1], set_b[2])

            nengo.Connection(set_b[2], in_b)

        # Third set of passthrough Nodes
        # Ensemble -> 0 -> 1 -> Ensemble
        set_c = list(nengo.Node(size_in=2, label="C{}".format(n))
                     for n in range(2))
        in_c = nengo.Ensemble(100, 2)

        nengo.Connection(in_c, set_c[0])
        nengo.Connection(set_c[0], set_c[1])
        nengo.Connection(set_c[1], in_c)

    # Get whether the passthrough Nodes should be marked for removal
    assert model_utils.get_force_removal_passnodes(model) == set(set_a + set_b)


def test_remove_operator_from_connection_map():
    """Test that operators are correctly removed from connection maps.

    We test the following model:

        O1 ------> O3 -----> O5
            /-----/  \
           /          \----> O6
        O2 --> O4

    Removing `O3' should result in:

        O1 --> O6
            /
           /-> 05
        O2 --> O4
    """
    # Construct the operators
    operators = [mock.Mock(name="O{}".format(i + 1)) for i in range(6)]

    # Create a connection map
    cm = model.ConnectionMap()

    # Add the connection O1 to O3
    sps = model.SignalParameters(True, 6)
    tps = PassthroughNodeTransmissionParameters(np.vstack([np.eye(3),
                                                           np.zeros((3, 3))]))
    rps = model.ReceptionParameters(None, 6)
    
    cm.add_connection(source_object=operators[0],
                      source_port=None,
                      signal_parameters=sps,
                      transmission_parameters=tps,
                      sink_object=operators[2],
                      sink_port=None,
                      reception_parameters=rps)

    # Add the connection O2 to O3
    sps = model.SignalParameters(False, 6)
    tps = PassthroughNodeTransmissionParameters(np.vstack([np.zeros((3, 3)),
                                                           np.eye(3)]))
    rps = model.ReceptionParameters(None, 6)
    
    cm.add_connection(source_object=operators[1],
                      source_port=None,
                      signal_parameters=sps,
                      transmission_parameters=tps,
                      sink_object=operators[2],
                      sink_port=None,
                      reception_parameters=rps)

    # Add the connection O2 to O4 (with a custom keyspace)
    sps = model.SignalParameters(False, 6, mock.Mock("Keyspace 1"))
    tps = PassthroughNodeTransmissionParameters(np.vstack([np.eye(3),
                                                           np.eye(3)]))
    rps = model.ReceptionParameters(None, 6)
    
    cm.add_connection(source_object=operators[1],
                      source_port=None,
                      signal_parameters=sps,
                      transmission_parameters=tps,
                      sink_object=operators[3],
                      sink_port=None,
                      reception_parameters=rps)

    # Add the connection O3 to O5
    sps = model.SignalParameters(False, 3)
    tps = PassthroughNodeTransmissionParameters(np.hstack((np.zeros((3, 3)),
                                                           np.eye(3))))
    rps = model.ReceptionParameters(None, 3)
    
    cm.add_connection(source_object=operators[2],
                      source_port=None,
                      signal_parameters=sps,
                      transmission_parameters=tps,
                      sink_object=operators[4],
                      sink_port=None,
                      reception_parameters=rps)

    # Add the connection O3 to O6
    sps = model.SignalParameters(False, 3)
    tps = PassthroughNodeTransmissionParameters(np.hstack([np.eye(3),
                                                           np.eye(3)]) * 2)
    rps = model.ReceptionParameters(None, 3)
    
    cm.add_connection(source_object=operators[2],
                      source_port=None,
                      signal_parameters=sps,
                      transmission_parameters=tps,
                      sink_object=operators[5],
                      sink_port=None,
                      reception_parameters=rps)

    # Remove O3 from the connection map
    model_utils.remove_operator_from_connection_map(cm, operators[2])

    # Check that the received and transmitted signals are as expected
    # FROM O1
    from_o1 = cm._connections[operators[0]]
    assert len(from_o1) == 1
    assert len(from_o1[None]) == 1

    ((signal_parameters, transmission_parameters), sinks) = from_o1[None][0]
    assert signal_parameters == model.SignalParameters(True, 3, None)
    assert transmission_parameters.transform.shape == (3, 3)
    assert np.all(transmission_parameters.transform == np.eye(3)*2)
    assert sinks == [(operators[5], None, model.ReceptionParameters(None, 3))]

    # FROM O2
    from_o2 = cm._connections[operators[1]]
    assert len(from_o2) == 1
    assert len(from_o2[None]) == 3

    for ((signal_parameters, transmission_parameters), sinks) in from_o2[None]:
        if transmission_parameters.transform.shape == (3, 3):
            assert (signal_parameters ==
                    model.SignalParameters(False, 3, None))

            if np.any(transmission_parameters.transform == 2.0):
                # TO O6
                assert np.all(transmission_parameters.transform ==
                              np.eye(3) * 2)
                assert sinks == [(operators[5], None,
                                  model.ReceptionParameters(None, 3))]
            else:
                # TO O5
                assert np.all(transmission_parameters.transform ==
                              np.eye(3))
                assert sinks == [(operators[4], None,
                                  model.ReceptionParameters(None, 3))]
        else:
            # TO O4
            assert transmission_parameters.transform.shape == (6, 3)
            assert np.all(transmission_parameters.transform ==
                          np.vstack([np.eye(3)]*2))
            assert sinks == [(operators[3], None,
                              model.ReceptionParameters(None, 6))]

    # We now add a connection from O4 to O6 with a custom keyspace.  Removing
    # O4 will fail because keyspaces can't be merged.
    signal_params = model.SignalParameters(False, 1, mock.Mock("Keyspace 2"))
    transmission_params = PassthroughNodeTransmissionParameters(1.0)
    reception_params = model.ReceptionParameters(None, 1)

    cm.add_connection(
        source_object=operators[3],
        source_port=None,
        signal_parameters=signal_params,
        transmission_parameters=transmission_params,
        sink_object=operators[5],
        sink_port=None,
        reception_parameters=reception_params
    )

    with pytest.raises(NotImplementedError) as err:
        model_utils.remove_operator_from_connection_map(
            cm, operators[3]
        )
    assert "keyspace" in str(err.value).lower()


def test_remove_operator_from_connection_map_unforced():
    """Check that a calculation is made to determine whether it is better to
    keep or remove an operator depending on the density of the outgoing
    connections. In the example:

                                               /- G[0]
        A[0] --\         /-- D[0] --\         /-- G[1]
        A[1] --- B --- C --- D[1] --- E --- F --- G[2]
        A[n] --/         \-- D[n] --/         \-- G[3]
                                               \- G[n]

    B, C and F should be removed but E should be retained:

                             /- G[0]
        A[0] --- D[0] --\   /-- G[1]
        A[1] --- D[1] --- E --- G[2]
        A[n] --- D[n] --/   \-- G[3]
                             \- G[n]
    """
    # Create the operators
    D = 512
    SD = 16

    op_A = [mock.Mock(name="A{}".format(i)) for i in range(D//SD)]
    op_B = mock.Mock(name="B")
    op_C = mock.Mock(name="C")
    op_D = [mock.Mock(name="D{}".format(i)) for i in range(D//SD)]
    op_E = mock.Mock(name="E")
    op_F = mock.Mock(name="F")
    op_G = [mock.Mock(name="G{}".format(i)) for i in range(D)]

    # Create a connection map
    cm = model.ConnectionMap()

    # Create the fan-in connections
    for sources, sink in ((op_A, op_B), (op_D, op_E)):
        # Get the signal and reception parameters
        sps = model.SignalParameters(True, D)
        rps = model.ReceptionParameters(None, D)

        for i, source in enumerate(sources):
            # Get the transform
            transform = np.zeros((D, SD))
            transform[i*SD:(i+1)*SD, :] = np.eye(SD)

            # Get the parameters
            tps = EnsembleTransmissionParameters(np.ones((1, SD)), transform)
    
            cm.add_connection(source_object=source, source_port=None,
                              signal_parameters=sps,
                              transmission_parameters=tps,
                              sink_object=sink, sink_port=None,
                              reception_parameters=rps)

    # Create the fan-out connection C to D[...]
    # Get the signal and reception parameters
    sps = model.SignalParameters(True, SD)
    rps = model.ReceptionParameters(None, SD)

    for i, sink in enumerate(op_D):
        # Get the transform
        transform = np.zeros((SD, D))
        transform[:, i*SD:(i+1)*SD] = np.eye(SD)

        # Get the parameters
        tps = PassthroughNodeTransmissionParameters(transform)

        cm.add_connection(source_object=op_C, source_port=None,
                          signal_parameters=sps,
                          transmission_parameters=tps,
                          sink_object=sink, sink_port=None,
                          reception_parameters=rps)

    # Create the connection B to C
    sps = model.SignalParameters(True, D)
    rps = model.ReceptionParameters(None, D)
    tps = PassthroughNodeTransmissionParameters(np.eye(D))

    cm.add_connection(source_object=op_B, source_port=None,
                      signal_parameters=sps,
                      transmission_parameters=tps,
                      sink_object=op_C, sink_port=None,
                      reception_parameters=rps)

    # Create the connection E to F
    transform = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            transform[i, j] = i + j

    sps = model.SignalParameters(True, D)
    rps = model.ReceptionParameters(None, D)
    tps = PassthroughNodeTransmissionParameters(transform)

    cm.add_connection(source_object=op_E, source_port=None,
                      signal_parameters=sps,
                      transmission_parameters=tps,
                      sink_object=op_F, sink_port=None,
                      reception_parameters=rps)

    # Create the fan-out connections from F
    sps = model.SignalParameters(True, SD)
    rps = model.ReceptionParameters(None, SD)

    for i, sink in enumerate(op_G):
        # Get the transform
        transform = np.zeros((1, D))
        transform[:, i] = 1.0

        # Get the parameters
        tps = PassthroughNodeTransmissionParameters(transform)

        cm.add_connection(source_object=op_F, source_port=None,
                          signal_parameters=sps,
                          transmission_parameters=tps,
                          sink_object=sink, sink_port=None,
                          reception_parameters=rps)

    # Remove all of the passthrough Nodes, only E should be retained
    assert model_utils.remove_operator_from_connection_map(cm, op_B,
                                                           force=False)
    assert model_utils.remove_operator_from_connection_map(cm, op_C,
                                                           force=False)
    assert not model_utils.remove_operator_from_connection_map(cm, op_E,
                                                               force=False)
    assert model_utils.remove_operator_from_connection_map(cm, op_F,
                                                           force=False)

    # Check that each A has only one outgoing signal and that it terminates at
    # the paired D.  Additionally check that each D has only one outgoing
    # signal and that it terminates at E.
    for a, d in zip(op_A, op_D):
        # Connections from A[n]
        from_a = cm._connections[a]
        assert len(from_a) == 1
        assert len(from_a[None]) == 1

        ((signal_parameters, transmission_parameters), sinks) = from_a[None][0]
        assert signal_parameters == model.SignalParameters(True, SD, None)
        assert transmission_parameters.transform.shape == (SD, SD)
        assert np.all(transmission_parameters.transform == np.eye(SD))
        assert sinks == [(d, None, model.ReceptionParameters(None, SD))]

        # Connection(s) from D[n]
        from_d = cm._connections[d]
        assert len(from_d) == 1
        assert len(from_d[None]) == 1

        ((signal_parameters, transmission_parameters), sinks) = from_d[None][0]
        assert signal_parameters == model.SignalParameters(True, D, None)
        assert transmission_parameters.transform.shape == (D, SD)

    # Check that there are many connections from E
    from_e = cm._connections[op_E]
    assert len(from_e) == 1
    print(from_e[None][0].parameters[1].transform)
    assert len(from_e[None]) == D


class TestCombineTransmissionAndReceptionParameters(object):
    """Test the correct combination of transmission parameters."""
    # Ensemble and Passthrough Node to StandardInput or Neurons - NOT global
    # inhibition
    @pytest.mark.parametrize("final_port", (model.InputPort.standard,
                                            EnsembleInputPort.neurons))
    def test_ens_to_x(self, final_port):
        # Create the ingoing connection parameters
        in_transmission_params = EnsembleTransmissionParameters(
            np.random.uniform(size=(100, 10)), 1.0
        )

        # Create the outgoing connection parameters
        out_transmission_params = PassthroughNodeTransmissionParameters(
            np.hstack([np.eye(5), np.zeros((5, 5))])
        )

        # Combine the parameter sets
        new_tps, new_in_port = model_utils._combine_transmission_params(
                in_transmission_params,
                out_transmission_params,
                final_port
            )

        # Check that all the parameters are correct
        assert np.all(new_tps.untransformed_decoders ==
                      in_transmission_params.untransformed_decoders)
        assert np.all(new_tps.transform ==
                      out_transmission_params.transform)
        assert new_tps.decoders.shape == (5, 100)
        assert new_in_port is final_port

    # Node and Passthrough Node to Standard Input or Neurons - NOT global
    # inhibition
    @pytest.mark.parametrize("passthrough", (False, True))
    @pytest.mark.parametrize("final_port", (model.InputPort.standard,
                                            EnsembleInputPort.neurons))
    def test_node_to_x(self, passthrough, final_port):
        # Create the ingoing connection parameters
        if not passthrough:
            in_transmission_params = NodeTransmissionParameters(
                slice(10, 15),
                mock.Mock(),
                np.random.uniform(size=(10, 5)),
            )
        else:
            in_transmission_params = PassthroughNodeTransmissionParameters(
                np.random.uniform(size=(10, 5)),
            )

        # Create the outgoing connection parameters
        out_transmission_params = PassthroughNodeTransmissionParameters(
            np.hstack((np.zeros((5, 5)), np.eye(5)))
        )

        # Combine the parameter sets
        new_tps, new_in_port = model_utils._combine_transmission_params(
                in_transmission_params,
                out_transmission_params,
                final_port
            )

        # Check that all the parameters are correct
        if not passthrough:
            assert new_tps.pre_slice == in_transmission_params.pre_slice
            assert new_tps.function is in_transmission_params.function

        assert np.all(new_tps.transform == np.dot(
            out_transmission_params.transform,
            in_transmission_params.transform
        ))
        assert new_tps.transform.shape == (5, 5)
        assert new_in_port is final_port

    # Ensemble and Passthrough Node to Neurons (Global Inhibition)
    def test_ens_to_gi(self):
        # Create the ingoing connection parameters
        in_transmission_params = EnsembleTransmissionParameters(
            np.random.uniform(size=(100, 7)), 1.0
        )

        # Create the outgoing connection parameters
        out_transmission_params = PassthroughNodeTransmissionParameters(
            np.ones((200, 7))
        )

        # Combine the parameter sets
        new_tps, new_in_port = model_utils._combine_transmission_params(
                in_transmission_params,
                out_transmission_params,
                EnsembleInputPort.neurons
            )

        # Check that all the parameters are correct
        assert np.all(new_tps.transform == 1.0)
        assert new_tps.transform.shape == (1, 7)
        assert new_tps.decoders.shape == (1, 100)
        assert new_in_port is EnsembleInputPort.global_inhibition

    # Node and Passthrough Node to Neurons (Global Inhibition)
    @pytest.mark.parametrize("passthrough", (False, True))
    def test_node_to_gi(self, passthrough):
        # Create the ingoing connection parameters
        if not passthrough:
            in_transmission_params = NodeTransmissionParameters(
                slice(10, 20),
                mock.Mock(),
                np.ones((100, 1))
            )
        else:
            in_transmission_params = PassthroughNodeTransmissionParameters(
                np.ones((100, 1))
            )

        # Create the outgoing connection parameters
        out_transmission_params = PassthroughNodeTransmissionParameters(
            np.eye(100)
        )

        # Combine the parameter sets
        new_tps, new_in_port = model_utils._combine_transmission_params(
                in_transmission_params, out_transmission_params,
                EnsembleInputPort.neurons
            )

        # Check that all the parameters are correct
        if not passthrough:
            assert new_tps.pre_slice == in_transmission_params.pre_slice
            assert new_tps.function is in_transmission_params.function

        assert np.all(new_tps.transform == np.dot(
            out_transmission_params.transform,
            in_transmission_params.transform
        )[0])
        assert new_tps.transform.shape[0] == 1
        assert new_in_port is EnsembleInputPort.global_inhibition

    @pytest.mark.parametrize("from_type", ("ensemble", "node", "ptn"))
    @pytest.mark.parametrize("final_port",
                             (model.InputPort.standard,
                              EnsembleInputPort.neurons,
                              EnsembleInputPort.global_inhibition))
    def test_x_to_x_optimize_out(self, from_type, final_port):
        # Create the ingoing connection parameters
        in_transmission_params = {
            "ensemble": EnsembleTransmissionParameters(
                            np.random.uniform(size=(100, 1)),
                            np.array([[1.0], [0.0]])
                        ),
            "node": NodeTransmissionParameters(
                        slice(10, 20),
                        mock.Mock(),
                        np.array([[1.0], [0.0]])
                    ),
            "ptn": PassthroughNodeTransmissionParameters(
                       np.array([[1.0], [0.0]])
                   ),
        }[from_type]

        # Create the outgoing connection parameters
        out_transmission_params = PassthroughNodeTransmissionParameters(
            np.array([[0.0, 0.0], [0.0, 1.0]])
        )

        # Combine the parameter sets
        new_tps, new_in_port = model_utils._combine_transmission_params(
                in_transmission_params,
                out_transmission_params,
                final_port
            )

        # Check that the connection is optimised out
        assert new_tps is None
        assert new_in_port is None

    def test_unknown_to_x(self):
        # Create the ingoing connection parameters
        in_transmission_params = mock.Mock()
        in_transmission_params.transform = 1.0

        # Create the outgoing connection parameters
        out_transmission_params = PassthroughNodeTransmissionParameters(
            np.array([[0.0, 0.0], [0.0, 1.0]])
        )

        # Combine the parameter sets
        with pytest.raises(NotImplementedError):
             model_utils._combine_transmission_params(
                    in_transmission_params,
                    out_transmission_params,
                    None
                )

    # Test combining reception parameters
    def test_combine_none_and_lowpass_filter(self):
        # Create the ingoing reception parameters
        reception_params_a = model.ReceptionParameters(nengo.Lowpass(0.05), 1)

        # Create the outgoing reception parameters
        reception_params_b = model.ReceptionParameters(None, 3)

        # Combine the parameter each way round
        for a, b in ((reception_params_a, reception_params_b),
                     (reception_params_a, reception_params_b)):
            new_rps = model_utils._combine_reception_params(a, b)

            # Check filter type
            assert new_rps.filter == reception_params_a.filter

            # Check width is the width of the receiving item
            assert new_rps.width == b.width

    def test_combine_linear_and_linear_filter(self):
        # Create the ingoing reception parameters
        reception_params_a = model.ReceptionParameters(nengo.Lowpass(0.05), 1)

        # Create the outgoing reception parameters
        reception_params_b = model.ReceptionParameters(nengo.Lowpass(0.01), 5)

        # Combine the parameter each way round
        for a, b in ((reception_params_a, reception_params_b),
                     (reception_params_a, reception_params_b)):
            new_rps = model_utils._combine_reception_params(a, b)

            # Check filter type
            synapse = new_rps.filter
            assert synapse.num == [1]
            assert np.all(synapse.den == [0.05 * 0.01, 0.05 + 0.01, 1])

            # Check width is the width of the receiving item
            assert new_rps.width == b.width

    def test_combine_unknown_filter(self):
        # Create the ingoing reception parameters
        reception_params_a = model.ReceptionParameters(nengo.Lowpass(0.05), 1)

        # Create the outgoing reception parameters
        reception_params_b = model.ReceptionParameters(mock.Mock(), 1)

        # Combine the parameter each way round
        for a, b in ((reception_params_a, reception_params_b),
                     (reception_params_a, reception_params_b)):
            with pytest.raises(NotImplementedError):
                new_rps = model_utils._combine_reception_params(a, b)
