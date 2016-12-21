import collections
import mock
import numpy as np
from rig.bitfield import BitField
from six import iteritems, itervalues
import pytest

import nengo
from nengo_spinnaker.builder import model
from nengo_spinnaker.builder.ports import InputPort, OutputPort
from nengo_spinnaker.builder.transmission_parameters import (
    PassthroughNodeTransmissionParameters, NodeTransmissionParameters,
    Transform
)


class TestSignalParameters(object):
    """SignalParameters should support __eq__.
    """
    def test_eq(self):
        # Create several SignalParameters and ensure that they only
        # report equal when they are actually equal.
        ks = BitField()
        ks.add_field("x")

        params = ((False, 5, ks(x=2)),
                  (True, 5, ks(x=2)),
                  (False, 4, ks(x=2)),
                  (False, 5, ks(x=3)),
                  )

        tps = tuple(model.SignalParameters(*args) for args in params)

        # None of these transmission parameters should test as equivalent.
        for a in tps:
            for b in tps:
                if a is not b:
                    assert a != b

        # Create a whole new set of transmission parameters using the same set
        # of parameters and ensure that they all test equivalent to their
        # counterparts in the original list.
        for a, b in zip(tps, tuple(model.SignalParameters(*args)
                                   for args in params)):
            assert a is not b
            assert a == b

    def test_combine_sig_pars_last_weight(self):
        """Check that the signal parameters are combined correctly."""
        # The last weight should be used
        sps1 = model.SignalParameters(weight=1)
        sps2 = model.SignalParameters(weight=5)

        combined_sps = sps1.concat(sps2)
        assert combined_sps.weight == sps2.weight

    @pytest.mark.parametrize(
        "latching_a, latching_b", [(True, True), (False, True), (True, False),
                                   (False, False)])
    def test_combine_sig_pars_latching(self, latching_a, latching_b):
        """Check that the signal parameters are combined correctly."""
        # The signal should be latching if any of its antecedents was latching.
        sps1 = model.SignalParameters(latching=latching_a)
        sps2 = model.SignalParameters(latching=latching_b)

        combined_sps = sps1.concat(sps2)
        assert combined_sps.latching is (latching_a or latching_b)

    def test_combine_sig_pars_keyspaces_all_none(self):
        """Check that the signal parameters are combined correctly."""
        # The keyspace should be None if all antecedents are None
        sps1 = model.SignalParameters()
        sps2 = model.SignalParameters()

        combined_sps = sps1.concat(sps2)
        assert combined_sps.keyspace is None

    @pytest.mark.parametrize("first", (False, True))
    def test_combine_sig_pars_keyspaces_one_specified(self, first):
        """Check that the signal parameters are combined correctly."""
        # If ONE keyspace is specified it should be used for the entire trace
        ks = object()
        sps1 = model.SignalParameters(keyspace=ks if first else None)
        sps2 = model.SignalParameters(keyspace=None if first else ks)

        combined_sps = sps1.concat(sps2)
        assert combined_sps.keyspace is ks

    def test_combine_sig_pars_keyspaces_two_specified(self):
        """Check that the signal parameters are combined correctly."""
        # An error should be raised if multiple keyspaces are found
        ks = object()
        sps1 = model.SignalParameters(keyspace=ks)
        sps2 = model.SignalParameters(keyspace=ks)

        with pytest.raises(Exception) as exc:
            sps1.concat(sps2)

        assert "keyspace" in str(exc.value)


class TestReceptionParameters(object):
    def test_combine_filters_none_none(self):
        # Create two reception parameters with None filters
        rp1 = model.ReceptionParameters(None, 1, None)
        rp2 = model.ReceptionParameters(None, 5, None)

        # Check that combined they have a None filter and that the last width
        # is used.
        rp_expected = model.ReceptionParameters(None, rp2.width, None)
        assert rp1.concat(rp2) == rp_expected

    @pytest.mark.parametrize("first", (True, False))
    def test_combine_filters_one_is_none(self, first):
        # Create two reception parameters, of which one has a None filter
        f = object()
        rp1 = model.ReceptionParameters(f if first else None, 1, None)
        rp2 = model.ReceptionParameters(None if first else f, 5, None)

        # Check that combined they have a None filter and that the last width
        # is used.
        rp_expected = model.ReceptionParameters(f, rp2.width, None)
        assert rp1.concat(rp2) == rp_expected

    def test_combine_lti_filters(self):
        f1 = nengo.synapses.LinearFilter([1, 2], [1, 2, 3])
        f2 = nengo.synapses.LinearFilter([2], [1, 2])

        rp1 = model.ReceptionParameters(f1, 1, None)
        rp2 = model.ReceptionParameters(f2, 1, None)

        # Combine and extract the filter
        f3 = rp1.concat(rp2).filter
        assert isinstance(f3, nengo.synapses.LinearFilter)
        assert np.array_equal(f3.num, [2, 4])
        assert np.array_equal(f3.den, [1, 4, 7, 6])

    def test_combine_unknown_filters(self):
        f1 = nengo.synapses.LinearFilter([1, 2], [1, 2, 3])
        rp1 = model.ReceptionParameters(f1, 1, None)
        rp2 = model.ReceptionParameters(object(), 1, None)

        # Combining filters should fail
        with pytest.raises(NotImplementedError):
            rp1.concat(rp2)

    @pytest.mark.parametrize("first", (True, False))
    def test_combine_learning_rules(self, first):
        # Create two reception parameters, of which one has a None filter
        lr = object()
        rp1 = model.ReceptionParameters(None, 1, lr if first else None)
        rp2 = model.ReceptionParameters(None, 1, None if first else lr)

        # Check that combined they have a None filter and that the last width
        # is used.
        rp_expected = model.ReceptionParameters(None, rp2.width, lr)
        assert rp1.concat(rp2) == rp_expected

    def test_combine_learning_rules_fails(self):
        # Cannot combine two learning rules
        rp1 = model.ReceptionParameters(None, 1, object())
        rp2 = model.ReceptionParameters(None, 1, object())

        with pytest.raises(NotImplementedError):
            rp1.concat(rp2)


class TestConnectionMap(object):
    def test_add_connection_basic(self):
        """Test adding connections to a connection map."""
        # Create parameters for the first connection
        source1 = mock.Mock(name="Source 1")
        soport1 = mock.Mock(name="Source Port 1")

        sp1 = model.SignalParameters(False)

        sink1 = mock.Mock(name="Sink 1")
        siport1 = mock.Mock(name="Sink Port 1")

        rp1 = mock.Mock(name="Reception Parameters")

        # Create the connection map and add the connection
        cm = model.ConnectionMap()
        assert len(list(cm.get_signals())) == 0
        cm.add_connection(source1, soport1, sp1, None, sink1, siport1, rp1)

        # Assert that this connection was added correctly
        assert len(list(cm.get_signals())) == 1
        assert len(cm._connections) == 1
        assert len(cm._connections[source1]) == 1
        assert (cm._connections[source1][soport1][(sp1, None)] ==
                [(sink1, siport1, rp1)])

    def test_add_connection_repeated(self):
        # Create parameters for the first connection
        source1 = mock.Mock(name="Source 1")
        soport1 = mock.Mock(name="Source Port 1")

        sp1 = model.SignalParameters(False)
        tp1 = mock.Mock(name="Transmission Parameters")

        sink1 = mock.Mock(name="Sink 1")
        siport1 = mock.Mock(name="Sink Port 1")

        rp1 = mock.Mock(name="Reception Parameters")

        # Create the connection map and add the connection
        cm = model.ConnectionMap()
        cm.add_connection(source1, soport1, sp1, tp1, sink1, siport1, rp1)

        # Add the "same" connection again
        cm.add_connection(source1, soport1, sp1, tp1, sink1, siport1, rp1)

        # Check that the connection was added twice
        assert (cm._connections[source1][soport1][(sp1, tp1)] ==
                [(sink1, siport1, rp1)]*2)

    def test_add_connection_different_port(self):
        # Create parameters for the first connection
        source1 = mock.Mock(name="Source 1")
        soport1 = mock.Mock(name="Source Port 1")

        sp1 = model.SignalParameters(False)
        tp1 = mock.Mock(name="Transmission Parameters")

        sink1 = mock.Mock(name="Sink 1")
        siport1 = mock.Mock(name="Sink Port 1")

        rp1 = mock.Mock(name="Reception Parameters")

        # Create the connection map and add the connection
        cm = model.ConnectionMap()
        cm.add_connection(source1, soport1, sp1, tp1, sink1, siport1, rp1)

        # Create and add the second connection with a different port
        soport2 = mock.Mock(name="Source Port 2")
        cm.add_connection(source1, soport2, sp1, tp1, sink1, siport1, rp1)

        # Assert the map is still correct
        assert (cm._connections[source1][soport1][(sp1, tp1)] ==
                [(sink1, siport1, rp1)])
        assert (cm._connections[source1][soport2][(sp1, tp1)] ==
                [(sink1, siport1, rp1)])

    def test_get_signals_from_object(self):
        # Create two ports
        sp_1 = mock.Mock(name="Source Port 1")
        sp_2 = mock.Mock(name="Source Port 2")

        # Create some connections from two objects
        source_a = mock.Mock(name="Source A")
        source_b = mock.Mock(name="Source B")

        sp1 = model.SignalParameters()
        sp2 = model.SignalParameters(True)
        sp3 = model.SignalParameters(weight=1)

        tp1 = mock.Mock(name="Transmission Parameters 1")
        tp2 = mock.Mock(name="Transmission Parameters 2")
        tp3 = mock.Mock(name="Transmission Parameters 3")

        # Add the connections
        cm = model.ConnectionMap()

        conns_a = ((sp_1, sp1, tp1), (sp_1, sp2, tp2), (sp_2, sp3, tp3))
        for port, sp, tp in conns_a:
            cm.add_connection(source_a, port, sp, tp, None, None, None)

        conns_b = ((sp_2, sp1, tp2), (sp_2, sp2, tp2), (sp_1, sp3, tp3))
        for port, sp, tp in conns_b:
            cm.add_connection(source_b, port, sp, tp, None, None, None)

        # Get the signals from source_a, check that they are as expected
        sigs_a = cm.get_signals_from_object(source_a)
        assert (sp1, tp1) in sigs_a[sp_1]
        assert (sp2, tp2) in sigs_a[sp_1]
        assert sigs_a[sp_2] == [(sp3, tp3)]

        # Get the signals from source_b, check that they are as expected
        sigs_b = cm.get_signals_from_object(source_b)
        assert sigs_b[sp_1] == [(sp3, tp3)]
        assert (sp1, tp2) in sigs_b[sp_2]
        assert (sp2, tp2) in sigs_b[sp_2]

    def test_get_signals_to_all_objects(self):
        # Create two ports
        sp_1 = mock.Mock(name="Sink Port 1")
        sp_2 = mock.Mock(name="Sink Port 2")

        # Create some connections to an object
        sink_a = mock.Mock(name="Sink A")

        tp1 = model.SignalParameters()
        tp2 = model.SignalParameters(True)
        tp3 = model.SignalParameters(weight=1)

        rp1 = mock.Mock(name="Reception Parameters 1")
        rp2 = mock.Mock(name="Reception Parameters 2")

        # Add the connections
        cm = model.ConnectionMap()

        conns_a = (
            (tp1, sink_a, sp_1, rp1),
            (tp1, sink_a, sp_1, rp2),
            (tp2, sink_a, sp_2, rp1),
            (tp3, sink_a, sp_2, rp1),
        )
        for tp, sink, sink_port, rp in conns_a:
            cm.add_connection(None, None, tp, None, sink, sink_port, rp)

        # Add another connection to another object
        sink_b = mock.Mock(name="Sink B")
        cm.add_connection(None, None, tp1, None, sink_b, sp_1, rp)

        # Get the signals to sink_a, check that they are as expected
        sigs_a = cm.get_signals_to_all_objects()[sink_a]
        assert len(sigs_a[sp_1]) == 2
        seen_rps = []

        for spec in sigs_a[sp_1]:
            assert spec.signal_parameters is tp1
            seen_rps.append(spec.reception_parameters)

        assert rp1 in seen_rps and rp2 in seen_rps

        assert len(sigs_a[sp_2]) == 2
        seen_tps = []

        for spec in sigs_a[sp_2]:
            assert spec.reception_parameters is rp1
            seen_tps.append(spec.signal_parameters)

        assert tp2 in seen_tps and tp3 in seen_tps

    def test_get_signals(self):
        # Construct some connections and ensure that these result in
        # appropriate signals being returned.
        # Objects to act as sources and sinks
        obj_a = mock.Mock(name="A")
        obj_b = mock.Mock(name="B")
        obj_c = mock.Mock(name="C")

        # Keyspaces
        ks_abc = mock.Mock(name="Keyspace A -> B,C")
        ks_cb = mock.Mock(name="Keyspace C -> B")

        # Add the connections
        cm = model.ConnectionMap()
        tp_a = mock.Mock(name="Transmission Parameters A")
        tp_b = mock.Mock(name="Transmission Parameters B")

        cm.add_connection(
            obj_a, None, model.SignalParameters(weight=3, keyspace=ks_abc),
            tp_a, obj_b, None, None
        )
        cm.add_connection(
            obj_a, None, model.SignalParameters(weight=3, keyspace=ks_abc),
            tp_a, obj_c, None, None
        )
        cm.add_connection(
            obj_c, None, model.SignalParameters(weight=5, keyspace=ks_cb),
            tp_b, obj_b, None, None
        )

        # Get the signals, this should be a list of two signals
        signals = list(cm.get_signals())
        assert len(signals) == 2
        for signal, transmission_params in signals:
            if signal.source is obj_a:
                # Assert the sinks are correct
                assert len(signal.sinks) == 2
                assert obj_b in signal.sinks
                assert obj_c in signal.sinks

                # Assert the keyspace is correct
                assert signal.keyspace is ks_abc

                # Assert the weight is correct
                assert signal.weight == 3

                # Assert the correct paired transmission parameters are used.
                assert transmission_params is tp_a
            else:
                # Source should be C, sink B
                assert signal.source is obj_c
                assert signal.sinks == [obj_b]
                assert signal.keyspace is ks_cb
                assert signal.weight == 5

                # Assert the correct paired transmission parameters are used.
                assert transmission_params is tp_b

    def test_get_coarsened_graph_and_extract_cliques(self):
        """Check that a coarsened representation of the graph can be extracted.
        """
        # Construct a graph of the form:
        #
        #         /--<-------------<--\
        #         |                   |
        # E ->-\  v /->- E ->-\       |
        # E ->--> o -->- E ->--> o ->-/
        # E ->-/    \->- E ->-/
        #
        # Where E are ensembles and `o' is a passthrough node.
        cm = model.ConnectionMap()

        # Network objects
        ens_a = [object() for _ in range(3)]
        ens_b = [object() for _ in range(3)]
        ptns = [model.PassthroughNode(), model.PassthroughNode()]

        # Add connections to the network, excluding the feedback loop.
        for ens in ens_a:
            cm.add_connection(
                ens, OutputPort.standard, model.SignalParameters(),
                object(), ptns[0], InputPort.standard, None
            )

        for ens in ens_b:
            cm.add_connection(
                ptns[0], OutputPort.standard, model.SignalParameters(),
                object(), ens, InputPort.standard, None
            )
            cm.add_connection(
                ens, OutputPort.standard, model.SignalParameters(),
                object(), ptns[1], InputPort.standard, None
            )

        # Get a coarsened representation of the connectivity
        graph = cm.get_coarsened_graph()

        for ens in ens_a:
            assert graph[ens].inputs == set()
            assert graph[ens].outputs == {ptns[0]}

        for ens in ens_b:
            assert graph[ens].inputs == {ptns[0]}
            assert graph[ens].outputs == {ptns[1]}

        assert graph[ptns[0]].inputs == set(ens_a)
        assert graph[ptns[0]].outputs == set(ens_b)

        assert graph[ptns[1]].inputs == set(ens_b)
        assert graph[ptns[1]].outputs == set()

        # Extract cliques from this graph, without the feedback in place there
        # should be two cliques.
        for sources, nodes in cm.get_cliques():
            if nodes == {ptns[0]}:
                assert sources == set(ens_a)
            elif nodes == {ptns[1]}:
                assert sources == set(ens_b)
            else:
                assert False, "Unexpected clique."

        # Add a feedback connection to the graph, this should mean that only a
        # single clique exists.
        cm.add_connection(
            ptns[1], OutputPort.standard, model.SignalParameters(),
            object(), ptns[0], InputPort.standard, None
        )

        # Get a coarsened representation of the connectivity
        graph = cm.get_coarsened_graph()

        for ens in ens_a:
            assert graph[ens].inputs == set()
            assert graph[ens].outputs == {ptns[0]}

        for ens in ens_b:
            assert graph[ens].inputs == {ptns[0]}
            assert graph[ens].outputs == {ptns[1]}

        assert graph[ptns[0]].inputs == set(ens_a) | {ptns[1]}
        assert graph[ptns[0]].outputs == set(ens_b)

        assert graph[ptns[1]].inputs == set(ens_b)
        assert graph[ptns[1]].outputs == {ptns[0]}

        # Extract cliques from this graph, without the feedback in place there
        # should be two cliques.
        for sources, nodes in cm.get_cliques():
            assert nodes == set(ptns)
            assert sources == set(ens_a) | set(ens_b)

    def test_insert_interposers_removes_passthrough_node(self):
        """Test that passthrough nodes are removed while inserting interposers.
        """
        cm = model.ConnectionMap()

        # Add a connection from a node to a passthrough node to the model
        node = object()
        ptn = model.PassthroughNode()
        cm.add_connection(
            node, OutputPort.standard, model.SignalParameters(weight=1),
            NodeTransmissionParameters(Transform(1, 1, 1)),
            ptn, InputPort.standard, model.ReceptionParameters(None, 1, None)
        )

        # Add a connection from the passthrough node to another node
        sink = object()
        cm.add_connection(
            ptn, OutputPort.standard, model.SignalParameters(weight=1),
            PassthroughNodeTransmissionParameters(Transform(1, 1, 1)),
            sink, InputPort.standard, model.ReceptionParameters(None, 1, None)
        )

        # Insert interposers, getting a list of interposers (empty) and a new
        # connection map.
        interposers, new_cm = cm.insert_interposers()
        assert len(interposers) == 0  # No interposers expected

        # Check that there is now just one connection from the node to the sink
        from_node = new_cm._connections[node]
        assert list(from_node) == [OutputPort.standard]
        
        for (signal_pars, transmission_pars), sinks in \
                iteritems(from_node[OutputPort.standard]):
            # Check the transmission parameters
            assert transmission_pars == NodeTransmissionParameters(
                Transform(1, 1, 1)
            )

            # Check that the sink is correct
            assert len(sinks) == 1
            for s in sinks:
                assert s.sink_object is sink

    def test_insert_interposers_simple(self):
        """Test that interposers are inserted correctly."""
        cm = model.ConnectionMap()

        # Add a connection from a node to a passthrough node to the model
        nodes = [object(), object()]
        ptn = model.PassthroughNode()

        for node in nodes:
            cm.add_connection(
                node, OutputPort.standard, model.SignalParameters(weight=1),
                NodeTransmissionParameters(Transform(1, 1, 1)),
                ptn, InputPort.standard,
                model.ReceptionParameters(None, 1, None)
            )

        # Add a connection from the passthrough node to another node
        sink = object()
        sink_port = object()
        cm.add_connection(
            ptn, OutputPort.standard, model.SignalParameters(weight=70),
            PassthroughNodeTransmissionParameters(
                Transform(1, 70, np.ones((70, 1)))
            ),
            sink, sink_port, model.ReceptionParameters(None, 70, None)
        )

        # Insert interposers, getting a list of interposers and a new
        # connection map.
        interposers, new_cm = cm.insert_interposers()
        assert len(interposers) == 1  # Should insert 1 interposer
        interposer = interposers[0]

        # Check that each of the nodes connects to the interposer
        for node in nodes:
            from_node = new_cm._connections[node][OutputPort.standard]
            assert len(from_node) == 1

            for sinks in itervalues(from_node):
                assert len(sinks) == 1
                for s in sinks:
                    assert s.sink_object is interposer
                    assert s.port is InputPort.standard

        # Check that the interposer connects to the sink
        from_interposer = new_cm._connections[interposer][OutputPort.standard]
        assert len(from_interposer) == 1

        for sinks in itervalues(from_interposer):
            assert len(sinks) == 1
            for s in sinks:
                assert s.sink_object is sink
                assert s.port is sink_port

    def test_insert_interposers_ignore_sinkless(self):
        """Test that interposers are inserted correctly, ignoring those that
        don't connect to anything.
        """
        cm = model.ConnectionMap()

        # Add a connection from a node to a passthrough node to the model
        nodes = [object(), object()]
        ptn = model.PassthroughNode()

        for node in nodes:
            cm.add_connection(
                node, OutputPort.standard, model.SignalParameters(weight=1),
                NodeTransmissionParameters(Transform(1, 1, 1)),
                ptn, InputPort.standard,
                model.ReceptionParameters(None, 1, None)
            )

        # Connect the passthrough node to another passthrough node
        ptn2 = model.PassthroughNode()
        cm.add_connection(ptn, OutputPort.standard, model.SignalParameters(),
                          PassthroughNodeTransmissionParameters(
                            Transform(1, 70, transform=np.ones((70, 1)))),
                          ptn2, InputPort.standard,
                          model.ReceptionParameters(None, 70, None))

        # Connect the passthrough node to another passthrough node
        ptn3 = model.PassthroughNode()
        cm.add_connection(ptn2, OutputPort.standard, model.SignalParameters(),
                          PassthroughNodeTransmissionParameters(
                            Transform(70, 70, 1)),
                          ptn3, InputPort.standard,
                          model.ReceptionParameters(None, 70, None))

        # Insert interposers, getting a list of interposers and a new
        # connection map.
        interposers, new_cm = cm.insert_interposers()
        assert len(interposers) == 0  # No interposers

        # No signals at all
        assert len(list(new_cm.get_signals())) == 0

    def test_insert_interposers_earliest_interposer_only(self):
        """Test that only the first interposer in a network of possible
        interposers is inserted.
        """
        cm = model.ConnectionMap()

        node = object()
        ptn1 = model.PassthroughNode()
        ptn2 = model.PassthroughNode()
        sink = object()

        # Add connections
        cm.add_connection(
            node, OutputPort.standard, model.SignalParameters(),
            NodeTransmissionParameters(Transform(16, 512, 1,
                                                 slice_out=slice(16, 32))),
            ptn1, InputPort.standard,
            model.ReceptionParameters(None, 512, None)
        )
        cm.add_connection(
            ptn1, OutputPort.standard, model.SignalParameters(),
            PassthroughNodeTransmissionParameters(
                Transform(512, 512, np.ones((512, 512)))
            ),
            ptn2, InputPort.standard,
            model.ReceptionParameters(None, 512, None)
        )
        cm.add_connection(
            ptn2, OutputPort.standard, model.SignalParameters(),
            PassthroughNodeTransmissionParameters(
                Transform(512, 1024, np.ones((1024, 512)))
            ),
            sink, InputPort.standard,
            model.ReceptionParameters(None, 1024, None)
        )

        # Insert interposers, only one should be included
        interposers, new_cm = cm.insert_interposers()
        assert len(interposers) == 1
        interposer = interposers[0]

        # Check that the node connects to the interposer and that the
        # interposer was the first passthrough node.
        from_node = new_cm._connections[node][OutputPort.standard]
        for (_, transmission_pars), sinks in iteritems(from_node):
            assert transmission_pars == NodeTransmissionParameters(
                Transform(16, 512, 1, slice_out=slice(16, 32))
            )

            assert len(sinks) == 1
            for s in sinks:
                assert s.sink_object is interposer

        # Check that the interposer connects to the sink
        from_interposer = new_cm._connections[interposer][OutputPort.standard]
        for (_, transmission_pars), sinks in iteritems(from_interposer):
            assert transmission_pars.size_in == 512
            assert transmission_pars.size_out == 1024

            assert len(sinks) == 1
            for s in sinks:
                assert s.sink_object is sink
