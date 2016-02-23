import collections
import mock
from rig.bitfield import BitField

from nengo_spinnaker.builder import model


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
        assert len(cm._connections) == 0
        cm.add_connection(source1, soport1, sp1, None, sink1, siport1, rp1)

        # Assert that this connection was added correctly
        assert len(cm._connections) == 1
        assert len(cm._connections[source1]) == 1
        assert (cm._connections[source1][soport1] ==
                [((sp1, None), [(sink1, siport1, rp1)])])

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
        assert (cm._connections[source1][soport1] ==
                [((sp1, tp1), [(sink1, siport1, rp1)]*2)])

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
        assert (cm._connections[source1][soport1] ==
                [((sp1, tp1), [(sink1, siport1, rp1)])])
        assert (cm._connections[source1][soport2] ==
                [((sp1, tp1), [(sink1, siport1, rp1)])])

    def test_add_default_keyspace(self):
        # Adding a default keyspace will modify all transmission parameters
        # which have their keyspace as None by adding a new keyspace instance
        # with the "object" and "connection" ID automatically set.
        #
        # Test this by creating 3 objects.  The first object will have 2
        # outgoing connection, the second will have an outgoing connection with
        # a pre-existing keyspace and the third will have 1 outgoing connection
        # only. Check that the object and connection IDs are set sequentially
        # only for connections without existing keyspaces.

        # Create 3 source objects
        sources = [mock.Mock(name="Source {}".format(i)) for i in range(3)]

        # Create transmission parameters
        sp00 = model.SignalParameters()
        sp01 = model.SignalParameters(latching=True)

        ks10 = mock.Mock(name="Keyspace")
        sp10 = model.SignalParameters(keyspace=ks10)

        sp20 = model.SignalParameters()
        port21 = mock.Mock(name="Port 2.1")
        sp21 = model.SignalParameters()

        # Manually create the connection map so that the order can be
        # guaranteed.
        cm = model.ConnectionMap()
        cm._connections = collections.OrderedDict()

        cm._connections[sources[0]] = collections.OrderedDict()
        cm._connections[sources[0]][None] = [
            model._ParsSinksPair((sp00, None)),
            model._ParsSinksPair((sp01, None))
        ]

        cm._connections[sources[1]] = collections.OrderedDict()
        cm._connections[sources[1]][None] = [
            model._ParsSinksPair((sp10, None))
        ]

        cm._connections[sources[2]] = collections.OrderedDict()
        cm._connections[sources[2]][None] = [
            model._ParsSinksPair((sp20, None))
        ]
        cm._connections[sources[2]][port21] = [
            model._ParsSinksPair((sp21, None))
        ]

        # Add the default keyspace
        ks = mock.Mock()
        kss = {
            0: mock.Mock(name="Keyspace00"),
            1: mock.Mock(name="Keyspace01"),
            2: mock.Mock(name="Keyspace20"),
            3: mock.Mock(name="Keyspace21"),
        }
        ks.side_effect = lambda connection_id: kss[connection_id]

        cm.add_default_keyspace(ks)

        # Ensure that the correct calls were made to "ks" in the correct order.
        ks.assert_has_calls([mock.call(connection_id=j) for j in range(4)])

        # Assert that the correct keyspaces were assigned to the correct
        # objects.
        assert sp00.keyspace is kss[0]
        assert sp01.keyspace is kss[1]
        assert sp10.keyspace is ks10
        assert sp20.keyspace is kss[2]
        assert sp21.keyspace is kss[3]

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

    def test_get_signals_to_object(self):
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
        sigs_a = cm.get_signals_to_object(sink_a)
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


def test_remove_sinkless_signals():
    """Removing sinkless signals will modify the connection map to remove any
    signals which no-longer have any sinks.
    """
    # Construct a connection map containing two signals.
    # Objects to act as sources and sinks
    obj_a = mock.Mock(name="A")
    obj_b = mock.Mock(name="B")

    # Add the connections
    cm = model.ConnectionMap()
    cm.add_connection(
        obj_a, None, model.SignalParameters(weight=1), None,
        obj_b, None, None
    )
    cm.add_connection(
        obj_b, None, model.SignalParameters(weight=2), None,
        obj_a, None, None
    )

    # Remove the sinks from one of the connections
    for _, sinks in cm._connections[obj_b][None]:
        for sink in sinks:
            sinks.remove(sink)

    # Remove all the sinkless signals
    model.remove_sinkless_signals(cm)

    # Assert that only one signal remains
    assert len(cm.get_signals_from_object(obj_a)[None]) == 1
    assert len(cm.get_signals_from_object(obj_b)) == 0
    assert len(list(cm.get_signals())) == 1

def test_remove_sinkless_objects():
    """Filter objects with no outgoing connections (hence sinks) can safely be
    removed from the network.  This process should be called repeatedly as long
    as there are filters which can be removed.

    We construct the following network and then expect that all items apart
    from A and G are removed from the connection map.

                /--> C --\
               /          \
        A --> B            E --> F
        |      \          /
        |       \--> D --/
        \-> G

    The result should be a reduced connection map and a list of all objects
    apart from A and G.
    """
    class G(object):
        """Separate class to indicate the type filtering works as expected."""

    # Construct the network objects
    a = mock.Mock(name="A")
    b = mock.Mock(name="B")
    c = mock.Mock(name="C")
    d = mock.Mock(name="D")
    e = mock.Mock(name="E")
    f = mock.Mock(name="F")
    g = G()

    # Make the connection map
    cm = model.ConnectionMap()
    cm.add_connection(a, None, model.SignalParameters(), None, b, None, None)
    cm.add_connection(a, None, model.SignalParameters(), None, g, None, None)
    cm.add_connection(b, None, model.SignalParameters(), None, c, None, None)
    cm.add_connection(b, None, model.SignalParameters(), None, d, None, None)
    cm.add_connection(c, None, model.SignalParameters(), None, e, None, None)
    cm.add_connection(d, None, model.SignalParameters(), None, e, None, None)
    cm.add_connection(e, None, model.SignalParameters(), None, f, None, None)
    cm._connections[f][None] = list()

    # Remove the sinkless filters
    removed = model.remove_sinkless_objects(cm, mock.Mock)

    # Check that the expected connection A->G still exists.
    assert len(cm.get_signals_from_object(a)[None]) == 1
    assert len(list(cm.get_signals())) == 1

    # Assert that the removed objects list is correct
    expected_removed = set((b, c, d, e, f))
    for x in expected_removed:
        assert len(cm.get_signals_from_object(x)) == 0
    assert removed == expected_removed
