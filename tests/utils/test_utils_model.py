import mock

from nengo_spinnaker.builder.builder import Model, ObjectPort, Signal
from nengo_spinnaker.operators import Filter
from nengo_spinnaker.utils.model import (remove_childless_filters,
                                         remove_sinkless_signals)


def test_remove_sinkless_signals():
    """Signals with no sink should be removed."""
    # Create a netlist including some signals with no sinks, these signals
    # should be removed.
    o1 = mock.Mock(name="O1")
    o2 = mock.Mock(name="O2")

    # Create 4 signals (2 associated with connections, 2 not)
    cs1 = Signal(ObjectPort(o1, None), ObjectPort(o2, None), None)
    cs2 = Signal(ObjectPort(o1, None), [], None)
    ss1 = Signal(ObjectPort(o1, None), ObjectPort(o2, None), None)
    ss2 = Signal(ObjectPort(o1, None), [], None)

    # Create two mock connections
    c1 = mock.Mock(name="Connection 1")
    c2 = mock.Mock(name="Connection 2")

    # Create the model
    model = Model()
    model.extra_operators = [o1, o2]
    model.connections_signals = {c1: cs1, c2: cs2}
    model.extra_signals = [ss1, ss2]

    # Remove sinkless signals
    remove_sinkless_signals(model)

    # Check that signals were removed as necessary
    assert model.connections_signals == {c1: cs1}
    assert model.extra_signals == [ss1]


def test_remove_childless_filters():
    """Filter operators which don't transmit to anything, and their incoming
    signals, can be removed.
    """
    # Create a netlist including some filters that do and don't transmit to
    # other objects, check that all the filters which don't connect to anything
    # are removed.
    #
    #          -S1---             F3
    #        /       \       S4  ^  \  S5
    #       /        v          /    v
    #     F1         O1 +S3-> F2     F5
    #      ^        /   |      \     ^
    #      \       /    |   S4  v   /  S6
    #       \-S2---     v        F4
    #                  O2
    #
    # F1 should remain, O1 and O2 should be untouched and F2..5 should be
    # removed.  S1 and S2 should be unchanged, S3 should have F2 removed from
    # its sinks and S4..6 should be removed entirely.

    # Create the filter operators
    f1 = mock.Mock(name="F1", spec=Filter)
    f2 = mock.Mock(name="F2", spec=Filter)
    f3 = mock.Mock(name="F3", spec=Filter)
    f4 = mock.Mock(name="F4", spec=Filter)
    f5 = mock.Mock(name="F5", spec=Filter)

    # The other operator
    o1 = mock.Mock(name="O1")
    o2 = mock.Mock(name="O2")

    # Create some objects which map to some of the operators
    oo1 = mock.Mock()
    of3 = mock.Mock()

    # Create the signals
    s1 = Signal(ObjectPort(f1, None), ObjectPort(o1, None), None)
    s2 = Signal(ObjectPort(o1, None), ObjectPort(f1, None), None)
    s3 = Signal(ObjectPort(o1, None), [ObjectPort(f2, None),
                                       ObjectPort(o2, None)], None)
    s4 = Signal(ObjectPort(f2, None), [ObjectPort(f3, None),
                                       ObjectPort(f4, None)], None)
    s5 = Signal(ObjectPort(f3, None), ObjectPort(f5, None), None)
    s6 = Signal(ObjectPort(f4, None), ObjectPort(f5, None), None)

    # Create some connections which map to the signals
    cs4 = mock.Mock()
    cs5 = mock.Mock()

    # Create the model
    model = Model()
    model.object_operators = {
        oo1: o1,
        of3: f3,
    }
    model.extra_operators = [f1, f2, f4, f5]
    model.connections_signals = {
        cs4: s4,
        cs5: s5,
    }
    model.extra_signals = [s1, s2, s3, s6]

    # Perform the optimisation
    remove_childless_filters(model)

    # Check that objects have been removed
    assert model.object_operators == {oo1: o1}
    assert model.extra_operators == [f1]
    assert model.connections_signals == {}
    assert model.extra_signals == [s1, s2, s3]
    assert [s.obj for s in s3.sinks] == [o2]
