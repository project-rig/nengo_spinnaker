import mock

from nengo_spinnaker.builder.builder import Model, ObjectPort, Signal
from nengo_spinnaker.operators import Filter
from nengo_spinnaker.utils.model import remove_childless_filters


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
    #     F1         O1 -S3-> F2     F5
    #      ^        /          \     ^
    #      \       /        S4  v   /  S6
    #       \-S2---              F4
    #
    # F1 should remain, 01 should be untouched and F2..5 should be removed.

    # Create the filter operators
    f1 = Filter(1)
    f2 = Filter(1)
    f3 = Filter(1)
    f4 = Filter(1)
    f5 = Filter(1)

    # The other operator
    o1 = mock.Mock()

    # Create some objects which map to some of the operators
    oo1 = mock.Mock()
    of3 = mock.Mock()

    # Create the signals
    s1 = Signal(ObjectPort(f1, None), ObjectPort(o1, None), None)
    s2 = Signal(ObjectPort(o1, None), ObjectPort(f1, None), None)
    s3 = Signal(ObjectPort(o1, None), ObjectPort(f2, None), None)
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
    assert model.extra_signals == [s1, s2]
