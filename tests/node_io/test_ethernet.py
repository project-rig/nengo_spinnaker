import nengo
import pytest

from nengo_spinnaker.builder import Model
from nengo_spinnaker.builder.model import OutputPort, InputPort
from nengo_spinnaker.node_io import ethernet as ethernet_io
from nengo_spinnaker.operators import SDPReceiver, SDPTransmitter


@pytest.mark.parametrize("transmission_period", [0.001, 0.002])
def test_Ethernet_init(transmission_period):
    """Test that the Ethernet initialisation creates a host network and stores
    appropriate rates.
    """
    # Create the EthernetIO
    io = ethernet_io.Ethernet(transmission_period=transmission_period)

    # Check that we stored the transmission period
    assert io.transmission_period == transmission_period

    # Check that there is a (empty) host network
    assert io.host_network.all_objects == list()
    assert io.host_network.all_connections == list()
    assert io.host_network.all_probes == list()

    # Check that the node input dictionary and lock are present
    with io.node_input_lock:
        assert io.node_input == dict()


def test_get_spinnaker_source_for_node():
    """Check that getting the SpiNNaker source for a Node returns an SDP Rx
    operator as the source object with OutputPort.standard as the port.  The
    spec should indicate that the connection should be latching.
    """
    with nengo.Network():
        a = nengo.Node(lambda t: t**2, size_out=1)
        b = nengo.Ensemble(100, 1)
        a_b = nengo.Connection(a, b)

    # Create an empty model and an Ethernet object
    model = Model()
    io = ethernet_io.Ethernet()
    spec = io.get_node_source(model, a_b)

    assert isinstance(spec.target.obj, SDPReceiver)
    assert spec.target.port is OutputPort.standard
    assert spec.latching
    assert model.extra_operators == [spec.target.obj]


def test_get_spinnaker_source_for_node_repeated():
    """Getting the source twice for the same Node should return the same
    object.
    """
    with nengo.Network():
        a = nengo.Node(lambda t: t**2, size_out=1)
        b = nengo.Ensemble(100, 1)
        a_b0 = nengo.Connection(a, b)
        a_b1 = nengo.Connection(a, b, transform=-0.5)

    # Create an empty model and an Ethernet object
    model = Model()
    io = ethernet_io.Ethernet()
    spec0 = io.get_node_source(model, a_b0)
    spec1 = io.get_node_source(model, a_b1)

    assert spec0.target.obj is spec1.target.obj
    assert model.extra_operators == [spec0.target.obj]


def test_get_spinnaker_sink_for_node():
    """Check that getting the SpiNNaker sink for a Node returns an SDP Tx
    operator as the sink object with InputPort.standard as the port.
    """
    with nengo.Network():
        a = nengo.Ensemble(100, 1)
        b = nengo.Node(lambda t, x: None, size_in=1)
        a_b = nengo.Connection(a, b)

    # Create an empty model and an Ethernet object
    model = Model()
    io = ethernet_io.Ethernet()
    spec = io.get_node_sink(model, a_b)

    assert isinstance(spec.target.obj, SDPTransmitter)
    assert spec.target.port is InputPort.standard
    assert model.extra_operators == [spec.target.obj]


def test_get_spinnaker_sink_for_node_repeated():
    """Check that getting the SpiNNaker sink for a Node twice returns the same
    target.
    """
    with nengo.Network():
        a = nengo.Ensemble(100, 1)
        b = nengo.Node(lambda t, x: None, size_in=1)
        a_b0 = nengo.Connection(a, b)
        a_b1 = nengo.Connection(a, b, synapse=0.3)

    # Create an empty model and an Ethernet object
    model = Model()
    io = ethernet_io.Ethernet()
    spec0 = io.get_node_sink(model, a_b0)
    spec1 = io.get_node_sink(model, a_b1)

    assert spec0.target.obj is spec1.target.obj
    assert model.extra_operators == [spec0.target.obj]
