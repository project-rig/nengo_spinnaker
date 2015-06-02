import mock
import nengo

from nengo_spinnaker import Simulator, add_spinnaker_params
from nengo_spinnaker.utils.config import getconfig


def test_getconfig():
    # Create a network, DON'T add nengo_spinnaker configuration
    with nengo.Network() as net:
        n = nengo.Node(lambda t: [t, t+2])

    # Use getconfig to get configuration options that don't exist get
    placer = mock.Mock()
    assert getconfig(net.config, Simulator, "placer", placer) is placer
    assert not getconfig(net.config, n, "function_of_time", False)

    # Now add the config
    add_spinnaker_params(net.config)
    net.config[n].function_of_time = True

    # Use getconfig to get configuration options from the config
    assert getconfig(net.config, Simulator, "placer", placer) is not placer
    assert getconfig(net.config, n, "function_of_time", False)
