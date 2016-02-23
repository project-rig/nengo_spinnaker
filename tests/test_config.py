import nengo
import pytest
from rig import place_and_route as par

from nengo_spinnaker import Simulator, add_spinnaker_params
from nengo_spinnaker.config import CallableParameter
from nengo_spinnaker import node_io


def test_add_spinnaker_params():
    """Test adding SpiNNaker specific parameters to a configuration object."""
    # Create a network
    with nengo.Network() as net:
        n_ft = nengo.Node(lambda t: [t, t**2])
        ptn = nengo.Node(size_in=2)

    # Setting SpiNNaker-specific options before calling `add_spinnaker_params`
    # should fail.

    for param, value in [
            ("function_of_time", True),
            ("function_of_time_period", 0.5),
            ]:
        with pytest.raises(AttributeError) as excinfo:
            setattr(net.config[n_ft], param, value)
        assert param in str(excinfo.value)

    for param, value in [
            ("n_cores_per_chip", 16),
            ("n_chips", 4),
            ("optimize_out", False),
            ]:
        with pytest.raises(AttributeError) as excinfo:
            setattr(net.config[ptn], param, value)

    for param, value in [
            ("placer", lambda r, n, m, c: None),
            ("placer_kwargs", {}),
            ("allocater", lambda r, n, m, c, p: None),
            ("allocater_kwargs", {}),
            ("router", lambda r, n, m, c, p, a: None),
            ("router_kwargs", {}),
            ("node_io", None),
            ("node_io_kwargs", {}),
            ]:
        with pytest.raises(KeyError) as excinfo:
            setattr(net.config[Simulator], param, value)
        assert "Simulator" in str(excinfo.value)

    # Adding the SpiNNaker parameters should allow all of these to pass
    add_spinnaker_params(net.config)

    assert net.config[nengo.Node].function_of_time is False
    assert net.config[nengo.Node].function_of_time_period is None
    assert net.config[nengo.Node].optimize_out is None

    assert net.config[nengo.Node].n_cores_per_chip is None
    assert net.config[nengo.Node].n_chips is None

    assert net.config[Simulator].placer is par.place
    assert net.config[Simulator].placer_kwargs == {}
    assert net.config[Simulator].allocater is par.allocate
    assert net.config[Simulator].allocater_kwargs == {}
    assert net.config[Simulator].router is par.route
    assert net.config[Simulator].router_kwargs == {}

    assert net.config[Simulator].node_io is node_io.Ethernet
    assert net.config[Simulator].node_io_kwargs == {}


def test_callable_parameter_validate():
    """Test that the callable parameter fails to validate if passed something
    other than a callable.
    """
    cp = CallableParameter()

    with pytest.raises(ValueError) as excinfo:
        cp.validate(None, "Not a function")
    assert "must be callable" in str(excinfo.value)

    cp.validate(None, lambda x: None)


@pytest.mark.xfail(reason="Problems with Parameters")
def test_function_of_time_node():
    # Test that function of time can't be marked on Nodes unless they have size
    # in == 0
    with nengo.Network() as net:
        not_f_of_t = nengo.Node(lambda t, x: t**2, size_in=1)
        f_of_t = nengo.Node(lambda t: t)

    # Modify the config
    add_spinnaker_params(net.config)
    net.config[f_of_t].function_of_time = True

    with pytest.raises(ValueError):
        net.config[not_f_of_t].function_of_time = True

    # Check the settings are valid
    assert not net.config[not_f_of_t].function_of_time
    assert net.config[f_of_t].function_of_time
