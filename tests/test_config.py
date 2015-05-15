import mock
import nengo
import pytest
from rig import place_and_route as par

from nengo_spinnaker import SpiNNakerSimulator, add_spinnaker_params
from nengo_spinnaker.config import CallableParameter


def test_add_spinnaker_params():
    """Test adding SpiNNaker specific parameters to a configuration object."""
    # Create a network
    with nengo.Network() as net:
        n_ft = nengo.Node(lambda t: [t, t**2])

    # Setting SpiNNaker-specific options before calling `add_spinnaker_params`
    # should fail.

    for param, value in [
            ("function_of_time", True),
            ("function_of_time_period", 0.5),
            ]:
        with pytest.raises(AttributeError) as excinfo:
            setattr(net.config[n_ft], param, value)
        assert ("Unknown config parameter '{}'".format(param) in
                str(excinfo.value))

    for param, value in [
            ("placer", lambda r, n, m, c: None),
            ("placer_kwargs", {}),
            ("allocater", lambda r, n, m, c, p: None),
            ("allocater_kwargs", {}),
            ("router", lambda r, n, m, c, p, a: None),
            ("router_kwargs", {}),
            ]:
        with pytest.raises(KeyError) as excinfo:
            setattr(net.config[SpiNNakerSimulator], param, value)
        assert "SpiNNakerSimulator" in str(excinfo.value)

    # Adding the SpiNNaker parameters should allow all of these to pass
    add_spinnaker_params(net.config)

    assert net.config[nengo.Node].function_of_time is False
    assert net.config[nengo.Node].function_of_time_period is None

    assert net.config[SpiNNakerSimulator].placer is par.place
    assert net.config[SpiNNakerSimulator].placer_kwargs == {}
    assert net.config[SpiNNakerSimulator].allocater is par.allocate
    assert net.config[SpiNNakerSimulator].allocater_kwargs == {}
    assert net.config[SpiNNakerSimulator].router is par.route
    assert net.config[SpiNNakerSimulator].router_kwargs == {}


def test_callable_parameter_validate():
    """Test that the callable parameter fails to validate if passed something
    other than a callable.
    """
    cp = CallableParameter()

    with pytest.raises(ValueError) as excinfo:
        cp.validate(None, "Not a function")
    assert "must be callable" in str(excinfo.value)

    cp.validate(None, lambda x: None)
