"""Nengo/SpiNNaker specific configuration."""
import nengo
from nengo.params import BoolParam, DictParam, NumberParam, Parameter
from rig import place_and_route as par

from nengo_spinnaker.node_io import Ethernet
from nengo_spinnaker.simulator import Simulator


def _set_param(obj, name, ParamType, *args, **kwargs):
    # Create the parameter
    try:
        param = ParamType(*args, **kwargs)
    except TypeError:
        param = ParamType(name, *args, **kwargs)

    obj.set_param(name, param)


def add_spinnaker_params(config):
    """Add SpiNNaker specific parameters to a configuration object."""
    # Add simulator parameters
    config.configures(Simulator)

    _set_param(config[Simulator], "placer", CallableParameter,
               default=par.place)
    _set_param(config[Simulator], "placer_kwargs", DictParam,
               default=dict())

    _set_param(config[Simulator], "allocator", CallableParameter,
               default=par.allocate)
    _set_param(config[Simulator], "allocator_kwargs", DictParam,
               default=dict())

    _set_param(config[Simulator], "router", CallableParameter,
               default=par.route)
    _set_param(config[Simulator], "router_kwargs", DictParam,
               default=dict())

    _set_param(config[Simulator], "node_io", Parameter, default=Ethernet)
    _set_param(config[Simulator], "node_io_kwargs", DictParam, default={})

    # Add function_of_time parameters to Nodes
    _set_param(config[nengo.Node], "function_of_time", BoolParam,
               default=False)
    _set_param(config[nengo.Node], "function_of_time_period",
               NumberParam, default=None, optional=True)

    # Add optimisation control parameters to (passthrough) Nodes. None means
    # that a heuristic will be used to determine if the passthrough Node should
    # be removed.
    _set_param(config[nengo.Node], "optimize_out", BoolParam,
               default=None, optional=True)

    # Add profiling parameters to Ensembles
    _set_param(config[nengo.Ensemble], "profile", BoolParam, default=False)
    _set_param(config[nengo.Ensemble], "profile_num_samples",
               NumberParam, default=None, optional=True)


class CallableParameter(Parameter):
    """Parameter which only accepts callables."""
    def validate(self, instance, callable_obj):
        if not callable(callable_obj):
            raise ValueError(
                "must be callable, got type {}".format(type(callable_obj))
            )
