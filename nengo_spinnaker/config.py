"""Nengo/SpiNNaker specific configuration."""
import nengo
from nengo.params import BoolParam, DictParam, NumberParam, Parameter, IntParam
from rig import place_and_route as par

from nengo_spinnaker.node_io import Ethernet
from nengo_spinnaker.simulator import Simulator


def add_spinnaker_params(config):
    """Add SpiNNaker specific parameters to a configuration object."""
    # Add simulator parameters
    config.configures(Simulator)

    config[Simulator].set_param("placer", CallableParameter(default=par.place))
    config[Simulator].set_param("placer_kwargs", DictParam(default={}))

    config[Simulator].set_param("allocater",
                                CallableParameter(default=par.allocate))
    config[Simulator].set_param("allocater_kwargs",
                                DictParam(default={}))

    config[Simulator].set_param("router", CallableParameter(default=par.route))
    config[Simulator].set_param("router_kwargs", DictParam(default={}))

    config[Simulator].set_param("node_io", Parameter(default=Ethernet))
    config[Simulator].set_param("node_io_kwargs", DictParam(default={}))

    # Add function_of_time parameters to Nodes
    config[nengo.Node].set_param("function_of_time", BoolParam(default=False))
    config[nengo.Node].set_param("function_of_time_period",
                                 NumberParam(default=None, optional=True))

    # Add multiple-core options to Nodes
    config[nengo.Node].set_param(
        "n_cores_per_chip",
        IntParam(default=None, low=1, high=16, optional=True)
    )
    config[nengo.Node].set_param(
        "n_chips",
        IntParam(default=None, low=1, optional=True)
    )
    # Add optimisation control parameters to (passthrough) Nodes. None means
    # that a heuristic will be used to determine if the passthrough Node should
    # be removed.
    config[nengo.Node].set_param("optimize_out",
                                 BoolParam(default=None, optional=True))

    # Add profiling parameters to Ensembles
    config[nengo.Ensemble].set_param("profile", BoolParam(default=False))
    config[nengo.Ensemble].set_param("profile_num_samples",
                                     NumberParam(default=None, optional=True))


class CallableParameter(Parameter):
    """Parameter which only accepts callables."""
    def validate(self, instance, callable_obj):
        if not callable(callable_obj):
            raise ValueError(
                "must be callable, got type {}".format(type(callable_obj))
            )
