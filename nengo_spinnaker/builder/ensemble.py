import collections
import enum
import nengo
from nengo.builder import ensemble
from nengo.dists import Distribution
from nengo.utils import numpy as npext
import numpy as np

from .builder import Model
from ..utils import collections as collections_ext

BuiltEnsemble = collections.namedtuple(
    "BuiltEnsemble", "eval_points, encoders, intercepts, max_rates, "
                     "scaled_encoders, gain, bias"
)
"""Parameters which describe an Ensemble."""


class EnsembleOutputPort(enum.Enum):
    """Ensemble only output ports."""
    neurons = 0
    """Spike-based neuron output."""


class EnsembleInputPort(enum.Enum):
    """Ensemble only input ports."""
    neurons = 0
    """Spike-based neuron input."""

    global_inhibition = 1
    """Global inhibition input."""


ensemble_builders = collections_ext.registerabledict()
"""Dictionary mapping neuron types to appropriate build methods."""


@Model.builders.register(nengo.Ensemble)
def build_ensemble(model, ens):
    ensemble_builders[type(ens.neuron_type)](model, ens)


@ensemble_builders.register(nengo.neurons.LIF)
def build_lif(model, ens):
    # Create a random number generator
    rng = np.random.RandomState(model.seeds[ens])

    # Get the eval points
    eval_points = ensemble.gen_eval_points(ens, ens.eval_points, rng=rng)

    # Get the encoders
    if isinstance(ens.encoders, Distribution):
        encoders = ens.encoders.sample(ens.n_neurons, ens.dimensions, rng=rng)
        encoders = np.asarray(encoders, dtype=np.float64)
    else:
        encoders = npext.array(ens.encoders, min_dims=2, dtype=np.float64)
    encoders /= npext.norm(encoders, axis=1, keepdims=True)

    # Get maximum rates and intercepts
    max_rates = ensemble.sample(ens.max_rates, ens.n_neurons, rng=rng)
    intercepts = ensemble.sample(ens.intercepts, ens.n_neurons, rng=rng)

    # Build the neurons
    if ens.gain is None and ens.bias is None:
        gain, bias = ens.neuron_type.gain_bias(max_rates, intercepts)
    elif ens.gain is not None and ens.bias is not None:
        gain = ensemble.sample(ens.gain, ens.n_neurons, rng=rng)
        bias = ensemble.sample(ens.bias, ens.n_neurons, rng=rng)
    else:
        raise NotImplementedError(
            "gain or bias set for {!s}, but not both. Solving for one given "
            "the other is not yet implemented.".format(ens)
        )

    # Scale the encoders
    scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

    # Store all the parameters
    model.params[ens] = BuiltEnsemble(
        eval_points=eval_points,
        encoders=encoders,
        scaled_encoders=scaled_encoders,
        max_rates=max_rates,
        intercepts=intercepts,
        gain=gain,
        bias=bias
    )

    # Create the object which will handle simulation of the LIF ensemble.  This
    # object will be responsible for adding items to the netlist and providing
    # functions to prepare the ensemble for simulation.  The object may be
    # modified by later methods.
    model.object_intermediates[ens] = EnsembleLIF(ens.size_in)


class EnsembleLIF(object):
    """Controller for an ensemble of LIF neurons."""
    def __init__(self, size_in):
        """Create a new LIF ensemble controller."""
        self.direct_input = np.zeros(size_in)
        self.local_probes = list()
