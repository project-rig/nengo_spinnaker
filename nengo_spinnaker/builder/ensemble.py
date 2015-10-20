import collections
import nengo
from nengo.builder import connection as connection_b
from nengo.builder import ensemble
from nengo.dists import Distribution
from nengo.processes import Process
from nengo.utils.builder import full_transform
from nengo.utils import numpy as npext
import numpy as np

from .builder import BuiltConnection, Model, ObjectPort, spec
from .connection import EnsembleTransmissionParameters
from .model import InputPort
from .ports import EnsembleInputPort
from .. import operators
from ..utils import collections as collections_ext

BuiltEnsemble = collections.namedtuple(
    "BuiltEnsemble", "eval_points, encoders, intercepts, max_rates, "
                     "scaled_encoders, gain, bias"
)
"""Parameters which describe an Ensemble."""


@Model.source_getters.register(nengo.ensemble.Neurons)
def get_neurons_source(model, connection):
    """Get the source for connections out of neurons."""
    raise NotImplementedError(
        "SpiNNaker does not currently support neuron to neuron connections"
    )


@Model.sink_getters.register(nengo.Ensemble)
def get_ensemble_sink(model, connection):
    """Get the sink for connections into an Ensemble."""
    ens = model.object_operators[connection.post_obj]

    if (isinstance(connection.pre_obj, nengo.Node) and
            not callable(connection.pre_obj.output) and
            not isinstance(connection.pre_obj.output, Process) and
            connection.pre_obj.output is not None):
        # Connections from constant valued Nodes are optimised out.
        # Build the value that will be added to the direct input for the
        # ensemble.
        val = connection.pre_obj.output[connection.pre_slice]

        if connection.function is not None:
            val = connection.function(val)

        transform = full_transform(connection, slice_pre=False)
        ens.direct_input += np.dot(transform, val)
    else:
        # Otherwise we just sink into the Ensemble
        return spec(ObjectPort(ens, InputPort.standard))


@Model.sink_getters.register(nengo.ensemble.Neurons)
def get_neurons_sink(model, connection):
    """Get the sink for connections into the neurons of an ensemble."""
    ens = model.object_operators[connection.post_obj.ensemble]

    if (connection.transform.ndim == 2 and
            np.all(connection.transform[1:] == connection.transform[0])):
        # Connections from non-neurons to Neurons where the transform delivers
        # the same value to all neurons are treated as global inhibition
        # connection.
        # Return a signal to the correct port.
        return spec(ObjectPort(ens, EnsembleInputPort.global_inhibition))
    else:
        #  - Connections from Neurons can go straight to the Neurons.
        #  - Otherwise we don't support arbitrary connections into neurons, but
        #    we allow them because they may be optimised out later when we come
        #    to remove passthrough nodes.
        return spec(ObjectPort(ens, EnsembleInputPort.neurons))


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
    model.object_operators[ens] = operators.EnsembleLIF(ens)


@Model.transmission_parameter_builders.register(nengo.Ensemble)
def build_from_ensemble_connection(model, conn):
    """Build the parameters object for a connection from an Ensemble."""
    if conn.solver.weights:
        raise NotImplementedError(
            "SpiNNaker does not currently support neuron to neuron connections"
        )

    # Create a random number generator
    rng = np.random.RandomState(model.seeds[conn])

    # Get the transform
    transform = full_transform(conn, slice_pre=False, allow_scalars=False)

    # Solve for the decoders
    eval_points, decoders, solver_info = connection_b.build_decoders(
        model, conn, rng
    )

    # Store the parameters in the model
    model.params[conn] = BuiltConnection(decoders=decoders,
                                         eval_points=eval_points,
                                         transform=transform,
                                         solver_info=solver_info)

    # Modify the transform if this is a global inhibition connection
    if (isinstance(conn.post_obj, nengo.ensemble.Neurons) and
            np.all(transform[0, :] == transform[1:, :])):
        transform = np.array([transform[0]])

    return EnsembleTransmissionParameters(decoders, transform)


@Model.transmission_parameter_builders.register(nengo.ensemble.Neurons)
def build_from_neurons_connection(model, conn):
    """Build the parameters object for a connection from Neurons."""
    raise NotImplementedError(
        "SpiNNaker does not currently support connections from Neurons"""
    )


@Model.probe_builders.register(nengo.Ensemble)
def build_ensemble_probe(model, probe):
    """Build a Probe which has an Ensemble as its target."""
    if probe.attr == "decoded_output":
        # Create an object to receive the probed data
        model.object_operators[probe] = operators.ValueSink(probe, model.dt)

        # Create a new connection from the ensemble to the probe
        seed = model.seeds[probe]
        conn = nengo.Connection(
            probe.target, probe, synapse=probe.synapse, solver=probe.solver,
            seed=seed, add_to_container=False
        )
        model.make_connection(conn)
    else:
        raise NotImplementedError(
            "SpiNNaker does not support probing '{}' on Ensembles.".format(
                probe.attr
            )
        )


@Model.probe_builders.register(nengo.ensemble.Neurons)
def build_neurons_probe(model, probe):
    """Build a probe which has Neurons as its target."""
    if probe.attr in ("output", "spikes", "voltage"):
        # Get the real target if the target is an ObjView
        if isinstance(probe.target, nengo.base.ObjView):
            ens = probe.target.obj.ensemble
        else:
            ens = probe.target.ensemble

        # Add this probe to the list of probes attached to the ensemble object.
        model.object_operators[ens].local_probes.append(probe)
    else:
        raise NotImplementedError(
            "SpiNNaker does not currently support probing '{}' on '{}' "
            "neurons".format(
                probe.attr,
                probe.target.ensemble.neuron_type.__class__.__name__
            )
        )
