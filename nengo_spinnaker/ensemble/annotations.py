import collections
import nengo
from nengo.utils.builder import full_transform
import numpy as np

from .. import annotations


class AnnotatedEnsemble(annotations.ObjectAnnotation):
    """Thin wrapper around an ensemble that can store input from
    constant-valued Nodes and indicate which voltage/spike probes are present.

    Attributes
    ----------
    direct_input : nd_array
        NumPy array (the same shape as the Ensemble's input) which can store
        input from constant-valued Nodes.
    local_probes : [:py:class:`nengo.Probe`, ...]
        Probes which store data local to the Ensemble (e.g., voltage or
        spikes).
    """
    __slots__ = ["direct_input", "local_probes"]

    def __init__(self, ensemble):
        """Create an intermediate representation for an Ensemble."""
        super(AnnotatedEnsemble, self).__init__(ensemble)

        # Create a holder for direct inputs and a list of local probes
        self.direct_input = np.zeros(ensemble.size_in)
        self.local_probes = list()

    @classmethod
    def from_model(cls, obj, built_obj):
        """Create a new annotation from an ensemble and its parameters."""
        return cls(obj)

# Register the annotated ensemble builder
annotations.Annotations.object_builders[nengo.Ensemble] =\
    AnnotatedEnsemble.from_model


@annotations.Annotations.sink_getters.register(nengo.Ensemble)
def get_ensemble_sink(conn, irn):
    """Get the sink object for a connection into an Ensemble.

    Parameters
    ----------
    conn : :py:class:`nengo.Connection`
    irn : `Annotations`
    """
    if (isinstance(conn.pre_obj, nengo.Node) and
            not isinstance(conn.pre_obj.output, collections.Callable)):
        # We can optimise out connections from constant values Nodes by
        # eventually including their contributions in bias currents, we do this
        # by annotating the intermediate representation and refusing to accept
        # the connection.
        if conn.function is None:
            val = conn.pre_obj.output[conn.pre_slice]
        else:
            val = conn.function(conn.pre_obj.output[conn.pre_slice])

        irn.objects[conn.post_obj].direct_input += np.dot(
            full_transform(conn, slice_pre=False), val)
        return None  # No connection should be made

    # Otherwise connecting to an Ensemble is just like connecting to anything
    # else.
    return annotations.get_sink_standard(conn, irn)


@annotations.Annotations.source_getters.register(nengo.ensemble.Neurons)
def get_neurons_source(conn, irn):
    """Get the source object (or an existing net to reuse) for a connection out
    of an Ensemble's neurons.

    Parameters
    ----------
    conn : :py:class:`nengo.Connection`
    irn : `Annotations`
    """
    assert isinstance(conn.post_obj, nengo.ensemble.Neurons)
    ensemble = conn.pre_obj.ensemble

    # See if we already have a connection from this Ensemble's neurons
    outgoing = irn.get_nets_starting_at(irn.objects[ensemble])
    if len(outgoing[annotations.OutputPort.neurons]) == 1:
        # We do, so we'll reuse this net
        return list(outgoing[annotations.OutputPort.neurons].keys())[0]
    else:
        # We don't, so we return a source
        return annotations.soss(
            annotations.NetAddress(irn.objects[ensemble],
                                   annotations.OutputPort.neurons),
            weight=ensemble.n_neurons
        )


@annotations.Annotations.sink_getters.register(nengo.ensemble.Neurons)
def get_neurons_sink(conn, irn):
    """Get the sink object for a connection into an Ensemble's neurons.

    Parameters
    ----------
    conn : :py:class:`nengo.Connection`
    irn : `Annotations`
    """
    if isinstance(conn.pre_obj, nengo.ensemble.Neurons):
        # Neurons -> Neurons connection
        return annotations.soss(
            annotations.NetAddress(irn.objects[conn.post_obj.ensemble],
                                   annotations.InputPort.neurons)
        )
    elif (conn.transform.ndim > 0 and
            np.all(conn.transform == conn.transform[0])):
        # This is a global inhibition connection and can be optimised
        return annotations.soss(
            annotations.NetAddress(irn.objects[conn.post_obj.ensemble],
                                   annotations.InputPort.global_inhibition),
            {}
        )
    raise NotImplementedError


@annotations.Annotations.probe_builders.register(nengo.ensemble.Neurons)
def get_neurons_probe(probe, seed, irn):
    """Add the probe for a set of Neurons.

    Parameters
    ----------
    probe : :py:class:`nengo.Probe`
    seed : int
    irn : :py:class:`.Annotations`
    """
    # Add the probe to the intermediate representation for the targeted
    # ensemble
    irn.objects[probe.target.ensemble].local_probes.append(probe)

    # Return no extra objects or connections
    return None, [], []
