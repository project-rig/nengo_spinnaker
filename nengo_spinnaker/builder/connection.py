import nengo
import numpy as np

from .builder import Model, ObjectPort, spec
from .model import ReceptionParameters, InputPort, OutputPort


@Model.source_getters.register(nengo.base.NengoObject)
def generic_source_getter(model, conn):
    obj = model.object_operators[conn.pre_obj]
    return spec(ObjectPort(obj, OutputPort.standard))


@Model.sink_getters.register(nengo.base.NengoObject)
def generic_sink_getter(model, conn):
    obj = model.object_operators[conn.post_obj]
    return spec(ObjectPort(obj, InputPort.standard))


@Model.reception_parameter_builders.register(nengo.base.NengoObject)
@Model.reception_parameter_builders.register(nengo.connection.LearningRule)
@Model.reception_parameter_builders.register(nengo.ensemble.Neurons)
def build_generic_reception_params(model, conn):
    """Build parameters necessary for receiving packets that simulate this
    connection.
    """
    # Just extract the synapse from the connection.
    return ReceptionParameters(conn.synapse, conn.post_obj.size_in,
                               conn.learning_rule)


class EnsembleTransmissionParameters(object):
    """Transmission parameters for a connection originating at an Ensemble.

    Attributes
    ----------
    decoders : array
        Decoders to use for the connection.
    """
    def __init__(self, decoders, transform, learning_rule):
        # Copy the decoders
        self.untransformed_decoders = np.array(decoders)
        self.transform = np.array(transform)

        # Cache learning rule
        self.learning_rule = learning_rule

        # Compute and store the transformed decoders
        self.decoders = np.dot(transform, decoders.T)

        # Make the arrays read-only
        self.untransformed_decoders.flags['WRITEABLE'] = False
        self.transform.flags['WRITEABLE'] = False
        self.decoders.flags['WRITEABLE'] = False

    def __ne__(self, other):
        return not (self == other)

    def __eq__(self, other):
        # Equal iff. the objects are of the same type
        if type(self) is not type(other):
            return False

        # Equal iff. neither connection has a learning rule
        if self.learning_rule is not None or other.learning_rule is not None:
            return False

        # Equal iff. the decoders are the same shape
        if self.decoders.shape != other.decoders.shape:
            return False

        # Equal iff. the decoder values are the same
        if np.any(self.decoders != other.decoders):
            return False

        return True


class PassthroughNodeTransmissionParameters(object):
    """Parameters describing connections which originate from pass through
    Nodes.
    """
    def __init__(self, transform):
        # Store the parameters, copying the transform
        self.transform = np.array(transform)

    def __ne__(self, other):
        return not (self == other)

    def __eq__(self, other):
        # Equivalent if the same type
        if type(self) is not type(other):
            return False

        # and the transforms are equivalent
        if (self.transform.shape != other.transform.shape or
                np.any(self.transform != other.transform)):
            return False

        return True


class NodeTransmissionParameters(PassthroughNodeTransmissionParameters):
    """Parameters describing connections which originate from Nodes."""
    def __init__(self, pre_slice, function, transform):
        # Store the parameters
        super(NodeTransmissionParameters, self).__init__(transform)
        self.pre_slice = pre_slice
        self.function = function

    def __hash__(self):
        # Hash by ID
        return hash(id(self))

    def __eq__(self, other):
        # Parent equivalence
        if not super(NodeTransmissionParameters, self).__eq__(other):
            return False

        # Equivalent if the pre_slices are exactly the same
        if self.pre_slice != other.pre_slice:
            return False

        # Equivalent if the functions are the same
        if self.function is not other.function:
            return False

        return True
