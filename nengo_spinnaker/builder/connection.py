import nengo
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
@Model.reception_parameter_builders.register(nengo.ensemble.Neurons)
def build_generic_reception_params(model, conn):
    """Build parameters necessary for receiving packets that simulate this
    connection.
    """
    # Just extract the synapse from the connection.
    return ReceptionParameters(conn.synapse, conn.post_obj.size_in)
