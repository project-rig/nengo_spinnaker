import nengo
from nengo.utils.builder import full_transform

from .builder import (
    BuiltConnection, InputPort, Model, ObjectPort, OutputPort, spec
)


@Model.source_getters.register(nengo.base.NengoObject)
def generic_source_getter(model, conn):
    obj = model.object_operators[conn.pre_obj]
    return spec(ObjectPort(obj, OutputPort.standard))


@Model.sink_getters.register(nengo.base.NengoObject)
def generic_sink_getter(model, conn):
    obj = model.object_operators[conn.post_obj]
    return spec(ObjectPort(obj, InputPort.standard))


@Model.connection_parameter_builders.register(nengo.base.NengoObject)
def build_generic_connection_params(model, conn):
    return BuiltConnection(
        decoders=None,
        transform=full_transform(conn, slice_pre=False, allow_scalars=False),
        eval_points=None,
        solver_info=None
    )
