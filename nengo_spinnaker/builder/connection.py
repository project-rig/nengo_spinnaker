import nengo
from nengo.utils.builder import full_transform

from .builder import BuiltConnection, Model


@Model.connection_parameter_builders.register(nengo.base.NengoObject)
def build_generic_connection_params(model, conn):
    transform = full_transform(conn)
    return BuiltConnection(
        decoders=None,
        transform=transform,
        eval_points=None,
        solver_info=None
    )
