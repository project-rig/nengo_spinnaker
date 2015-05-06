import collections

BuiltConnection = collections.namedtuple(
    "BuiltConnection", "decoders, eval_points, transform, solver_info"
)
"""Parameters which describe a Connection."""
