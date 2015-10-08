import collections


class netlistspec(collections.namedtuple(
        "netlistspec", "vertices, load_function, before_simulation_function, "
                       "after_simulation_function, constraints")):
    """Specification of how an operator should be added to a netlist."""
    def __new__(cls, vertices, load_function=None,
                before_simulation_function=None,
                after_simulation_function=None, constraints=None):
        return super(netlistspec, cls).__new__(
            cls, vertices, load_function, before_simulation_function,
            after_simulation_function, constraints
        )
