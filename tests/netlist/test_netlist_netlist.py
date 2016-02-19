import mock
import pytest
from rig.bitfield import BitField
from rig.place_and_route import Cores, SDRAM
from rig.place_and_route.constraints import (ReserveResourceConstraint,
                                             LocationConstraint)

from nengo_spinnaker import netlist
from nengo_spinnaker.utils.itertools import flatten


def test_before_simulation():
    """Test that all methods are called when asked to prepare a simulation and
    that the simulation duration is written in correctly.
    """
    # Create some "before_simulation" functions
    before_a = mock.Mock()
    before_b = mock.Mock()

    # Create a vertex
    vertex = mock.Mock()

    # Create a netlist
    model = netlist.Netlist(
        nets=[],
        vertices=[vertex],
        keyspaces={},
        groups={},
        load_functions=[],
        before_simulation_functions=[before_a, before_b]
    )
    model.placements[vertex] = (1, 2)
    model.allocations[vertex] = {Cores: slice(5, 7)}

    # Call the before_simulation_functions
    simulator = mock.Mock(name="Simulator")
    model.before_simulation(simulator, 100)

    before_a.assert_called_once_with(model, simulator, 100)
    before_b.assert_called_once_with(model, simulator, 100)

    # Check we wrote in the run time
    simulator.controller.write_vcpu_struct_field.assert_called_once_with(
        "user1", 100, 1, 2, 5
    )


def test_after_simulation():
    """Test that all methods are called when asked to finish a simulation."""
    # Create some "before_simulation" functions
    after_a = mock.Mock()
    after_b = mock.Mock()

    # Create a netlist
    model = netlist.Netlist(
        nets=[],
        vertices=[],
        keyspaces={},
        groups={},
        load_functions=[],
        after_simulation_functions=[after_a, after_b]
    )

    # Call the before_simulation_functions
    simulator = mock.Mock(name="Simulator")
    model.after_simulation(simulator, 100)

    after_a.assert_called_once_with(model, simulator, 100)
    after_b.assert_called_once_with(model, simulator, 100)
