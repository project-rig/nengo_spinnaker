"""Simulator

The Nengo simulator for SpiNNaker machines.
"""
from rig.machine_control import MachineController
from rig.place_and_route import wrapper as place_and_route


class Simulator(object):
    """SpiNNaker simulator for Nengo models.

    Initialising a Simulator with a model will cause the model to be built and
    loaded to a given SpiNNaker machine.
    """

    def __init__(self, network, dt=0.001, seed=None):
        """Initialise the simulator with a given neural network.

        Parameters
        ----------
        network : :py:class:`nengo.Network`
            Nengo network to simulate.
        dt : float
            Granularity of simulation time-steps.
        seed :
            Seed to use for random number generators.
        """
        raise NotImplementedError
        # Prepare the model for simulation and load it onto the SpiNNaker
        # machine ready to go.  We delegate the task of building the model into
        # an object-graph to the Builder, the objects it returns are directly
        # useable to perform place-and-route for a SpiNNaker machine.  Having
        # placed and routed we generate data to load onto the SpiNNaker board
        # in preparation for the simulation; we load this data and the
        # applications and are then ready for the simulation to start.
        vertices_resources, vertices_applications, nets, net_keys, constraints\
            = Builder(network, dt, seed)

        # Build up a connection to the machine to simulate the network on
        # TODO Get the hostname for the machine somewhere, whether it should be
        # booted and arguments relating to that.
        cn = MachineController()

        # Boot the machine if necessary
        if boot_machine:
            cn.boot(machine_width, machine_height, **boot_kwargs)

        # Discover the properties of the machine we're simulating on
        machine = cn.get_machine()

        # Place and route
        placements, allocations, application_map, routing_tables = \
            place_and_route(vertices_resources, vertices_applications, nets,
                            net_keys, machine, constraints)

        # Load the data

        # Load routing tables and applications
        cn.load_routing_tables(routing_tables)
        cn.load_applications(application_map)

    def run(self, time_in_seconds):
        """Run the simulation for a given period of time.

        Parameters
        ----------
        time_in_seconds : float
            Time for which to run the simulation expressed in seconds. If not
            specified then the simulation will run indefinitely.
        """
        raise NotImplementedError
        # Determine how many steps will be necessary and then run for this
        # number of steps.

    def step(self):
        """Simulate a single step of `dt`."""
        raise NotImplementedError
        # Run for a single step

    def run_steps(self, steps):
        """Run the simulation for a given number of simulation steps.

        Parameters
        ----------
        steps : int
            Number of `dt` steps to simulate.
        """
        raise NotImplementedError
        # Ensure that the simulators are in a neutral state, then repeatedly
        # run and retrieve probe data.
        # While the appropriate number of steps haven't been completed:
        # 1) Load any data we need for this next set of steps (e.g., nodes
        #    which are functions of time).
        # 2) Run for the given number of steps/as many steps as we have memory
        #    for.
        # 3) Retrieve data accumulated during the last block of simulation.

    def reset(self):
        """Reset the simulator state.

        Returns the simulator to time t=0 and resets all probed data.
        """
        raise NotImplementedError
        # Send a "reset" signal to the application?

    def trange(self, dt=None):
        """Create a range of times matching probe data.

        Parameters
        ----------
        dt : float (optional)
            The sampling period of the probe to create a range for.  If empty
            the default probe sampling period will be used.
        """
        raise NotImplementedError
