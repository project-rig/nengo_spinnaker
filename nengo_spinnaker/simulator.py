"""Simulator

The Nengo simulator for SpiNNaker machines.
"""
from rig.machine import SDRAM
from rig.machine_control import MachineController
from rig.place_and_route import wrapper as place_and_route


class Simulator(object):
    """SpiNNaker simulator for Nengo models.

    Initialising a Simulator with a model will cause the model to be built and
    loaded to a given SpiNNaker machine.


    Attributes
    ----------
    n_steps : int
        Number of steps simulated.
    model : :py:class:`~nengo_spinnaker.builder.Model`
        Model being simulated.
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
        self.n_steps = 0
        # Prepare the model for simulation and load it onto the SpiNNaker
        # machine ready to go.  We delegate the task of building the model into
        # an object-graph to the Builder, the objects it returns are directly
        # useable to perform place-and-route for a SpiNNaker machine.  Having
        # placed and routed we generate data to load onto the SpiNNaker board
        # in preparation for the simulation; we load this data and the
        # applications and are then ready for the simulation to start.
        self.model = build_model(network, dt, seed)

        # Build up a connection to the machine to simulate the network on
        # TODO Get the hostname for the machine somewhere, whether it should be
        # booted and arguments relating to that.
        self.controller = MachineController()

        # Boot the machine if necessary
        if boot_machine:
            self.controller.boot(machine_width, machine_height, **boot_kwargs)

        # Discover the properties of the machine we're simulating on
        self.machine = self.controller.get_machine()

        # Place and route
        (self.placements, self.allocations, self.application_map,
         self.routing_tables) = \
            place_and_route(self.model.vertex_resources,
                            self.model.vertices_applications,
                            self.model.nets,
                            self.model.net_keys,
                            self.machine,
                            self.model.constraints)

        # Load the data by calling each prepare SpiNNaker callback
        for fn in self.model.prepare_spinnaker_callbacks:
            fn(self)

        # Load routing tables and applications
        self.controller.load_routing_tables(self.routing_tables)
        self.allocations.load_applications(self.application_map)

    def run(self, time_in_seconds):
        """Run the simulation for a given period of time.

        Parameters
        ----------
        time_in_seconds : float
            Time for which to run the simulation expressed in seconds. If not
            specified then the simulation will run indefinitely.
        """
        # TODO Deal with continuous execution for, e.g., robotic experiments.
        # Determine how many steps will be necessary and then run for this
        # number of steps.
        n_steps = int(math.ceil(float(time_in_seconds) / self.model.dt))
        self.run_steps(n_steps)

    def step(self):
        """Simulate a single step of `dt`."""
        self.run_steps(1)

    def run_steps(self, steps):
        """Run the simulation for a given number of simulation steps.

        Parameters
        ----------
        steps : int
            Number of `dt` steps to simulate.
        """
        while steps > 0:
            # Determine how many steps can be run, run for this many steps and
            # then repeat.
            n_steps = min(steps, self.model.max_steps)
            self._run_steps(n_steps)

            # Modify the current progress of the simulator, then update the
            # number of steps we have left to run.
            self.n_steps += n_steps
            steps -= n_steps

    def _run_steps(self, steps):
        """Run the simulation for a given number of steps.

        ..warning::
            The number of steps is not broken up.  It is strongly recommended
            to use :py:meth:`.run_steps` instead which will run steps in blocks
            of as many steps can be handled.
        """
        raise NotImplementedError
        # ENSURE ALL REQUIRED CORES ARE IN SYNC0
        # 1) Load any data we need for this next set of steps (e.g., nodes
        #    which are functions of time).
        for fn in self.model.pre_simulation_callbacks:
            fn(self, self.n_steps, steps)
        # PASS SYNC0
        # ENSURE ALL CORES ARE IN SYNC1
        # PASS SYNC1

        # 2) Run for the given number of steps/as many steps as we have memory
        #    for.

        # ENSURE ALL CORES ARE IN SYNC0
        # 3) Retrieve data accumulated during the last block of simulation.
        for fn in self.model.post_simulation_callbacks:
            fn(self, steps)

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

    def _allocate_memory_for_vertex(self, vertex):
        """Allocate SDRAM for a vertex and get the reserved memory region as a
        file-like object.
        """
        # Determine the amount of memory to allocate
        n_bytes = self.model.vertex_resources[vertex][SDRAM]

        # Get the co-ordinates and the tag
        assert len(self.placements[vertex] == 1)
        (x, y), ps = next(six.iteritems(self.placements[vertex]))
        assert len(ps) == 1
        tag = ps[0]

        # Perform the allocation
        return self.controller.sdram_alloc_as_filelike(
            n_bytes, tag=tag, x=x, y=y)
