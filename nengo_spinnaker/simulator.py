import logging
import nengo
import numpy as np
from rig.machine_control import MachineController
import time

from .builder import Model
from .node_io import Ethernet
from .rc import rc
from .utils.config import getconfig
from .utils.machine_control import test_and_boot

logger = logging.getLogger(__name__)


class SpiNNakerSimulator(object):
    """SpiNNaker simulator for Nengo models."""
    def __init__(self, network, dt=0.001):
        """Create a new Simulator with the given network."""
        # Create a controller for the machine and boot if necessary
        hostname = rc.get("spinnaker_machine", "hostname")
        machine_width = rc.getint("spinnaker_machine", "width")
        machine_height = rc.getint("spinnaker_machine", "height")

        self.controller = MachineController(hostname)
        test_and_boot(self.controller, hostname, machine_width, machine_height)

        # Create the IO controller
        io_cls = getconfig(network.config, SpiNNakerSimulator,
                           "node_io", Ethernet)
        io_kwargs = getconfig(network.config, SpiNNakerSimulator,
                              "node_io_kwargs", dict())
        self.io_controller = io_cls(**io_kwargs)

        # Create a model from the network, using the IO controller
        logger.debug("Building model")
        self.model = Model(dt)
        self.model.build(network, **self.io_controller.builder_kwargs)
        self.dt = self.model.dt

        # Build the host simulator
        self.host_sim = nengo.Simulator(self.io_controller.host_network,
                                        dt=self.dt)

        # Holder for probe data
        self.data = {}

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        # Determine how many steps to simulate for
        steps = int(np.round(float(time_in_seconds) / self.dt))
        self.run_steps(steps)

    def run_steps(self, steps):
        """Simulate for the given number of steps."""
        # NOTE: constructing a netlist, placing, routing and loading should
        # move into Simulator initialisation when the new simulation protocol
        # is implemented.
        self._n_steps_last = steps

        # Convert the model into a netlist
        logger.info("Building netlist")
        netlist = self.model.make_netlist(steps)  # TODO remove steps!

        # Get a machine object to place & route against
        logger.info("Getting SpiNNaker machine specification")
        machine = self.controller.get_machine()

        # Place & Route
        logger.info("Placing and routing")
        netlist.place_and_route(machine)

        # Prepare the simulator against the placed, allocated and routed
        # netlist.
        self.io_controller.prepare(self.controller, netlist)
        io_thread = self.io_controller.spawn()

        # Load the application
        logger.info("Loading application")
        netlist.load_application(self.controller, steps)

        # TODO: Implement a better simulation protocol
        # Check if any cores are in bad states
        failed_cores = (
            self.controller.count_cores_in_state("runtime_exception") +
            self.controller.count_cores_in_state("watchdog") +
            self.controller.count_cores_in_state("dead") +
            self.controller.count_cores_in_state("exit")
        )
        if failed_cores:
            # TODO: Find the failed cores
            raise Exception("Unexpected core failures.")

        try:
            # Prep
            exp_time = steps * self.dt
            io_thread.start()

            # TODO: Wait for all cores to hit SYNC0
            logger.info("Running simulation...")
            self.controller.send_signal("sync0")

            # Execute the local model
            while exp_time > 0:
                # Run a step
                start = time.time()
                self.host_sim.step()
                run_time = time.time() - start

                # If that step took less than timestep then spin
                time.sleep(0.0001)
                while run_time < self.dt:
                    run_time = time.time() - start

                exp_time -= run_time
        finally:
            # Stop the IO thread whatever occurs
            io_thread.stop()

        # Retrieve simulation data
        logger.info("Retrieving simulation data")
        netlist.after_simulation(self, steps)

        # Stop the application
        self.controller.send_signal("stop")
        self.io_controller.close()

    def trange(self, dt=None):
        return np.arange(self._n_steps_last) * (self.dt or dt)
