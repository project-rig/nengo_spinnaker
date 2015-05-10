import logging
import numpy as np
from rig.machine_control import MachineController

from .builder import Model
from .rc import rc
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

        # Create a model from the network
        # TODO Include the IO builder in the build process
        self.model = Model(dt)
        self.model.build(network)
        self.dt = self.model.dt

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
        # Convert the model into a netlist
        logger.info("Building netlist")
        netlist = self.model.make_netlist(steps)  # TODO remove steps!

        # Get a machine object to place & route against
        logger.info("Getting SpiNNaker machine specification")
        machine = self.controller.get_machine()

        # Place & Route
        logger.info("Placing and routing")
        netlist.place_and_route(machine)

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

        # TODO: Wait for all cores to hit SYNC0
        logger.info("Running simulation...")
        self.controller.send_signal("sync0")

        # TODO: Execute the local model
        import time
        time.sleep(10.)

        # Retrieve simulation data
        logger.info("Retrieving simulation data")
        netlist.after_simulation(self, steps)

        # Done, for now
        # TODO: Allow further simulation
        # self.controller.send_signal("stop")
