import atexit
import logging
import nengo
from nengo.cache import get_default_decoder_cache
import numpy as np
from rig.machine_control import MachineController
from rig.machine_control.consts import AppState
from rig.machine import Cores
import six
import time

from .builder import Model
from .node_io import Ethernet
from .rc import rc
from .utils.config import getconfig
from .utils.machine_control import test_and_boot
from .utils import model as model_optimisations

logger = logging.getLogger(__name__)


class Simulator(object):
    """SpiNNaker simulator for Nengo models.

    The simulator period determines how much data will be stored on SpiNNaker
    and is the maximum length of simulation allowed before data is transferred
    between the machine and the host PC. If the period is set to `None`
    function of time Nodes will not be optimised and probes will be disabled.
    For any other value simulation lengths of less than or equal to the period
    will be in real-time, longer simulations will be possible but will include
    short gaps when data is transferred between SpiNNaker and the host.

    :py:meth:`~.close` should be called when the simulator will no longer be
    used. This will close all sockets used to communicate with the SpiNNaker
    machine and will leave the machine in a clean state. Failure to call
    `close` may result in later failures. Alternatively `with` may be used::

        sim = nengo_spinnaker.Simulator(network)
        with sim:
            sim.run(10.0)
    """
    _open_simulators = set()

    @classmethod
    def _add_simulator(cls, simulator):
        cls._open_simulators.add(simulator)

    @classmethod
    def _remove_simulator(cls, simulator):
        cls._open_simulators.remove(simulator)

    def __init__(self, network, dt=0.001, period=10.0):
        """Create a new Simulator with the given network.

        Parameters
        ----------
        period : float or None
            Duration of one period of the simulator. This determines how much
            memory will be allocated to store precomputed and probed data.
        """
        # Add this simulator to the set of open simulators
        Simulator._add_simulator(self)

        # Create a controller for the machine and boot if necessary
        hostname = rc.get("spinnaker_machine", "hostname")
        machine_width = rc.getint("spinnaker_machine", "width")
        machine_height = rc.getint("spinnaker_machine", "height")

        self.controller = MachineController(hostname)
        test_and_boot(self.controller, hostname, machine_width, machine_height)

        # Create the IO controller
        io_cls = getconfig(network.config, Simulator, "node_io", Ethernet)
        io_kwargs = getconfig(network.config, Simulator, "node_io_kwargs",
                              dict())
        self.io_controller = io_cls(**io_kwargs)

        # Determine the maximum run-time
        self.max_steps = None if period is None else int(period / dt)

        self.steps = 0  # Steps simulated

        # If the simulator is in "run indefinite" mode (i.e., max_steps=None)
        # then we modify the builders to ignore function of time Nodes and
        # probes.
        builder_kwargs = self.io_controller.builder_kwargs
        if self.max_steps is None:
            raise NotImplementedError

        # Create a model from the network, using the IO controller
        logger.debug("Building model")
        start_build = time.time()
        self.model = Model(dt, decoder_cache=get_default_decoder_cache())
        self.model.build(network, **builder_kwargs)
        model_optimisations.remove_childless_filters(self.model)
        logger.info("Build took {:.3f} seconds".format(time.time() -
                                                       start_build))

        self.model.decoder_cache.shrink()
        self.dt = self.model.dt
        self._closed = False  # Whether the simulator has been closed or not

        self.host_sim = self._create_host_sim()

        # Holder for probe data
        self.data = {}

        # Convert the model into a netlist
        logger.info("Building netlist")
        start = time.time()
        self.netlist = self.model.make_netlist(self.max_steps or 0)

        # Get a machine object to place & route against
        logger.info("Getting SpiNNaker machine specification")
        machine = self.controller.get_machine()

        # Place & Route
        logger.info("Placing and routing")
        self.netlist.place_and_route(machine)

        logger.info("{} cores in use".format(len(self.netlist.placements)))
        chips = set(six.itervalues(self.netlist.placements))
        logger.info("Using {}".format(chips))

        # Prepare the simulator against the placed, allocated and routed
        # netlist.
        self.io_controller.prepare(self.model, self.controller, self.netlist)

        # Load the application
        logger.info("Loading application")
        self.netlist.load_application(self.controller)

        # Check if any cores are in bad states
        if self.controller.count_cores_in_state(["exit", "dead", "watchdog",
                                                 "runtime_exception"]):
            for vertex in self.netlist.vertices:
                x, y = self.netlist.placements[vertex]
                p = self.netlist.allocations[vertex][Cores].start
                status = self.controller.get_processor_status(p, x, y)
                if status.cpu_state is not AppState.sync0:
                    print("Core ({}, {}, {}) in state {!s}".format(
                        x, y, p, status.cpu_state))
            raise Exception("Unexpected core failures.")

        logger.info("Preparing and loading machine took {:3f} seconds".format(
            time.time() - start
        ))

        logger.info("Setting router timeout to 16 cycles")
        for x in range(machine_width):
            for y in range(machine_height):
                with self.controller(x=x, y=y):
                    if (x, y) in machine:
                        data = self.controller.read(0xf1000000, 4)
                        self.controller.write(0xf1000000, data[:-1] + b'\x10')

    def __enter__(self):
        """Enter a context which will close the simulator when exited."""
        # Return self to allow usage like:
        #
        #     with nengo_spinnaker.Simulator(model) as sim:
        #         sim.run(1.0)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Exit a context and close the simulator."""
        self.close()

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        # Determine how many steps to simulate for
        steps = int(np.round(float(time_in_seconds) / self.dt))
        self.run_steps(steps)

    def run_steps(self, steps):
        """Simulate a give number of steps."""
        while steps > 0:
            n_steps = min((steps, self.max_steps))
            self._run_steps(n_steps)
            steps -= n_steps

    def _run_steps(self, steps):
        """Simulate for the given number of steps."""
        if self._closed:
            raise Exception("Simulator has been closed and can't be used to "
                            "run further simulations.")

        if steps is None:
            if self.max_steps is not None:
                raise Exception(
                    "Cannot run indefinitely if a simulator period was "
                    "specified. Create a new simulator with Simulator(model, "
                    "period=None) to perform indefinite time simulations."
                )
        else:
            assert steps <= self.max_steps

        # Prepare the simulation
        self.netlist.before_simulation(self, steps)

        # Wait for all cores to hit SYNC0
        self.controller.wait_for_cores_to_reach_state(
            "sync0", len(self.netlist.vertices)
        )
        self.controller.send_signal("sync0")

        # Get a new thread for the IO
        io_thread = self.io_controller.spawn()

        # Run the simulation
        try:
            # Prep
            exp_time = steps * (self.model.machine_timestep / float(1e6))
            io_thread.start()

            # Wait for all cores to hit SYNC1
            self.controller.wait_for_cores_to_reach_state(
                "sync1", len(self.netlist.vertices)
            )
            logger.info("Running simulation...")
            self.controller.send_signal("sync1")

            # Execute the local model
            host_steps = 0
            start_time = time.time()
            run_time = 0.0
            while run_time < exp_time:
                # Run a step
                self.host_sim.step()
                run_time = time.time() - start_time

                # If that step took less than timestep then spin
                time.sleep(0.0001)
                while run_time < host_steps * self.dt:
                    time.sleep(0.0001)
                    run_time = time.time() - start_time
        finally:
            # Stop the IO thread whatever occurs
            io_thread.stop()

        # Check if any cores are in bad states
        if self.controller.count_cores_in_state(["dead", "watchdog",
                                                 "runtime_exception"]):
            for vertex in self.netlist.vertices:
                x, y = self.netlist.placements[vertex]
                p = self.netlist.allocations[vertex][Cores].start
                status = self.controller.get_processor_status(p, x, y)
                if status.cpu_state is not AppState.sync0:
                    print("Core ({}, {}, {}) in state {!s}".format(
                        x, y, p, status.cpu_state))
            raise Exception("Unexpected core failures.")

        # Retrieve simulation data
        start = time.time()
        logger.info("Retrieving simulation data")
        self.netlist.after_simulation(self, steps)
        logger.info("Retrieving data took {:3f} seconds".format(
            time.time() - start
        ))

        # Increase the steps count
        self.steps += steps

    def _create_host_sim(self):
        # change node_functions to reflect time
        # TODO: improve the reference simulator so that this is not needed
        #       by adding a realtime option
        node_functions = {}
        node_info = dict(start=None)
        for node in self.io_controller.host_network.all_nodes:
            if callable(node.output):
                old_func = node.output
                if node.size_in == 0:
                    def func(t, f=old_func):
                        now = time.time()
                        if node_info['start'] is None:
                            node_info['start'] = now
                        return f(now - node_info['start'])
                else:
                    def func(t, x, f=old_func):
                        now = time.time()
                        if node_info['start'] is None:
                            node_info['start'] = now
                        return f(now - node_info['start'], x)
                node.output = func
                node_functions[node] = old_func

        # Build the host simulator
        host_sim = nengo.Simulator(self.io_controller.host_network,
                                   dt=self.dt)
        # reset node functions
        for node, func in node_functions.items():
            node.output = func

        return host_sim

    def close(self):
        """Clean the SpiNNaker board and prevent further simulation."""
        if not self._closed:
            # Stop the application
            self._closed = True
            self.io_controller.close()
            self.controller.send_signal("stop")

            # Remove this simulator from the list of open simulators
            Simulator._remove_simulator(self)

    def trange(self, dt=None):
        return np.arange(1, self.steps + 1) * (self.dt or dt)


@atexit.register
def _close_open_simulators():
    """Close all remaining open simulators."""
    # We make a list to avoid modifying the object we're iterating over.
    for sim in list(Simulator._open_simulators):
        # Ignore any errors which may occur during this shut-down process.
        try:
            sim.close()
        except:
            pass
