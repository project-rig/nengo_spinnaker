import atexit
import logging
import nengo
from nengo.cache import get_default_decoder_cache
import numpy as np
from rig.machine_control import MachineController
from rig.machine_control.consts import AppState
from rig.place_and_route import Cores
import rig.place_and_route
import six
import time

from .builder import Model
from .node_io import Ethernet
from .rc import rc
from .utils.config import getconfig
from .utils.model import (get_force_removal_passnodes,
                          optimise_out_passthrough_nodes)

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

    def __init__(self, network, dt=0.001, period=10.0, timescale=1.0,
                 hostname=None, use_spalloc=None,
                 allocation_fudge_factor=0.6):
        """Create a new Simulator with the given network.

        Parameters
        ----------
        period : float or None
            Duration of one period of the simulator. This determines how much
            memory will be allocated to store precomputed and probed data.
        timescale : float
            Scaling factor to apply to the simulation, e.g., a value of `0.5`
            will cause the simulation to run at half real-time.
        hostname : string or None
            Hostname of the SpiNNaker machine to use; if None then the machine
            specified in the config file will be used.
        use_spalloc : bool or None
            Allocate a SpiNNaker machine for the simulator using ``spalloc``.
            If None then the setting specified in the config file will be used.

        Other Parameters
        ----------------
        allocation_fudge_factor:
           Fudge factor to allocate more cores than really necessary when using
           `spalloc` to ensure that (a) there are sufficient "live" cores in
           the allocated machine, (b) there is sufficient room for a good place
           and route solution. This should generally be more than 0.1 (10% more
           cores than necessary) to account for the usual rate of missing
           chips.
        """
        # Add this simulator to the set of open simulators
        Simulator._add_simulator(self)

        # Create the IO controller
        io_cls = getconfig(network.config, Simulator, "node_io", Ethernet)
        io_kwargs = getconfig(network.config, Simulator, "node_io_kwargs",
                              dict())
        self.io_controller = io_cls(**io_kwargs)

        # Calculate the machine timestep, this is measured in microseconds
        # (hence the 1e6 scaling factor).
        self.timescale = timescale
        machine_timestep = int((dt / timescale) * 1e6)

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
        self.model = Model(dt=dt, machine_timestep=machine_timestep,
                           decoder_cache=get_default_decoder_cache())
        self.model.build(network, **builder_kwargs)

        forced_removals = get_force_removal_passnodes(network)
        optimise_out_passthrough_nodes(self.model,
                                       self.io_controller.passthrough_nodes,
                                       network.config, forced_removals)

        logger.info("Build took {:.3f} seconds".format(time.time() -
                                                       start_build))

        self.model.decoder_cache.shrink()
        self.dt = self.model.dt
        self._closed = False  # Whether the simulator has been closed or not

        self.host_sim = self._create_host_sim()

        # Holder for probe data
        self.data = {}

        # Holder for profiling data
        self.profiler_data = {}

        # Convert the model into a netlist
        logger.info("Building netlist")
        start = time.time()
        self.netlist = self.model.make_netlist(self.max_steps or 0)

        # Determine whether to use a spalloc machine or not
        if use_spalloc is None:
            # Default is to not use spalloc; this is indicated by either the
            # absence of the option in the config file OR the option being set
            # to false.
            use_spalloc = (
                rc.has_option("spinnaker_machine", "use_spalloc") and
                rc.getboolean("spinnaker_machine", "use_spalloc"))

        # Create a controller for the machine and boot if necessary
        self.job = None
        if not use_spalloc:
            # Use the specified machine rather than trying to get one
            # allocated.
            if hostname is None:
                hostname = rc.get("spinnaker_machine", "hostname")
        else:
            # Attempt to get a machine allocated to us
            from spalloc import Job

            # Determine how many boards to ask for (assuming 16 usable cores
            # per chip and 48 chips per board).
            n_cores = (sum(v.resources.get(Cores, 0) for v in
                           self.netlist.vertices) *
                       (1.0 + allocation_fudge_factor))
            n_boards = int(np.ceil((n_cores / 16.) / 48.))

            # Request the job
            self.job = Job(n_boards)
            logger.info("Allocated job ID %d...", self.job.id)

            # Wait until we're given the machine
            logger.info("Waiting for machine allocation...")
            self.job.wait_until_ready()

            # spalloc recommends a slight delay before attempting to boot the
            # machine, later versions of spalloc server may relax this
            # requirement.
            time.sleep(5.0)

            # Store the hostname
            hostname = self.job.hostname
            logger.info("Using %d board(s) of \"%s\" (%s)",
                        len(self.job.boards), self.job.machine_name, hostname)

        self.controller = MachineController(hostname)
        self.controller.boot()

        # Get a system-info object to place & route against
        logger.info("Getting SpiNNaker machine specification")
        system_info = self.controller.get_system_info()

        # Place & Route
        logger.info("Placing and routing")
        self.netlist.place_and_route(
            system_info,
            place=getconfig(network.config, Simulator,
                            'placer', rig.place_and_route.place),
            place_kwargs=getconfig(network.config, Simulator,
                                   'placer_kwargs', {}),
        )

        logger.info("{} cores in use".format(len(self.netlist.placements)))
        chips = set(six.itervalues(self.netlist.placements))
        logger.info("Using {}".format(chips))

        # Prepare the simulator against the placed, allocated and routed
        # netlist.
        self.io_controller.prepare(self.model, self.controller, self.netlist)

        # Load the application
        logger.info("Loading application")
        self.netlist.load_application(self.controller, system_info)

        # Check if any cores are in bad states
        if self.controller.count_cores_in_state(["exit", "dead", "watchdog",
                                                 "runtime_exception"]):
            for vertex in self.netlist.vertices:
                x, y = self.netlist.placements[vertex]
                p = self.netlist.allocations[vertex][Cores].start
                status = self.controller.get_processor_status(p, x, y)
                if status.cpu_state is not AppState.sync0:
                    print("Core ({}, {}, {}) in state {!s}".format(
                        x, y, p, status))
                    print(self.controller.get_iobuf(p, x, y))
            raise Exception("Unexpected core failures.")

        logger.info("Preparing and loading machine took {:3f} seconds".format(
            time.time() - start
        ))

        logger.info("Setting router timeout to 16 cycles")
        for x, y in system_info.chips():
            with self.controller(x=x, y=y):
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

        # Wait for all cores to hit SYNC0 (either by remaining it or entering
        # it from init)
        self._wait_for_transition(AppState.init, AppState.sync0,
                                  len(self.netlist.vertices))
        self.controller.send_signal("sync0")

        # Get a new thread for the IO
        io_thread = self.io_controller.spawn()

        # Run the simulation
        try:
            # Prep
            exp_time = steps * self.dt / self.timescale
            io_thread.start()

            # Wait for all cores to hit SYNC1
            self._wait_for_transition(AppState.sync0, AppState.sync1,
                                      len(self.netlist.vertices))
            logger.info("Running simulation...")
            self.controller.send_signal("sync1")

            # Execute the local model
            host_steps = 0
            start_time = time.time()
            run_time = 0.0
            local_timestep = self.dt / self.timescale
            while run_time < exp_time:
                # Run a step
                self.host_sim.step()
                run_time = time.time() - start_time

                # If that step took less than timestep then spin
                time.sleep(0.0001)
                while run_time < host_steps * local_timestep:
                    time.sleep(0.0001)
                    run_time = time.time() - start_time
        finally:
            # Stop the IO thread whatever occurs
            io_thread.stop()

        # Wait for cores to re-enter sync0
        self._wait_for_transition(AppState.run, AppState.sync0,
                                  len(self.netlist.vertices))

        # Retrieve simulation data
        start = time.time()
        logger.info("Retrieving simulation data")
        self.netlist.after_simulation(self, steps)
        logger.info("Retrieving data took {:3f} seconds".format(
            time.time() - start
        ))

        # Increase the steps count
        self.steps += steps

    def _wait_for_transition(self, from_state, desired_to_state, num_verts):
        while True:
            # If no cores are still in from_state, stop
            if self.controller.count_cores_in_state(from_state) == 0:
                break

            # Wait a bit
            time.sleep(1.0)

        # Check if any cores haven't exited cleanly
        num_ready = self.controller.wait_for_cores_to_reach_state(
            desired_to_state, num_verts, timeout=5.0)

        if num_ready != num_verts:
            # Loop through all placed vertices
            for vertex, (x, y) in six.iteritems(self.netlist.placements):
                p = self.netlist.allocations[vertex][Cores].start
                status = self.controller.get_processor_status(p, x, y)
                if status.cpu_state is not desired_to_state:
                    print("Core ({}, {}, {}) in state {!s}".format(
                        x, y, p, status.cpu_state))
                    print(self.controller.get_iobuf(p, x, y))

            raise Exception("Unexpected core failures before reaching %s "
                            "state." % desired_to_state)

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

                        t = (now - node_info['start']) * self.timescale
                        return f(t)
                else:
                    def func(t, x, f=old_func):
                        now = time.time()
                        if node_info['start'] is None:
                            node_info['start'] = now

                        t = (now - node_info['start']) * self.timescale
                        return f(t, x)
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

            # Destroy the job if we allocated one
            if self.job is not None:
                self.job.destroy()

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
