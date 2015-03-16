Nengo/SpiNNaker Design
======================

There are four fundamental Nengo objects:

 * Ensembles
 * Nodes
 * Probes
 * Connections

Each with different requirements of the SpiNNaker system.  Ensembles will
typically end up mapped to several processing cores and connections map to
streams of multicast packets.  However, Nodes may represent any of the
following things:

 * "Passthrough" Nodes - which aren't required at all on SpiNNaker and can be
   removed by changing the Connections present in a network.
 * Constant inputs - these may be combined into Ensemble bias currents.
 * Functions of time - inputs whose value varies only with time.  The output of
   these functions can be precomputed and played back on the SpiNNaker machine.
    * Periodic functions can be simplified by only computing one period and
      looping the playback.
 * Functions of other values in the network - these may be simulated on the
   host PC at the cost of reducing their sampling frequency; communication
   between SpiNNaker and the PC may be managed by various means, and this will
   change the on-SpiNNaker requirements of the node.
 * External devices (either sensors or actuators) which map to devices
   connected to the SpiNNaker network and may require additional compute
   resources on chip to provide them with input/output data processing.

Likewise Probes have different incarnations:

 * Probes of "output values" (e.g., decoded values of populations, the outputs
   of nodes) are implemented as cores which receive and store multicast
   packets.
 * Probes of spikes or neuron voltages are implemented as blocks of memory
   managed by the appropriate ensemble processing cores.

There are also two modes of operation, depending upon the experiment being run
or the needs of the modeller.

 * In "normal" mode any values which are being recorded (e.g., by Probes) or
   played back (e.g., by Nodes whose output is purely a function of time) can
   be read from or written to the SpiNNaker machine periodically during the
   simulation.  Consequently simulating a model for a period of time is a
   repetition of three steps:

    1. Load data needed for the next _n_ steps.
    2. Simulate _n_ steps.
    3. Retrieve data generated during these steps.

   In this case the simulation may be run for any finite period of time, paused
   and restarted, or reset and restarted as required.

 * In some cases it may not be desirable to stop the simulation to load and
   retrieve data (e.g., when driving a robot or interacting with other
   real-time devices).  In these cases, iff the duration of the simulation is
   known in advance then enough memory may be reserved to store all the data that
   will be required or generated in the simulation.  Once simulation is
   complete the SpiNNaker machine may need reconfiguring before further
   simulations.

    * As an additional case: the duration of the simulation may not be known and
      interruptions may not be made.  In this case probing must either be
      disabled, or data streamed over the network or read back during the
      simulation.

Preparing Nengo models for simulation
-------------------------------------

The general build process is as follows:
 1. Remove passthrough nodes.
 2. Partition into the SpiNNaker network and the PC network
     - Intermediate objects to handle IO for nodes are added.
 3. Build objects and connections:
     - Build ensembles
     - Build Node IO objects
       - This process is pluggable to allow building IO for robotics or for
         optimising out constant Nodes.
 4. Place and route.
 5. Generate and load data, load applications, load routing tables.
     - TBD how data is best generated when local memory is at a premium...
