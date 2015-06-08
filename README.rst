SpiNNaker based Nengo simulator
###############################

.. image:: https://travis-ci.org/project-rig/nengo_spinnaker.svg?branch=master
   :alt: Build Status
   :target: https://travis-ci.org/project-rig/nengo_spinnaker
.. image:: https://coveralls.io/repos/project-rig/nengo_spinnaker/badge.svg?branch=master
   :alt: Coverage Status
   :target: https://coveralls.io/r/project-rig/nengo_spinnaker?branch=master

``nengo_spinnaker`` is a SpiNNaker-based simulator for models built using
`Nengo <https://github.com/nengo/nengo>`_. It allows real-time simulation of
large-scale models.

Quick Start
===========

Install using ``pip``::

    pip install nengo_spinnaker

Settings File
-------------

To use SpiNNaker with Nengo you must create a ``nengo_spinnaker.conf`` file in
either the directory you will be running your code from or, more usefully, a
centralised location. The centralised location varies based on your operating
system:

- Windows: ``%userprofile%\nengo\nengo_spinnaker.conf``
- Other: ``~/.config/nengo/nengo_spinnaker.conf``

This file exists to inform ``nengo_spinnaker`` of the nature of the SpiNNaker
machine you wish to simulate with and how to communicate with it. This file may
look like::

    ### SpiNNaker system configuration
    #
    # Settings for the SpiNNaker machine which will be used to simulate Nengo
    # models. 

    [spinnaker_machine]
    hostname: <host name of the machine here>
    width: <width of the machine here>
    height: <height of the machine here>

    # Required parameters are:
    #   - hostname: (string) either the hostname or the IP address of the board
    #         containing chip (0, 0).
    #   - width: (int) width of the machine (0 <= width < 256)
    #   - height: (int) height of the machine (0 <= height < 256)
    #
    # Optional parameters are:
    #   - "hardware_version: (int) Version number of the SpiNNaker boards
    #         used in the system (e.g. SpiNN-5 boards would be 5). At the
    #         time of writing this value is ignored and can be safely set to
    #         the default value of 0.
    #   - "led_config": (int) Defines LED pin numbers for the SpiNNaker boards
    #         used in the system.  The four least significant bits (3:0) give
    #         the number of LEDs. The next four bits give the pin number of the
    #         first LED, the next four the pin number of the second LED, and so
    #         forth. At the time of writing, all SpiNNaker board versions have
    #         their first LED attached to pin 0 and thus the default value of
    #         0x00000001 is safe. 
    # 
    # For a Spin3 board connected to 192.168.240.253 this section would look
    # like:
    # 
    # hostname: 192.168.240.253
    # width: 2
    # height: 2
    # hardware_version: 3
    # led_config: 0x00000502
    #
    # For a Spin5 board connected to 192.168.1.1 this section would look
    # like:
    # 
    # hostname: 192.168.1.1
    # width: 8
    # height: 8
    # hardware_version: 5
    # led_config: 0x00000001


Using ``nengo_spinnaker``
-------------------------

To use SpiNNaker to simulate your Nengo model first construct the model as
normal. Then use ``nengo_spinnaker.Simulator`` to simulate your model.::

    import nengo_spinnaker

    # Build model as normal

    sim = nengo_spinnaker.Simulator(network)
    sim.run(10.0)

    # When done
    sim.close()

After running your model you must call ``close`` to leave the SpiNNaker machine
in a clean state. Alternatively a ``with`` block may be used to ensure the
simulator is closed after use::

    with sim:
        sim.run(10.0)

Some specific configuration options are available for SpiNNaker. To use these::

    # Modify config to use SpiNNaker parameters
    nengo_spinnaker.add_spinnaker_params(network.config)

Current settings are:

* ``function_of_time`` - Mark a Node as being a function of time only.
* ``function_of_time_period`` - Provide the period of the Node.

For example::

    with model:
        signal = nengo.Node(lambda t: np.sin(t))

    nengo_spinnaker.add_params(model.config)
    model.config[signal].function_of_time = True


Developers
==========

See `DEVELOP.md`__ for information on how to get involved in
``nengo_spinnaker`` development and how to install and build the latest copy of
``nengo_spinnaker``.

__ ./DEVELOP.md
