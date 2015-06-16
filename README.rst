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

    $ pip install nengo_spinnaker

Configure ``nengo_spinnaker`` to use your local SpiNNaker system::

    $ nengo_spinnaker_setup


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

    nengo_spinnaker.add_spinnaker_params(model.config)
    model.config[signal].function_of_time = True

Settings File
-------------

In order to know which SpiNNaker system to use, ``nengo_spinnaker`` uses a
config file called ``nengo_spinnaker.conf`` file in either the directory you
will be running your code from or, more usefully, a centralised location. The
centralised location varies based on your operating system:

- Windows: ``%userprofile%\.nengo\nengo_spinnaker.conf``
- Other: ``~/.config/nengo/nengo_spinnaker.conf``

A utility called ``nengo_spinnaker_setup`` installed with ``nengo_spinnaker``
can be used to create this file. By default, the config file is created
centrally but adding the ``--project`` option will create a config file in the
current directory which applies only  .

An annotated `example config file <./nengo_spinnaker.conf.example>`_ is provided
for users who wish to create their config file by hand.


Developers
==========

See `DEVELOP.md`__ for information on how to get involved in
``nengo_spinnaker`` development and how to install and build the latest copy of
``nengo_spinnaker``.

__ ./DEVELOP.md
