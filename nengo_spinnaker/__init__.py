"""
Nengo SpiNNaker
===============

nengo_spinnaker provides a means of running models built using Nengo
(https://github.com/nengo/nengo) on the SpiNNaker
(http://apt.cs.manchester.ac.uk) platform.
"""

from .config import add_spinnaker_params
from .simulator import Simulator
