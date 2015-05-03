"""SpiNNaker Simulator for Nengo."""
from nengo.builder import Model
from nengo.cache import get_default_decoder_cache
from .annotations import Annotations


class Simulator(object):
    """SpiNNaker Simulator for Nengo Models."""

    def __init__(self, network, dt=0.001, seed=None, model=None):
        """Initialise the simulator with a network (and optionally, a model).

        Parameters
        ----------
        network : :py:class:`~nengo.Network`
            Nengo network to simulate.
        dt : float
            Length of a simulation timestep in seconds.
        """
        dt = float(dt)
        raise NotImplementedError
