import numpy as np


class EnsembleLIF(object):
    """Controller for an ensemble of LIF neurons."""
    def __init__(self, size_in):
        """Create a new LIF ensemble controller."""
        self.direct_input = np.zeros(size_in)
        self.local_probes = list()
