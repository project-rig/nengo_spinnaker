class ValueSink(object):
    """Operator which receives and stores values across the SpiNNaker multicast
    network.

    Attributes
    ----------
    size_in : int
        Number of packets to receive and store per timestep.
    sample_every : int
        Number of machine timesteps between taking samples.
    """
    def __init__(self, size_in, sample_every=1):
        self.size_in = size_in
        self.sample_every = sample_every
