class NMNet(object):
    """A net represents connectivity from many to many vertices.

    Attributes
    ----------
    sources : [object, ...]
    sinks : [object, ...]
    weight : int
        Number of packets transmitted across the net every simulation
        time-step.
    keyspace : :py:class:`rig.bitfield.BitField`
        32-bit bitfield instance that can be used to derive the routing key and
        mask for the net.
    """
    def __init__(self, sources, sinks, weight, keyspace):
        # Source(s) and sink(s) must be stored as lists
        if not isinstance(sources, list):
            sources = [sources]

        if not isinstance(sinks, list):
            sinks = [sinks]

        # Store all the parameters (copying source and sink lists)
        self.sources = list(sources)
        self.sinks = list(sinks)
        self.weight = weight
        self.keyspace = keyspace


class Vertex(object):
    """Represents a nominal unit of computation (a single instance or many
    instances of an application running on a SpiNNaker machine) or an external
    device that is connected to the SpiNNaker network.

    Attributes
    ----------
    application : str or None
        Path to application which should be loaded onto SpiNNaker to simulate
        this vertex, or None if no application is required.
    resource : {resource: usage, ...}
        Mapping from resource type to the consumption of that resource, in
        whatever is an appropriate unit.
    cluster : int or None
        Index of the cluster the vertex is a part of.
    """
    def __init__(self, application=None, resources=dict()):
        """Create a new Vertex.
        """
        self.application = application
        self.resources = dict(resources)
        self.cluster = None


class VertexSlice(Vertex):
    """Represents a portion of a nominal unit of computation.

    Attributes
    ----------
    application : str or None
        Path to application which should be loaded onto SpiNNaker to simulate
        this vertex, or None if no application is required.
    resource : {resource: usage, ...}
        Mapping from resource type to the consumption of that resource, in
        whatever is an appropriate unit.
    slice : :py:class:`slice`
        Slice of the unit of computation which is represented by this vertex
        slice.
    """
    def __init__(self, slice, application=None, resources=dict()):
        super(VertexSlice, self).__init__(application, resources)
        self.slice = slice
