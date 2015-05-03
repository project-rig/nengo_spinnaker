"""Higher and lower level netlist items.
"""
import rig.netlist


class Net(rig.netlist.Net):
    """A net represents connectivity from one vertex (or vertex slice) to many
    vertices and vertex slices.

    ..note::
        This extends the Rig :py:class:`~rig.netlist.Netlist` to add Nengo
        specific attributes and terms.

    Attributes
    ----------
    source : :py:class:`.Vertex` or :py:class:`.VertexSlice`
        Vertex or vertex slice which is the source of the net.
    sinks : [:py:class:`.Vertex` or :py:class:`.VertexSlice`, ...]
        List of vertices and vertex slices which are the sinks of the net.
    weight : int
        Number of packets transmitted across the net every simulation
        time-step.
    keyspace : :py:class:`rig.bitfield.BitField`
        32-bit bitfield instance that can be used to derive the routing key and
        mask for the net.
    """
    __slots__ = ['keyspace']  # Only add keyspace to the list of slots

    def __init__(self, source, sinks, weight, keyspace):
        """Create a new net.

        See :py:meth:`~rig.netlist.Net.__init__`.

        Parameters
        ----------
        keyspace : :py:class:`rig.bitfield.BitField`
            32-bit bitfield instance that can be used to derive the routing key
            and mask for the net.
        """
        # Assert that the keyspace is 32-bits long
        if keyspace.length != 32:
            raise ValueError(
                "keyspace: Must be 32-bits long, not {}".format(
                    keyspace.length)
            )
        super(Net, self).__init__(source, sinks, weight)
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
    constraints : [constraint, ...]
        The :py:mod:`~rig.place_and_route.constraints` which should be applied
        to the placement and routing related to the vertex.
    resource : {resource: usage, ...}
        Mapping from resource type to the consumption of that resource, in
        whatever is an appropriate unit.
    cluster : int or None
        Index of the cluster the vertex is a part of.
    """
    __slots__ = ["application", "constraints", "resources", "cluster"]

    def __init__(self, application=None, resources=dict(), constraints=list()):
        """Create a new Vertex.
        """
        self.application = application
        self.constraints = list(constraints)
        self.resources = dict(resources)
        self.cluster = None


class VertexSlice(Vertex):
    """Represents a portion of a nominal unit of computation.

    Attributes
    ----------
    application : str or None
        Path to application which should be loaded onto SpiNNaker to simulate
        this vertex, or None if no application is required.
    constraints : [constraint, ...]
        The :py:mod:`~rig.place_and_route.constraints` which should be applied
        to the placement and routing related to the vertex.
    resource : {resource: usage, ...}
        Mapping from resource type to the consumption of that resource, in
        whatever is an appropriate unit.
    slice : :py:class:`slice`
        Slice of the unit of computation which is represented by this vertex
        slice.
    """
    __slots__ = ["slice"]

    def __init__(self, slice, application=None, resources=dict(),
                 constraints=list()):
        super(VertexSlice, self).__init__(application, resources, constraints)
        self.slice = slice
