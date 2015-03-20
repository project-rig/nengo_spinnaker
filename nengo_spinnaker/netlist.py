"""Higher and lower level netlist items.
"""
import collections
import enum
import rig.netlist

from . import params


class OutputPort(enum.Enum):
    """Indicate the intended transmitting part of an executable."""
    standard = 0  # Standard, value based transmission

    # Ensembles only
    neurons = 1  # Transmits spike data


class InputPort(enum.Enum):
    """Indicate the intended receiving part of an executable."""
    standard = 0  # Standard, value based transmission

    # Ensembles only
    neurons = 1  # Receives spike data
    global_inhibition = 2  # Receives value-encoded inhibition data


NetAddress = collections.namedtuple("NetAddress", "object port")
"""Source or sink of a stream of packets.

Predominantly used in intermediate representations.

Parameters
----------
object : :py:class:`.IntermediateObject`
port : :py:class:`.OutputPort` or :py:class:`.InputPort`
"""


class Net(rig.netlist.Net):
    """A net represents connectivity from one vertex (or vertex slice) to many
    vertices and vertex slices.

    .. note::
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
    instances of an application running on a SpiNNaker machine).
    """
    n_atoms = params.IntParam(allow_none=True, min=0, default=None)


class VertexSlice(collections.namedtuple("VertexSlice", "vertex slice")):
    """Partition of a Vertex such that it will fit within the constraints of a
    single SpiNNaker application core.

    The intent of a vertex slice is to present a view onto an existing
    :py:class:`.Vertex` object that can be used to represent the partitioning
    of that vertex across many SpiNNaker application cores.  A netlist as
    presented to the Rig place and tools may contain a mix of
    :py:class:`.Vertex` and :py:class:`.VertexSlice` instances as demanded by
    the model.

    Attributes
    ----------
    vertex : :py:class:`.Vertex`
        Object this is a slice of.
    slice : :py:class:`slice`
        Contiguous slice of this object.
    """
    def __new__(cls, vertex, vertex_slice):
        """Create a new slice representation of a vertex.

        Parameters
        ----------
        vertex : :py:class:`.Vertex`
            Vertex to create the slice of.
        vertex_slice : :py:class:`slice`
            A contiguous (non-strided) and absolute (non-relative) slice.

        Returns
        -------
        :py:class:`.VertexSlice`
        """
        # Check the validity of the slice
        if vertex.n_atoms is None:
            # The vertex can't be sliced at all
            raise ValueError(
                "{}: cannot be represented by a slice".format(vertex)
            )
        if (not isinstance(vertex_slice, slice) or
                vertex_slice.step not in (None, 1) or
                vertex_slice.start < 0 or vertex_slice.stop < 0 or
                vertex_slice.stop < vertex_slice.start):
            raise ValueError(
                "vertex_slice: must be contiguous and non-relative, "
                "{!s} was not valid.".format(vertex_slice)
            )
        if vertex_slice.stop > vertex.n_atoms:
            raise ValueError(
                "slice {} beyond range of vertex {}".format(
                    vertex_slice, vertex)
            )
        return super(cls, VertexSlice).__new__(cls, vertex, vertex_slice)

    def __repr__(self):
        return "<VertexSlice {!s}[{}:{}]>".format(self.vertex,
                                                  self.slice.start,
                                                  self.slice.stop)
