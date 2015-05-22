import math
import nengo.synapses
from six import iteritems, itervalues
import struct

from .region import Region
from nengo_spinnaker.utils.collections import registerabledict
from nengo_spinnaker.utils import type_casts as tp


def make_filter_regions(signals_and_connections, dt, minimise=False,
                        filter_routing_tag="filter_routing",
                        index_field="index", width=None):
    """Create a filter region and a filter routing region from the given
    signals and connections.

    Parameters
    ----------
    signals_and_connections : {Signal: [Connection, ...], ...}
        Map of signals to the connections they represent.
    dt : float
        Simulation timestep.
    width : int
        Force all filters to be a given width.

    Other Parameters
    ----------------
    minimise : bool
        It is possible to reduce the amount of memory and computation required
        to simulate filters by combining equivalent filters together.  If
        minimise is `True` then this is done, otherwise not.
    """
    # Build the set of filters and the routing entries
    filters = list()
    keyspace_routes = list()

    for signal, connections in iteritems(signals_and_connections):
        for connection in connections:
            # Make the filter
            f = FilterRegion.supported_filter_types[type(connection.synapse)].\
                from_signal_and_connection(signal, connection, width=width)

            # Store the filter and add the route
            for index, g in enumerate(filters):
                if f == g and minimise:
                    break
            else:
                index = len(filters)
                filters.append(f)

            keyspace_routes.append((signal.keyspace, index))

    # Create the regions
    filter_region = FilterRegion(filters, dt)
    routing_region = FilterRoutingRegion(keyspace_routes, filter_routing_tag,
                                         index_field)
    return filter_region, routing_region


class FilterRegion(Region):
    """Region of memory which contains filter parameters."""

    supported_filter_types = registerabledict()
    """Dictionary mapping synapse type to a supported type of filter."""

    def __init__(self, filters, dt):
        """Create a new filter region."""
        self.filters = filters
        self.dt = dt

        # Get the size of the largest supported filter
        self._largest_struct = max(
            f.size for f in itervalues(self.supported_filter_types))

    def sizeof(self, *args):
        """Get the size of the filter region in bytes."""
        # 1 word + the size of the largest supported filter * number of filters
        return 4 + self._largest_struct * len(self.filters)

    def write_subregion_to_file(self, fp, *args, **kwargs):
        """Write the region to a file-like object."""
        data = bytearray(self.sizeof())

        # Write the number of elements
        struct.pack_into("<I", data, 0, len(self.filters))

        # Pack each element in, allowing enough space for the largest element
        # in the union.
        for i, f in enumerate(self.filters):
            f.pack_into(self.dt, data, 4 + i*self._largest_struct)

        # Write the buffer back
        fp.write(data)


class Filter(object):
    _pack_chars = "<2I"
    size = struct.calcsize(_pack_chars)

    def __init__(self, width, latching):
        self.width = width
        self.latching = latching

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.width == other.width and
                self.latching == other.latching)

    def pack_into(self, dt, buffer, offset=0):
        """Pack the struct describing the filter into the buffer."""
        struct.pack_into(self._pack_chars, buffer, offset,
                         0xffffffff if self.latching else 0x00000000,
                         self.width)


@FilterRegion.supported_filter_types.register(nengo.synapses.Lowpass)
class LowpassFilter(Filter):
    """Represents a Lowpass filter."""
    _pack_chars = "<2I"
    size = struct.calcsize(_pack_chars) + struct.calcsize(Filter._pack_chars)

    def __init__(self, width, latching, time_constant):
        """Create a new Lowpass filter."""
        super(LowpassFilter, self).__init__(width, latching)
        self.time_constant = time_constant

    @classmethod
    def from_signal_and_connection(cls, signal, connection, width=None):
        if width is None:
            width = connection.post_obj.size_in
        return cls(width, signal.latching, connection.synapse.tau)

    def __eq__(self, other):
        return (super(LowpassFilter, self).__eq__(other) and
                self.time_constant == other.time_constant)

    def pack_into(self, dt, buffer, offset=0):
        """Pack the struct describing the filter into the buffer."""
        val = math.exp(-dt / self.time_constant)
        struct.pack_into(self._pack_chars, buffer, offset,
                         tp.value_to_fix(val),
                         tp.value_to_fix(1 - val))
        super(LowpassFilter, self).pack_into(
            dt, buffer, offset + struct.calcsize(self._pack_chars))


@FilterRegion.supported_filter_types.register(type(None))
class NoneFilter(Filter):
    """Represents a filter which does nothing."""
    _pack_chars = "<2I"
    size = struct.calcsize(_pack_chars) + struct.calcsize(Filter._pack_chars)

    @classmethod
    def from_signal_and_connection(cls, signal, connection, width=None):
        if width is None:
            width = connection.post_obj.size_in
        return cls(width, signal.latching)

    def pack_into(self, dt, buffer, offset=0):
        """Pack the struct describing the filter into the buffer."""
        struct.pack_into(self._pack_chars, buffer, offset,
                         tp.value_to_fix(0.0), tp.value_to_fix(1.0))
        super(NoneFilter, self).pack_into(
            dt, buffer, offset + struct.calcsize(self._pack_chars))


class FilterRoutingRegion(Region):
    """Region of memory which maps routing entries to filter indices.

    Attributes
    ----------
    keyspace_routes : [(BitField, int), ...]
        Pairs of BitFields (keyspaces) to the index of the filter that packets
        matching the entry should be routed.
    """

    def __init__(self, keyspace_routes, filter_routing_tag="filter_routing",
                 index_field="index"):
        """Create a new routing region."""
        self.keyspace_routes = keyspace_routes
        self.filter_routing_tag = filter_routing_tag
        self.index_field = index_field

    def sizeof(self, *args):
        """Get the memory requirements of this region as a number of bytes."""
        # 1 word + 4 words per entry
        return 4 * (1 + 4*len(self.keyspace_routes))

    def write_subregion_to_file(self, fp, *args, **kwargs):
        """Write the routing region to a file-like object."""
        data = bytearray(self.sizeof())

        # Write the number of entries
        struct.pack_into("<I", data, 0, len(self.keyspace_routes))

        # Write each entry in turn
        for i, (ks, index) in enumerate(self.keyspace_routes):
            struct.pack_into("<4I", data, 4 + 16*i,
                             ks.get_value(tag=self.filter_routing_tag),
                             ks.get_mask(tag=self.filter_routing_tag),
                             index,
                             ks.get_mask(field=self.index_field))

        # Write to file
        fp.write(data)
