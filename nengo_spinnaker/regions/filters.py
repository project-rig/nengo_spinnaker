import nengo.synapses
import numpy as np
from six import iteritems
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
    """Region of memory which contains filter parameters.

    The first word of the region is a count of the number of filters in the
    region.  Subsequent values are always a concatenation of the four values
    with whatever additional parameters are required to simulate the filter.
    These words are:

    * The number of words that follow this set of parameters
    * Index into the array of filter types in C.
    * Number of dimensions in the filtered values ("width")
    * Flags related to the simulation of the filter.

    Subsequent words are filter type-specific.
    """

    supported_filter_types = registerabledict()
    """Dictionary mapping synapse type to a supported type of filter."""

    def __init__(self, filters, dt):
        """Create a new filter region."""
        self.filters = filters
        self.dt = dt

    def sizeof(self, *args):
        """Get the size of the filter region in bytes."""
        # 1 word + the size of the each filter
        words = (1 + 4*len(self.filters) +
                 sum(f.size_words() for f in self.filters))
        return words * 4

    def write_subregion_to_file(self, fp, *args, **kwargs):
        """Write the region to a file-like object."""
        # Create a buffer for the region to write into, write in the first word
        # and then write in each region in turn.
        data = bytearray(self.sizeof())
        struct.pack_into("<I", data, 0, len(self.filters))

        # Write in each region
        offset = 1
        for f in self.filters:
            f.pack_into(self.dt, data, offset*4)
            offset += f.size_words() + 4

        # Write the data block to file
        fp.write(data)


class Filter(object):
    def __init__(self, width, latching):
        self.width = width
        self.latching = latching

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.width == other.width and
                self.latching == other.latching)

    def size_words(self):
        """Get the number of words used to store the parameters for this
        filter.
        """
        raise NotImplementedError

    def pack_into(self, dt, buffer, offset=0):
        """Pack the header struct describing the filter into the buffer."""
        # Pack the header
        struct.pack_into("<4I", buffer, offset,
                         self.size_words(),
                         self.method_index(),
                         self.width,
                         0x1 if self.latching else 0x0)

        # Pack any data
        self.pack_data(dt, buffer, offset + 16)


@FilterRegion.supported_filter_types.register(type(None))
class NoneFilter(Filter):
    """Represents a filter which does nothing."""
    @classmethod
    def from_signal_and_connection(cls, signal, connection, width=None):
        if width is None:
            width = connection.post_obj.size_in
        return cls(width, signal.latching)

    def method_index(self):
        """Get the index into the array of filter functions."""
        return 0

    def size_words(self):
        return 0

    def pack_data(self, dt, buffer, offset=0):
        """Pack the struct describing the filter into the buffer."""
        # None filter, so no data
        pass


@FilterRegion.supported_filter_types.register(nengo.synapses.Lowpass)
class LowpassFilter(Filter):
    """Represents a Lowpass filter."""
    def __init__(self, width, latching, time_constant):
        """Create a new Lowpass filter."""
        super(LowpassFilter, self).__init__(width, latching)
        self.time_constant = time_constant

    @classmethod
    def from_signal_and_connection(cls, signal, connection, width=None):
        if width is None:
            width = connection.post_obj.size_in
        return cls(width, signal.latching, connection.synapse.tau)

    def method_index(self):
        """Get the index into the array of filter functions."""
        return 1

    def size_words(self):
        """Number of words required to represent the filter parameters."""
        return 2

    def __eq__(self, other):
        return (super(LowpassFilter, self).__eq__(other) and
                self.time_constant == other.time_constant)

    def pack_data(self, dt, buffer, offset=0):
        """Pack the struct describing the filter into the buffer."""
        # Compute the coefficients
        a = np.exp(-dt / self.time_constant)
        b = 1.0 - a
        struct.pack_into("<2I", buffer, offset,
                         tp.value_to_fix(a), tp.value_to_fix(b))


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
                             ks.get_mask(field=self.index_field),
                             index)

        # Write to file
        fp.write(data)
