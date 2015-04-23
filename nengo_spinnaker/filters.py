"""Representing filters, constructing regions for filters and extracting
filters from annotated Nengo connections.
"""
from rig.type_casts import float_to_fix
import nengo
import numpy as np
from six import iteritems
import struct

from .utils.collections import registerabledict


s1615 = float_to_fix(True, 32, 15)


class FilterRegion(object):
    """A region which represents the routing and parameters for a set of
    filters.

    When written out a filter region consists of three blocks of data.  The
    first block is header data for the entire region, the second consists of
    routing information for the filters and the last contains parameters for
    filters themselves.

    ..todo::
        Reference the actual structs...

    The first region is a transcription of the struct::

        struct filter_header {
            uint16_t routing_offset;  // Offset (in bytes) of the 2nd region
            uint16_t filters_offset;  // Offset (in bytes) of the 3rd region
            uint16_t n_routes;        // Number of entries in the 2nd region
            uint16_t n_filters;       // Number of entries in the 3rd region
        };

    The second region is a collection of structs::

        struct filter_route {
            // A key and mask, packets which match the key and mask should be
            // included in the input to the indicated filter.
            uint32_t key;
            uint32_t mask;

            uint32_t filter_id;  // Filter for packets matching this entry
            uint32_t dimension_mask;  // Mask to apply to get the dimension
                                      // index out of the routing key.
        };

    The third region is (currently) a collection of the structs for first-order
    low-pass filters.  This is likely to change::

        struct filter_parameters {
            value_t filter; // exp{-dt / time_constant}
            value_t filter_;  // 1 - exp{-dt / time_constant}
            uint32_t latch_mask;  // Mask applied to perform latching
            uint32_t width;  // Number of dimensions in filter
        };

        // The latch mask has two meaningful values:
        // 0x00000000 - means that the filter is non-latching (it is reset once
        //              per timestep)
        // 0xffffffff - means that the filter is latching (it is reset every
        //              time a new value is inserted)

    These values are determined from a combination of a Nengo Connection and an
    annotation (`IntermediateNet`) as represented in `Filter` objects.
    """

    filter_builders = registerabledict()
    """Callables which can construct filter objects from an IntermediateNet, a
    list of Nengo connections and a given width.
    """

    def __init__(self, dt, filters, keyspace_maps):
        """Create a new filter region.

        Parameters
        ----------
        dt : float
            Timestep the filters will be simulated with.
        filters : [:py:class:`.Filter`, ...]
            List of filter objects to contain within the region.
        keyspace_maps : [(:py:class:`rig.bitfield.BitField`, int), ...]
            Map of keyspaces to indices of the `filters` list represented as a
            list of tuples (to allow repeats and unhashable types).
        """
        self.dt = dt
        self.filters = list(filters)
        self.keyspace_maps = list(keyspace_maps)

    @classmethod
    def from_annotations(cls, dt, nets_connections, width, minimize=False):
        """Create a new filter region by extracting filters from intermediate
        representations of Nengo Connections.

        Parameters
        ----------
        nets_connections : {`AnnotatedNet`: [`nengo.Connection`, ...], ...}
            Map of annotation nets to the connections they represent.  This
            mapping can be retrieved by calling `get_nets_ending_at` on an
            `IntermediateRepresentation` and indexing with the port the filters
            should be specified for.
        width : int or {`IntermediateNet`: int, ...}
            Widths of the filters that should be produced.  If an int then the
            same width is used for all filters.  If a dict _then no
            minimisation will be performed_ and the width for each filter will
            be taken from the width associated with the net.
        minimize : bool
            It is often possible to reduce the amount of memory and computation
            required to process filters by combining equivalent filters
            together.  If True this minimisation is performed.

        Returns
        -------
        :py:class:`.FilterRegion`
            A new filter region containing filters and routing determined from
            the nets and connections.
        {`IntermediateNet`: int, ...}
            Map of which nets were mapped to which filter index.
        """
        if isinstance(width, dict):
            minimize = False

        # Build up a list of filters and keyspace maps
        filters = []
        net_maps = {}

        for net, connections in iteritems(nets_connections):
            assert all(c.synapse == connections[0].synapse
                       for c in connections)
            synapse = connections[0].synapse

            # Build the filter
            if synapse.__class__ not in cls.filter_builders:
                raise TypeError("Synapse type {} not supported.".format(
                    synapse.__class__.__name__))

            w = width if not isinstance(width, dict) else width[net]
            new_filter = cls.filter_builders[synapse.__class__](
                net, connections, w)

            if not minimize or new_filter not in filters:
                filter_key = len(filters)
                filters.append(new_filter)
            else:
                filter_key = filters.index(new_filter)

            # Add the routing entry
            net_maps[net] = filter_key

        # Build the keyspace map
        keyspace_map = [(net.keyspace, f) for (net, f) in iteritems(net_maps)]

        return cls(dt, filters, keyspace_map), net_maps

    def sizeof(self):
        """Get the amount of memory in bytes required to represent this region.

        ..note::
            This region is not partitioned - the memory requirements are
            constant.

        Returns
        -------
        int
            Memory requirement of the region in bytes.
        """
        return (2 + 4*len(self.keyspace_maps) + 4*len(self.filters)) * 4

    @property
    def n_filters(self):
        """Get the number of filters represented by the region, this can be
        used to calculate (e.g.) the amount of DTCM necessary to store filter
        state.
        """
        return len(self.filters)

    def write_subregion_to_file(self, fp, **formatter_args):
        """Write a portion of the region to a file.

        Parameters
        ----------
        fp : file-like object
            The file-like object to which data from the region will be written.
            This must support a `write` method.
        formatter_args : optional
            Arguments which will be passed to the (optional) formatter along
            with each value that is being written.
        """
        # Cache the data because writes are expensive.
        data = bytearray(self.sizeof())

        # Write the header data
        routing_offset = 8  # May change later
        n_routes = len(self.keyspace_maps)
        filters_offset = routing_offset + n_routes*16
        n_filters = self.n_filters
        struct.pack_into("<4H", data, 0, routing_offset, filters_offset,
                         n_routes, n_filters)

        # Write the routing data for each route in turn
        for i, (keyspace, f_id) in enumerate(self.keyspace_maps):
            struct.pack_into("<4I", data, i*16 + routing_offset,
                             keyspace.get_value(tag="filter_routing"),
                             keyspace.get_mask(tag="filter_routing"),
                             f_id,
                             keyspace.get_mask(field="index"))

        # Write the filter data
        offset = filters_offset
        for f in self.filters:
            filter_data = f.pack(self.dt)
            data[offset:offset + len(filter_data)] = filter_data
            offset += len(filter_data)

        # Write all the data
        fp.write(bytes(data))


class Filter(object):
    """Representation of a filter as implemented on SpiNNaker.

    In general only subclasses of this type will be instantiated.

    The `__hash__` and `__eq__` methods are supplied such that filters with
    equivalent attributes will be merged together.

    Attributes
    ----------
    latching : bool
        True if the filter input buffer should not be reset every timestep but
        should instead be cleared when a new value is received.
    width : int
        "Dimensionality" of the filter, for example `width=5` states that the
        output of the filter will be a 5-D vector.
    """
    def __init__(self, width, latching=False):
        """Create a new filter.

        Parameters
        ----------
        latching : bool
            True if the filter input buffer should not be reset every timestep
            but should instead be cleared when a new value is received.
        width : int
            "Dimensionality" of the filter, for example `width=5` states that
            the output of the filter will be a 5-D vector.
        """
        self.latching = latching
        self.width = width

    def pack(self, dt):
        """Pack a struct representation of the filter.

        Parameters
        ----------
        dt : float
            The simulation timestep that will be used to simulate the filter.

        Returns
        -------
        bytes
            Bytestring representation of the packed filter struct.
        """
        return struct.pack(
            "<2I", self.width, 0xffffffff if self.latching else 0x00000000)

    @property
    def _hash_params(self):
        return self.__class__, self.latching, self.width

    def __hash__(self):
        """Hash the attributes of the filter."""
        return hash(self._hash_params)

    def __eq__(self, other):
        """Filters are equivalent if they are of the same class and have
        equivalent parameters.
        """
        return (self.__class__ is other.__class__ and
                self.latching == other.latching and
                self.width == other.width)


class LowPassFilter(Filter):
    """A first-order low-pass filter.

    Attributes
    ----------
    time_constant : float
        Time constant of the low-pass filter.
    """
    def __init__(self, time_constant, width, latching=False):
        """Create a new low-pass filter."""
        super(LowPassFilter, self).__init__(width, latching)
        self.time_constant = time_constant

    @classmethod
    def from_annotations(cls, net, connections, width):
        """Create a new filter from inspection of a net and connections."""
        # Get the filter time-constant from the connection(s) and the latching
        # specification from the net.
        assert all(c.synapse == connections[0].synapse for c in connections)
        return cls(connections[0].synapse.tau, width, net.latching)

    def pack(self, dt):
        val = 0 if self.time_constant == 0 else np.exp(-dt/self.time_constant)
        val_ = 1.0 - val
        return (struct.pack("<2I", s1615(val), s1615(val_)) +
                super(LowPassFilter, self).pack(dt))

    def __hash__(self):
        return hash(super(LowPassFilter, self)._hash_params +
                    (self.time_constant, ))

    def __eq__(self, other):
        return (super(LowPassFilter, self).__eq__(other) and
                self.time_constant == other.time_constant)

FilterRegion.filter_builders[nengo.synapses.Lowpass] = \
    LowPassFilter.from_annotations
