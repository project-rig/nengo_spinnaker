import math
import mock
import nengo
import pytest
import struct
import tempfile

from nengo_spinnaker.regions.filters import (
    FilterRegion, FilterRoutingRegion, LowpassFilter, NoneFilter,
    make_filter_regions)
from nengo_spinnaker.utils.keyspaces import KeyspaceContainer
from nengo_spinnaker.utils import type_casts as tp


class TestNoneFilter(object):
    """Test creating and writing out None filters."""
    @pytest.mark.parametrize("width, latching", [(5, True), (10, False)])
    def test_standard(self, width, latching):
        nf = NoneFilter(width, latching)

        # Pack the filter into some data
        data = bytearray(nf.size)
        nf.pack_into(0.001, data)

        # Check the values are sane
        v1, v2, mask, size = struct.unpack("<4I", data)
        assert v1 == 0
        assert v2 == tp.value_to_fix(1.0)
        assert mask == (0xffffffff if latching else 0x00000000)
        assert size == width

    @pytest.mark.parametrize("width, latching", [(5, True), (10, False)])
    def test_from_signal_and_connection(self, latching, width):
        # Create the mock signal and connection
        signal = mock.Mock(name="signal", spec_set=["latching"])
        signal.latching = latching

        connection = mock.Mock(name="connection",
                               spec_set=["size_out", "synapse"])
        connection.size_out = width
        connection.synapse = None

        # Build the filter
        nf = NoneFilter.from_signal_and_connection(signal, connection)
        assert NoneFilter(width, latching) == nf


class TestLowpassFilter(object):
    @pytest.mark.parametrize("width, latching, dt, tc",
                             [(3, False, 0.001, 0.01), (1, True, 0.002, 0.2)])
    def test_standard(self, width, latching, dt, tc):
        """Test creating and writing out lowpass filters."""
        lpf = LowpassFilter(width, latching, tc)

        # Pack the filter into some data
        data = bytearray(lpf.size)
        lpf.pack_into(dt, data)

        # Check the values are sane
        v1, v2, mask, size = struct.unpack("<4I", data)
        val = math.exp(-dt / tc)
        assert v1 == tp.value_to_fix(val)
        assert v2 == tp.value_to_fix(1 - val)
        assert mask == (0xffffffff if latching else 0x00000000)
        assert size == width

    @pytest.mark.parametrize("width, latching, tc",
                             [(3, False, 0.01), (1, True, 0.2)])
    def test_from_signal_and_connection(self, width, latching, tc):
        # Create the mock signal and connection
        signal = mock.Mock(name="signal", spec_set=["latching"])
        signal.latching = latching

        connection = mock.Mock(name="connection",
                               spec_set=["size_out", "synapse"])
        connection.size_out = width
        connection.synapse = mock.Mock(spec=nengo.Lowpass)
        connection.synapse.tau = tc

        # Create the filter
        lpf = LowpassFilter.from_signal_and_connection(signal, connection)
        assert lpf == LowpassFilter(width, latching, tc)

    def test_eq(self):
        lpfs = [LowpassFilter(5, True, 0.01),
                LowpassFilter(5, True, 0.02),
                LowpassFilter(3, True, 0.01),
                LowpassFilter(5, False, 0.01)]

        for lpf in lpfs[1:]:
            assert lpf != lpfs[0]

        lpf = LowpassFilter(5, True, 0.01)
        assert lpf == lpfs[0]

        nf = NoneFilter(5, True)
        assert lpf != nf and nf != lpf


def test_filter_region():
    """Test creation of a filter data region."""
    # Create two filters
    fs = [LowpassFilter(5, False, 0.1),
          NoneFilter(3, False)]

    # Create the filter region
    fr = FilterRegion(fs, dt=0.001)

    # The size should be correct
    assert fr.sizeof() == 4 + 16 * len(fs)

    # Check that the data is written out correctly
    fp = tempfile.TemporaryFile()
    fr.write_subregion_to_file(fp)
    fp.seek(0)

    length, = struct.unpack("<I", fp.read(4))
    assert length == len(fs)

    expected_data = bytearray(32)
    fs[0].pack_into(fr.dt, expected_data)
    fs[1].pack_into(fr.dt, expected_data, 16)
    assert fp.read() == expected_data


def test_filter_routing_region():
    """Test creation of a filter routing region."""
    # Define some keyspaces
    ksc = KeyspaceContainer()
    ks_a = ksc["nengo"](object=0, connection=3)
    ks_b = ksc["nengo"](object=127, connection=255, cluster=63, index=15)
    ksc.assign_fields()

    # Define the filter routes, these map a keyspace to an integer
    ks_routes = [(ks_a, 12), (ks_b, 17)]

    # Create the region
    filter_region = FilterRoutingRegion(
        ks_routes, filter_routing_tag=ksc.filter_routing_tag,
        index_field="index"
    )

    # Check that the memory requirement is sane
    assert filter_region.sizeof() == 4 * (1 + 4*len(ks_routes))

    # Check that the written out data is sensible
    fp = tempfile.TemporaryFile()
    filter_region.write_subregion_to_file(fp)

    # Check that the data is sensible
    fp.seek(0)
    length, = struct.unpack("<I", fp.read(4))
    assert length == 2

    # Determine valid values
    valid_mask = ksc["nengo"].get_mask(tag=ksc.filter_routing_tag)
    valid_d_mask = ksc["nengo"].get_mask(field="index")

    for _ in range(len(ks_routes)):
        key, mask, i, d_mask = struct.unpack("<4I", fp.read(16))
        assert mask == valid_mask, hex(mask) + " != " + hex(valid_mask)
        assert d_mask == valid_d_mask

        for ks, j in ks_routes:
            if key == ks.get_value(tag=ksc.filter_routing_tag):
                assert i == j
                break
        else:
            assert False, "Unexpected key " + hex(key)


class TestMakeFilterRegions(object):
    """Test the helper for constructing these regions."""
    @pytest.mark.parametrize("minimise", [True, False])
    def test_equivalent_filters(self, minimise):
        """Test construction of filter regions from signals and keyspaces."""
        # Create two keyspaces, two signals and two connections with equivalent
        # synapses.
        ks_a = mock.Mock(name="Keyspace[A]")
        signal_a = mock.Mock(name="Signal[A]")
        signal_a.keyspace, signal_a.latching = ks_a, False

        ks_b = mock.Mock(name="Keyspace[B]")
        signal_b = mock.Mock(name="Signal[B]")
        signal_b.keyspace, signal_b.latching = ks_b, False

        conn_a = mock.Mock(name="Connection[A]")
        conn_a.size_out = 3
        conn_a.synapse = nengo.Lowpass(0.01)

        conn_b = mock.Mock(name="Connection[B]")
        conn_b.size_out = 3
        conn_b.synapse = nengo.Lowpass(0.01)

        # Create the type of dictionary that is expected as input
        signals_connections = {
            signal_a: [conn_a],
            signal_b: [conn_b],
        }

        # Create the regions, with minimisation
        filter_region, routing_region = make_filter_regions(
            signals_connections, 0.001,
            minimise=minimise,
            filter_routing_tag="spam",
            index_field="eggs"
        )

        # Check that the filter region is as expected
        assert filter_region.dt == 0.001

        if minimise:
            assert len(filter_region.filters) == 1
            assert filter_region.filters[0] == LowpassFilter(3, False, 0.01)
        else:
            assert len(filter_region.filters) == 2
            assert filter_region.filters[0] == LowpassFilter(3, False, 0.01)
            assert filter_region.filters[1] == LowpassFilter(3, False, 0.01)

        # Check that the routing region is as expected
        assert routing_region.filter_routing_tag == "spam"
        assert routing_region.index_field == "eggs"
        if minimise:
            assert (ks_a, 0) in routing_region.keyspace_routes
            assert (ks_b, 0) in routing_region.keyspace_routes
        else:
            if (ks_a, 0) in routing_region.keyspace_routes:
                assert (ks_b, 1) in routing_region.keyspace_routes
            else:
                assert (ks_b, 0) in routing_region.keyspace_routes

    def test_different_filters(self):
        """Test construction of filter regions from signals and keyspaces."""
        # Create two keyspaces, two signals and two connections with equivalent
        # synapses.
        ks_a = mock.Mock(name="Keyspace[A]")
        signal_a = mock.Mock(name="Signal[A]")
        signal_a.keyspace, signal_a.latching = ks_a, False

        ks_b = mock.Mock(name="Keyspace[B]")
        signal_b = mock.Mock(name="Signal[B]")
        signal_b.keyspace, signal_b.latching = ks_b, False

        conn_a = mock.Mock(name="Connection[A]")
        conn_a.size_out = 3
        conn_a.synapse = nengo.Lowpass(0.01)

        conn_b = mock.Mock(name="Connection[B]")
        conn_b.size_out = 3
        conn_b.synapse = None

        # Create the type of dictionary that is expected as input
        signals_connections = {
            signal_a: [conn_a],
            signal_b: [conn_b],
        }

        # Create the regions, with minimisation
        filter_region, routing_region = make_filter_regions(
            signals_connections, 0.001,
            minimise=True,  # Shouldn't achieve anything
            filter_routing_tag="spam",
            index_field="eggs"
        )

        # Check that the filter region is as expected
        assert filter_region.dt == 0.001
        assert len(filter_region.filters) == 2

        for f in filter_region.filters:
            assert (f == LowpassFilter(3, False, 0.01) or
                    f == NoneFilter(3, False))  # noqa: E711

        # Check that the routing region is as expected
        assert routing_region.filter_routing_tag == "spam"

        assert routing_region.index_field == "eggs"
        if (ks_a, 0) in routing_region.keyspace_routes:
            assert (ks_b, 1) in routing_region.keyspace_routes
        else:
            assert (ks_b, 0) in routing_region.keyspace_routes
