import mock
import nengo
import numpy as np
import pytest
from rig.type_casts import float_to_fix
import struct

from nengo_spinnaker.annotations import Annotations
from nengo_spinnaker import annotations as anns
from nengo_spinnaker.keyspaces import keyspaces
from nengo_spinnaker import filters


default_ks = keyspaces["nengo"]
bitsk = float_to_fix(True, 32, 15)  # S16.15


class TestFilterRegion(object):
    """Test the construction and writing of regions representing blocks of
    filters and there associated routing information.
    """
    @pytest.mark.parametrize("n_filters, n_routes", [(5, 7), (1, 3), (9, 10)])
    @pytest.mark.parametrize("vertex_slice", [slice(0, 2), slice(1, 5)])
    def test_sizeof(self, n_filters, n_routes, vertex_slice):
        dt = 0.001

        # Generate some filters and routes
        fs = [mock.Mock(name="filter", spec_set=[]) for _ in range(n_filters)]
        routes = [(mock.Mock(name="keyspace", spec_set=[]), 3)
                  for _ in range(n_routes)]

        # Construct the region and assert that the size is reported correctly
        region = filters.FilterRegion(dt, fs, routes)

        HEADER_LENGTH = 2  # WORDS
        assert (region.sizeof(vertex_slice) ==
                (HEADER_LENGTH + 4*n_routes + 4*n_filters) * 4)

        # Check that it reports the number of filters correctly
        assert region.n_filters == n_filters

    def test_write_subregion_to_file(self):
        # Generate some filters and some routes
        fs = [
            filters.LowPassFilter(0.3, 5, False),
            filters.LowPassFilter(0.05, 2, True)
        ]

        mock_ks1 = mock.Mock(name="keyspace",
                             spec_set=["get_mask", "get_value"])
        mock_ks1.get_mask.side_effect = lambda tag: (
            0xffff0000 if tag == "filter_routing" else 0x000000ff)
        mock_ks1.get_value.return_value = 0xaaaa0000

        mock_ks2 = mock.Mock(name="keyspace",
                             spec_set=["get_mask", "get_value"])
        mock_ks2.get_mask.side_effect = lambda tag: (
            0xfff00000 if tag == "filter_routing" else 0x0000007f)
        mock_ks2.get_value.return_value = 0xbbb00000

        filter_routes = [(mock_ks1, 0), (mock_ks2, 1)]

        # Create the region
        dt = 0.003
        region = filters.FilterRegion(dt, fs, filter_routes)

        # Write out to a mock file
        fp = mock.Mock(name="file", spec_set=["write"])
        region.write_subregion_to_file(slice(None), fp)

        # Assert that the keyspaces were called correctly
        for ks in [mock_ks1, mock_ks2]:
            assert ks.get_mask.call_count == 2
            ks.get_mask.assert_any_call(tag="filter_routing")
            ks.get_mask.assert_any_call(tag="dimension")

            ks.get_value.assert_called_once_with(tag="filter_routing")

        # Now check that we can unpack the bytestring that was written
        data = fp.write.call_args[0][0]
        assert isinstance(data, bytes)
        assert len(data) == region.sizeof(slice(None))

        # Unpack the header struct
        (routing_offset, filters_offset, n_routes, n_filters) = \
            struct.unpack("<4H", data[:8])
        assert routing_offset == 8
        assert filters_offset == 8 + n_routes * 16
        assert n_routes == 2
        assert n_filters == 2

        # Proceed to unpack the routing entries
        valid_entries = [
            struct.pack("<4I", 0xaaaa0000, 0xffff0000, 0, 0x000000ff),
            struct.pack("<4I", 0xbbb00000, 0xfff00000, 1, 0x0000007f)
        ]
        routing_entries = data[routing_offset:routing_offset + n_routes*16]
        assert routing_entries[:16] in valid_entries
        assert routing_entries[16:] in valid_entries
        assert routing_entries[:16] != routing_entries[16:]

        # Unpack the filter entries
        filter_entries = data[filters_offset:]
        assert filter_entries[:16] == fs[0].pack(dt)
        assert filter_entries[16:] == fs[1].pack(dt)

    def test_from_intermediate_representation_with_minimization(self):
        """Test construction of filter regions from Nengo connections and nets
        extracted from an intermediate representation.
        """
        net = nengo.Network()
        with net:
            a = nengo.Ensemble(100, 2)
            b = nengo.Ensemble(100, 2)
            c = nengo.Ensemble(100, 2)

            a_c = nengo.Connection(a, c)
            b_c = nengo.Connection(b, c)

        # Construct the intermediate representation
        model = mock.Mock()
        model.params = {a: None, b: None, c: None, a_c: None, b_c: None}
        irn = Annotations.from_model(model)
        irn.apply_default_keyspace(default_ks)

        # Build the filter region for c
        region, net_map = \
            filters.FilterRegion.from_intermediate_representation(
                0.001,
                irn.get_nets_ending_at(irn.objects[c])[
                    anns.InputPort.standard], 3, minimize=True
            )

        # Should have 1 filter and 2 routes
        assert len(region.filters) == 1
        assert region.filters[0] == filters.LowPassFilter(a_c.synapse.tau, 3)

        for val in region.keyspace_maps:
            assert val in [(irn.connections[x].keyspace, 0) for x in
                           [a_c, b_c]]

    def test_from_intermediate_representation_with_specified_widths(self):
        """Test construction of filter regions from Nengo connections and nets
        extracted from an intermediate representation.
        """
        net = nengo.Network()
        with net:
            a = nengo.Ensemble(100, 2)
            b = nengo.Ensemble(100, 2)
            c = nengo.Ensemble(100, 2)

            a_c = nengo.Connection(a, c)
            b_c = nengo.Connection(b, c)

        # Construct the intermediate representation
        model = mock.Mock()
        model.params = {a: None, b: None, c: None, a_c: None, b_c: None}
        irn = Annotations.from_model(model)
        irn.apply_default_keyspace(default_ks)

        # Build the filter region for c
        widths = {irn.connections[a_c]: 4, irn.connections[b_c]: 5}
        region, net_map = \
            filters.FilterRegion.from_intermediate_representation(
                0.001,
                irn.get_nets_ending_at(irn.objects[c])[
                    anns.InputPort.standard],
                widths, minimize=True
            )

        # Should have 2 filters and 2 routes
        assert len(region.filters) == 2

        for net in (irn.connections[x] for x in [a_c, b_c]):
            for ks, x in region.keyspace_maps:
                if ks is net.keyspace:
                    assert x == net_map[net]
            assert region.filters[net_map[net]].width == widths[net]

    def test_from_intermediate_representation_unknown_filter_type(self):
        net = anns.AnnotatedNet(None, None)
        conn = mock.Mock(name="Connection", spec_set=["synapse"])

        with pytest.raises(TypeError) as excinfo:
            filters.FilterRegion.from_intermediate_representation(
                0.001, {net: [conn]}, 5)
        assert conn.synapse.__class__.__name__ in str(excinfo.value)


class TestFilter(object):
    def test_hashing_and_eq(self):
        f1 = filters.Filter(4, True)
        f2 = filters.Filter(5, True)

        assert hash(f1) != hash(f2)
        assert f1 != f2

        f3 = filters.Filter(5, True)
        assert hash(f2) == hash(f3)
        assert f2 == f3

        f4 = filters.Filter(5, False)
        assert hash(f2) != hash(f4)
        assert f2 != f4

    @pytest.mark.parametrize("width, latching", [(3, False), (5, True)])
    def test_to_struct(self, width, latching):
        f = filters.Filter(width, latching)
        assert (struct.unpack("<2I", f.pack(0.001)) ==
                (width, 0xffffffff if latching else 0x00000000))


class TestLowPassFilter(object):
    def test_hashing_and_eq(self):
        f1 = filters.LowPassFilter(0.5, 3, False)
        f2 = filters.LowPassFilter(0.3, 3, False)
        assert hash(f1) != hash(f2)
        assert f1 != f2

        f3 = filters.LowPassFilter(0.3, 2, False)
        assert hash(f2) != hash(f3)
        assert f2 != f3

        f4 = filters.LowPassFilter(0.3, 2, True)
        assert hash(f3) != hash(f4)
        assert f3 != f4

        f5 = filters.Filter(2, True)
        assert hash(f4) != hash(f5)
        assert f4 != f5

        f6 = filters.LowPassFilter(0.3, 2, True)
        assert hash(f4) == hash(f6)
        assert f4 == f6

    @pytest.mark.parametrize("tau", [0.04, 0.1])
    @pytest.mark.parametrize("width", [4, 1])
    @pytest.mark.parametrize("latching", [True, False])
    def test_from_intermediate_representation(self, tau, width, latching):
        with nengo.Network():
            a = nengo.Ensemble(100, 2)
            b = nengo.Ensemble(100, 2)
            c = nengo.Connection(a, b, synapse=tau)

        n = anns.AnnotatedNet(None, None, latching=latching)

        # Create the low pass filter
        f = filters.LowPassFilter.from_intermediate_representation(
            n, [c], width)
        assert f.time_constant == tau
        assert f.latching == latching
        assert f.width == width

    @pytest.mark.parametrize("dt, tc, width, latching",
                             [(0.001, 0.0, 3, False), (0.01, 0.5, 5, True)])
    def test_pack(self, dt, tc, width, latching):
        f = filters.LowPassFilter(tc, width, latching)
        tc_, tc__, width_, mask = struct.unpack("<4I", f.pack(dt))
        assert width_ == width
        assert mask == (0xffffffff if latching else 0x00000000)
        assert tc_ == 0 if tc == 0 else bitsk(np.exp(-dt / tc))
        assert tc__ == (bitsk(1) if tc == 0 else bitsk(1.0 - np.exp(-dt / tc)))
