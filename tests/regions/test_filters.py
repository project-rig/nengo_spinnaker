import mock
import nengo
from nengo.utils.filter_design import cont2discrete
import numpy as np
import pytest
import struct
import tempfile

from nengo_spinnaker.builder.model import (
    SignalParameters, ReceptionParameters, ReceptionSpec)
from nengo_spinnaker.regions.filters import (
    FilterRegion, FilterRoutingRegion, LowpassFilter, NoneFilter,
    LinearFilter, make_filter_regions, Filter)
from nengo_spinnaker.utils.keyspaces import KeyspaceContainer
from nengo_spinnaker.utils import type_casts as tp


@pytest.mark.parametrize("width, latching, init_index, words",
                         [(10, False, 0xDEAD, 3),
                          (11, True, 0xCAFE, 5)])
def test_filter_pack_into(width, latching, init_index, words):
    class MyFilter(Filter):
        def size_words(self):
            return words

        def method_index(self):
            return init_index

    # Create a filter instance, assert that it packs data correctly
    fil = MyFilter(width, latching)
    fil.pack_data = mock.Mock()

    # Create a buffer to pack into
    data = bytearray(100)
    fil.pack_into(0.001, data, 5)

    # Assert we were called correctly
    fil.pack_data.assert_called_once_with(0.001, data, 21)

    # Unpack, check that the values were fine
    (n_words, init_method, size, flags) = struct.unpack_from("<4I", data, 5)
    assert n_words == words
    assert init_method == init_index
    assert flags == (0x1 if latching else 0x0)
    assert size == width


class TestNoneFilter(object):
    """Test creating and writing out None filters."""
    def test_size_words(self):
        f = NoneFilter(1, False)
        assert f.size_words() == 0

    def test_pack_data(self):
        f = NoneFilter(1, False)

        # Packing into an empty array should be fine because there is nothing
        # to pack
        data = bytearray(0)
        f.pack_data(0.001, data)

    @pytest.mark.parametrize("width, latching", [(5, True), (10, False)])
    def test_from_parameters(self, latching, width):
        # Create the mock signal and connection
        signal = SignalParameters(latching=latching)
        rps = ReceptionParameters(None, width)

        # Build the filter
        nf = NoneFilter.from_parameters(signal, rps)
        assert NoneFilter(width, latching) == nf

    @pytest.mark.parametrize("width, latching", [(5, True)])
    def test_from_parameters_force_width(self, latching, width):
        # Create the mock signal and connection
        signal = SignalParameters(latching=latching)
        rps = ReceptionParameters(None, width)

        # Build the filter
        nf = NoneFilter.from_parameters(signal, rps, width=1)
        assert NoneFilter(1, latching) == nf


class TestLowpassFilter(object):
    @pytest.mark.parametrize("tau, dt", [(0.01, 0.001), (0.03, 0.002)])
    def test_pack_data(self, tau, dt):
        # Create the filter
        f = LowpassFilter(0, False, tau)

        # Pack into an array
        data = bytearray(8)
        f.pack_data(dt, data, 0)

        # Compute expected values
        exp_a = tp.value_to_fix(np.exp(-dt / tau))
        exp_b = tp.value_to_fix(1.0 - np.exp(-dt / tau))

        # Unpack and check for accuracy
        (a, b) = struct.unpack_from("<2I", data)
        assert a == exp_a
        assert b == exp_b

    @pytest.mark.parametrize("width, latching, tc",
                             [(3, False, 0.01), (1, True, 0.2)])
    def test_from_parameters(self, width, latching, tc):
        # Create the mock signal and connection
        signal = SignalParameters(latching=latching)
        rps = ReceptionParameters(nengo.Lowpass(tc), width)

        # Create the filter
        lpf = LowpassFilter.from_parameters(signal, rps)
        assert lpf == LowpassFilter(width, latching, tc)

    @pytest.mark.parametrize("width, latching, tc", [(3, False, 0.01)])
    def test_from_parameters_force_width(self, width, latching, tc):
        # Create the mock signal and connection
        signal = SignalParameters(latching=latching)
        rps = ReceptionParameters(nengo.Lowpass(tc), width)

        # Create the filter
        lpf = LowpassFilter.from_parameters(signal, rps, width=2)
        assert lpf == LowpassFilter(2, latching, tc)

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


class TestLinearFilter(object):
    @pytest.mark.parametrize(
        "num, den, dt, order",
        [([1.0], [0.3, 1.0], 0.001, 1),
         ([1.0], [0.03, 0.4, 1.0], 0.01, 2),
         ]
    )
    def test_pack_data(self, num, den, dt, order):
        # Create the filter
        lf = LinearFilter(0, False, num, den)

        # Create a buffer to pack data into
        data = bytearray((order*2 + 1)*4)

        # Pack the parameters
        lf.pack_data(dt, data, 0)

        # Generate what we expect the data to look like
        numd, dend, _ = cont2discrete((num, den), dt)
        numd = numd.flatten()
        exp = list()
        for a, b in zip(dend[1:], numd[1:]):
            exp.append(-a)
            exp.append(b)
        expected_data = tp.np_to_fix(np.array(exp)).tostring()

        # Check that's what we get
        assert struct.unpack_from("<I", data, 0)[0] == order
        assert data[4:] == expected_data

    def test_from_parameters_force_width(self):
        # Create the mock signal and connection
        signal = SignalParameters(latching=True)
        rps = ReceptionParameters(nengo.LinearFilter([1.0], [0.5, 1.0]), 1)

        # Create the filter
        lpf = LinearFilter.from_parameters(signal, rps, width=2)
        assert lpf == LinearFilter(2, True, [1.0], [0.5, 1.0])

    def test_method_index(self):
        lf = LinearFilter(0, False, [1.0], [1.0, 0.5])
        assert lf.method_index() == 2

    @pytest.mark.parametrize(
        "num, den, size_words",
        [([1.0], [1.0], 1),
         ([1.0], [0.1, 1.0], 3),
         ([1.0], [0.1, 0.1, 1.0], 5),
         ]
    )
    def test_size_words(self, num, den, size_words):
        lf = LinearFilter(0, False, num, den)
        assert lf.size_words() == size_words

    def test_eq(self):
        # Check that linear filters are equal iff. they share a numerator and
        # denominator.
        lpf = LowpassFilter(0, False, 0.1)
        lf1 = LinearFilter(0, False, [1.0], [0.1, 0.2])
        lf2 = LinearFilter(0, False, [1.0], [0.1, 0.3])
        lf3 = LinearFilter(0, False, [1.0], [0.1, 0.2])

        assert lf1 != lpf
        assert lf2 != lf1
        assert lf3 != lf2
        assert lf3 == lf1


def test_filter_region():
    """Test creation of a filter data region."""
    # Create two filters
    fs = [LowpassFilter(5, False, 0.1),
          NoneFilter(3, False)]

    # Create the filter region
    fr = FilterRegion(fs, dt=0.001)

    # The size should be correct (count of words + header 1 + data 1 + header 2
    # + data 2)
    assert fr.sizeof() == 4 + 16 + 8 + 16 + 0

    # Check that the data is written out correctly
    fp = tempfile.TemporaryFile()
    fr.write_subregion_to_file(fp)
    fp.seek(0)

    length, = struct.unpack("<I", fp.read(4))
    assert length == len(fs)

    expected_data = bytearray(fr.sizeof() - 4)
    fs[0].pack_into(fr.dt, expected_data)
    fs[1].pack_into(fr.dt, expected_data, (fs[0].size_words() + 4)*4)
    assert fp.read() == expected_data


def test_filter_routing_region():
    """Test creation of a filter routing region."""
    # Define some keyspaces
    ksc = KeyspaceContainer()
    ks_a = ksc["nengo"](connection_id=3)
    ks_b = ksc["nengo"](connection_id=255, cluster=63, index=15)
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
        key, mask, d_mask, i = struct.unpack("<4I", fp.read(16))
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
        # Create two keyspaces, two signal parameters and two reception
        # parameters with equivalent synapses.
        ks_a = mock.Mock(name="Keyspace[A]")
        signal_a = SignalParameters(keyspace=ks_a, latching=False)

        ks_b = mock.Mock(name="Keyspace[B]")
        signal_b = SignalParameters(keyspace=ks_b, latching=False)

        rp_a = ReceptionParameters(nengo.Lowpass(0.01), 3)
        rp_b = ReceptionParameters(nengo.Lowpass(0.01), 3)

        # Create the data structure that is expected as input
        specs = [
            ReceptionSpec(signal_a, rp_a),
            ReceptionSpec(signal_b, rp_b),
        ]

        # Create the regions, with minimisation
        filter_region, routing_region = make_filter_regions(
            specs, 0.001, minimise=minimise,
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
        # Create two keyspaces, two signals and two reception parameters with
        # different synapses.
        ks_a = mock.Mock(name="Keyspace[A]")
        signal_a = SignalParameters(keyspace=ks_a, latching=False)

        ks_b = mock.Mock(name="Keyspace[B]")
        signal_b = SignalParameters(keyspace=ks_b, latching=False)

        rp_a = ReceptionParameters(nengo.Lowpass(0.01), 3)
        rp_b = ReceptionParameters(None, 3)

        # Create the type of dictionary that is expected as input
        specs = [
            ReceptionSpec(signal_a, rp_a),
            ReceptionSpec(signal_b, rp_b),
        ]

        # Create the regions, with minimisation
        filter_region, routing_region = make_filter_regions(
            specs, 0.001,
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

    def test_forced_filter_width(self):
        """Test construction of filter regions from signals and keyspaces."""
        # Create two keyspaces, two signals and two connections with equivalent
        # synapses.
        # Create two keyspaces, two signals and two reception parameters with
        # different synapses.
        ks_a = mock.Mock(name="Keyspace[A]")
        signal_a = SignalParameters(keyspace=ks_a, latching=False)

        ks_b = mock.Mock(name="Keyspace[B]")
        signal_b = SignalParameters(keyspace=ks_b, latching=False)

        rp_a = ReceptionParameters(nengo.Lowpass(0.01), 3)
        rp_b = ReceptionParameters(None, 5)

        # Create the type of dictionary that is expected as input
        specs = [
            ReceptionSpec(signal_a, rp_a),
            ReceptionSpec(signal_b, rp_b),
        ]

        # Create the regions, with minimisation
        filter_region, routing_region = make_filter_regions(specs, 0.001,
                                                            width=1)

        # Check that the filter region is as expected
        for f in filter_region.filters:
            assert (f == LowpassFilter(1, False, 0.01) or
                    f == NoneFilter(1, False))  # noqa: E711
