import pytest
from rig.bitfield import BitField
from rig import machine

from nengo_spinnaker import netlist, params


class NullVertex(netlist.Vertex):
    n_atoms = params.IntParam(min=0, allow_none=True, default=0)


class TestNet(object):
    def test_single_sink(self):
        """Create a net with a single sink and source."""
        # Create source, sink and keyspace
        v_source = NullVertex()
        v_sink = NullVertex()
        v_source.n_atoms = 100
        v_sink.n_atoms = 100

        source = netlist.VertexSlice(v_source, slice(0, 5))
        sink = netlist.VertexSlice(v_sink, slice(1, 6))
        weight = 3
        keyspace = BitField(length=32)

        # Create the Net, assert the values are stored
        net = netlist.Net(source, sink, weight, keyspace)

        assert net.source is source
        assert net.sinks == [sink]
        assert net.weight == weight
        assert net.keyspace is keyspace

    def test_multiple_sinks(self):
        """Create a net with a single sink and source."""
        # Create source, sink and keyspace
        v_source = NullVertex()
        v_sink = NullVertex()
        v_source.n_atoms = 100
        v_sink.n_atoms = 100

        source = netlist.VertexSlice(v_source, slice(0, 5))
        sinks = [netlist.VertexSlice(v_sink, slice(1, 6)),
                 netlist.VertexSlice(v_sink, slice(5, 8))]
        weight = 3
        keyspace = BitField(length=32)

        # Create the Net, assert the values are stored
        net = netlist.Net(source, sinks, weight, keyspace)

        assert net.source is source
        assert net.sinks == sinks
        assert net.weight == weight
        assert net.keyspace is keyspace

    @pytest.mark.parametrize("length", [16, 31, 33])
    def test_assert_keyspace_length(self, length):
        """Check that the keyspace is only accepted if it is of length 32
        bits.
        """
        v_source = NullVertex()
        v_sink = NullVertex()
        v_source.n_atoms = 100
        v_sink.n_atoms = 100

        source = netlist.VertexSlice(v_source, slice(0, 5))
        sink = netlist.VertexSlice(v_sink, slice(1, 6))
        weight = 3
        keyspace = BitField(length=length)

        with pytest.raises(ValueError) as excinfo:
            netlist.Net(source, sink, weight, keyspace)

        err_string = str(excinfo.value)
        assert "keyspace" in err_string
        assert "32" in err_string
        assert "{}".format(length) in err_string


class TestVertex(object):
    def test_init(self):
        v = netlist.Vertex()
        assert v.n_atoms is None

        resources = {machine.Cores: 2, machine.SDRAM: 8*1024*1024}
        v = netlist.Vertex(resources)
        assert v.resources == resources
        assert v.resources is not resources


class TestVertexSlice(object):
    @pytest.mark.parametrize(
        "sl", [slice(0, 5, 2),  # Has stride != None or 1
               (slice(0, 2), slice(3, 4)),  # Has multiple parts
               slice(-3, -1),  # Relative to end of array
               slice(3, 2),  # Slice goes backwards
               ]
    )
    def test_fails_if_slice_non_contiguous_or_relative(self, sl):
        v = NullVertex()
        v.n_atoms = 100

        with pytest.raises(ValueError) as excinfo:
            netlist.VertexSlice(v, sl)
        assert "slice" in str(excinfo.value)

    @pytest.mark.parametrize(
        "n_atoms, sl, reason",
        [(100, slice(50, 101), "beyond range"),
         (None, slice(50, 101), "cannot be represented by a slice"),
         ]
    )
    def test_fails_if_slices_beyond_range_of_atoms(self, n_atoms, sl, reason):
        # Create the vertex
        v = NullVertex()
        v.n_atoms = n_atoms

        with pytest.raises(ValueError) as excinfo:
            netlist.VertexSlice(v, sl)
        assert reason in str(excinfo.value)

    @pytest.mark.parametrize(
        "sl", [slice(0, 2), slice(0, 4, 1)])
    def test_valid(self, sl):
        v = NullVertex()
        v.n_atoms = 100
        resources = {machine.Cores: 1, machine.SDRAM: 2*1024*1014}

        vs = netlist.VertexSlice(v, sl, resources)
        assert vs.vertex is v
        assert vs.slice == sl
        assert vs.cluster is None
        assert vs.resources == resources
        assert vs.resources is not resources

        assert repr(vs) == "<VertexSlice {!s}[{}:{}]>".format(
            v, sl.start, sl.stop)
