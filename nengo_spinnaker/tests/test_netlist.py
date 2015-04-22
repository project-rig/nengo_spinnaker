import mock
import pytest
from rig.bitfield import BitField
from rig import machine

from nengo_spinnaker import netlist


def test_vertex():
    constraints = [mock.Mock()]
    resources = {machine.Cores: 1, machine.SDRAM: 8*1024}

    v = netlist.Vertex(application="test", constraints=constraints,
                       resources=resources)

    assert v.constraints == constraints
    assert v.constraints is not constraints
    assert v.resources == resources
    assert v.resources is not resources
    assert v.application == "test"


def test_vertex_slice():
    # No resources or constraints
    v = netlist.VertexSlice(slice(None))
    assert v.slice == slice(None)
    assert v.application is None
    assert v.constraints == list()
    assert v.resources == dict()

    # Provide resources and constraints and application
    resources = {machine.Cores: 1, machine.SDRAM: 8*1024}
    constraints = [mock.Mock()]

    v = netlist.VertexSlice(slice(0, 10), application="test",
                            constraints=constraints,
                            resources=resources)
    assert v.application == "test"
    assert v.constraints == constraints
    assert v.constraints is not constraints
    assert v.resources == resources
    assert v.resources is not resources


class TestNet(object):
    def test_single_sink(self):
        """Create a net with a single sink and source."""
        # Create source, sink and keyspace
        v_source = netlist.Vertex()
        v_sink = netlist.Vertex()

        source = netlist.VertexSlice(slice(0, 5))
        sink = netlist.VertexSlice(slice(1, 6))
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
        v_source = netlist.Vertex()
        v_sink = netlist.Vertex()

        source = netlist.VertexSlice(slice(0, 5))
        sinks = [netlist.VertexSlice(slice(1, 6)),
                 netlist.VertexSlice(slice(5, 8))]
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
        v_source = netlist.Vertex()
        v_sink = netlist.Vertex()

        source = netlist.VertexSlice(slice(0, 5))
        sink = netlist.VertexSlice(slice(1, 6))
        weight = 3
        keyspace = BitField(length=length)

        with pytest.raises(ValueError) as excinfo:
            netlist.Net(source, sink, weight, keyspace)

        err_string = str(excinfo.value)
        assert "keyspace" in err_string
        assert "32" in err_string
        assert "{}".format(length) in err_string
