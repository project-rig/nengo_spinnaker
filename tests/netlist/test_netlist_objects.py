import pytest
import random
from rig import machine

from nengo_spinnaker.netlist.objects import Vertex, VertexSlice, NMNet


def test_vertex():
    resources = {machine.Cores: 1, machine.SDRAM: 8*1024}

    v = Vertex(application="test", resources=resources)

    assert v.resources == resources
    assert v.resources is not resources
    assert v.application == "test"


def test_vertex_slice():
    # No resources
    v = VertexSlice(slice(None))
    assert v.slice == slice(None)
    assert v.application is None
    assert v.resources == dict()

    # Provide resources and application
    resources = {machine.Cores: 1, machine.SDRAM: 8*1024}

    v = VertexSlice(slice(0, 10), application="test", resources=resources)
    assert v.application == "test"
    assert v.resources == resources
    assert v.resources is not resources


@pytest.mark.parametrize("single_source", (True, False))
@pytest.mark.parametrize("single_sink", (True, False))
def test_net(single_source, single_sink):
    # Create the source(s), sink(s), weight and keyspace
    sources = object() if single_source else [object() for _ in range(3)]
    sinks = object() if single_sink else [object() for _ in range(3)]
    weight = random.randint(0, 100)
    keyspace = object()

    # Create the net
    net = NMNet(sources, sinks, weight, keyspace)

    # Check the net
    if single_source:
        assert net.sources == [sources]
    else:
        assert net.sources == sources
        assert net.sources is not sources

    if single_sink:
        assert net.sinks == [sinks]
    else:
        assert net.sinks == sinks
        assert net.sinks is not sinks

    assert net.weight == weight
    assert net.keyspace is keyspace
