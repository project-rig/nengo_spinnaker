"""Test netlist export works as expected."""

import pytest

import os

from tempfile import mkstemp

import pickle

import nengo

from nengo_spinnaker.utils.place_and_route import create_network_netlist


@pytest.fixture
def example_network():
    with nengo.Network("Test network") as network:
        a = nengo.Ensemble(200, 1)
        b = nengo.Ensemble(200, 1)
        nengo.Connection(a, b)
    return network


@pytest.yield_fixture
def temp_filename():
    name = mkstemp()[1]
    yield name
    os.remove(name)


def test_create_network_netlist(example_network, temp_filename):
    with open(temp_filename, "wb") as fp:
        create_network_netlist(example_network, 1.0, fp)

    with open(temp_filename, "rb") as fp:
        netlist = pickle.load(fp)

    # The compiled network should contain one core for each ensemble
    assert len(netlist["vertices_resources"]) == 2

    for vertex in netlist["vertices_resources"]:
        # Should be a native Python object and nothing more fancy!
        assert type(vertex) is object

    # The vertices should be connected together with a single net
    assert len(netlist["nets"]) == 1
    assert netlist["nets"][0].source in netlist["vertices_resources"]
    assert len(netlist["nets"][0].sinks) == 1
    assert netlist["nets"][0].sinks[0] in netlist["vertices_resources"]
    assert netlist["nets"][0].source is not netlist["nets"][0].sinks[0]
