"""Place and route utilities.
"""
from __future__ import absolute_import

import pickle

from six import iteritems

from collections import defaultdict

from rig.netlist import Net

from rig.place_and_route.constraints import (LocationConstraint,
                                             RouteEndpointConstraint,
                                             SameChipConstraint)

from nengo_spinnaker.builder import Model
from nengo_spinnaker.node_io import Ethernet


def create_network_netlist(network, n_steps, fp, dt=0.001):
    """Create a netlist of a network running for a number of steps, dump that
    netlist to file.
    """
    # Build the network, assuming EthernetIO
    model = Model(dt)
    node_io = Ethernet()
    model.build(network, **node_io.builder_kwargs)

    # Build the netlist
    netlist = model.make_netlist(n_steps).as_rig_arguments()
    pickle_netlist(netlist, fp)


def pickle_netlist(netlist_dict, fp, **kwargs):
    """Dump a pickle of a netlist to a file.

    This function replaces all vertices with `object` instances so that
    nengo-specific or project-specific dependencies are not included.
    """
    # {old_vertex: new_vertex, ...}
    new_vertices = defaultdict(object)

    netlist_dict["vertices_resources"] = {
        new_vertices[vertex]: resources
        for (vertex, resources)
        in iteritems(netlist_dict["vertices_resources"])
    }

    netlist_dict["nets"] = [
        Net(new_vertices[net.source],
            [new_vertices[sink] for sink in net.sinks],
            net.weight)
        for net in netlist_dict["nets"]
    ]

    old_constraints = netlist_dict["constraints"]
    netlist_dict["constraints"] = []
    for constraint in old_constraints:
        if isinstance(constraint, LocationConstraint):
            netlist_dict["constraints"].append(
                LocationConstraint(new_vertices[constraint.vertex],
                                   constraint.location))
        elif isinstance(constraint, RouteEndpointConstraint):
            netlist_dict["constraints"].append(
                RouteEndpointConstraint(new_vertices[constraint.vertex],
                                        constraint.route))
        elif isinstance(constraint, SameChipConstraint):
            # Get the new vertices
            vs = [new_vertices[v] for v in constraint.vertices]
            netlist_dict["constraints"].append(SameChipConstraint(vs))
        else:
            netlist_dict["constraints"].append(constraint)

    pickle.dump(netlist_dict, fp, **kwargs)
