"""Place and route utilities.
"""
import pickle

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
    netlist = model.make_netlist(n_steps)
    pickle_netlist(netlist, fp)


def pickle_netlist(netlist, fp, **kwargs):
    """Dump a pickle of a netlist to a file."""
    pickle.dump(netlist.as_rig_arguments(), fp, **kwargs)
