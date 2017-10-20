import struct

import numpy as np
import nengo

import nengo_spinnaker
from nengo_spinnaker.operators.lif import Regions

def get_learnt_decoders(sim, ens):
    data = []
    for cluster in sim.model.object_operators[ens].clusters:
        for vx in cluster.vertices:
            mem = vx.region_memory[Regions.learnt_decoders]
            mem.seek(0)

            d = mem.read()
            d = struct.unpack('%di' % (len(d)/4), d)
            data.append(d)

    d = np.hstack(data)
    d.shape = -1, ens.n_neurons

    return nengo_spinnaker.utils.type_casts.fix_to_np(d.T) * sim.dt


class FixedSolver(nengo.solvers.Solver):
    def __init__(self, fixed):
        super(FixedSolver, self).__init__(weights=False)
        self.fixed=fixed
    def __call__(self, A, Y, rng=None, E=None):
        return self.fixed, {}

