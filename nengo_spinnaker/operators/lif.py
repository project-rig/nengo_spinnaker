"""LIF Ensemble

Takes an intermediate representation of a LIF ensemble and returns a vertex and
appropriate callbacks to load and prepare the ensemble for simulation on
SpiNNaker.  The build method also manages the partitioning of the ensemble into
appropriate sized slices.
"""

import collections
import numpy as np
from rig import type_casts
import struct

s1615 = type_casts.float_to_fix(True, 32, 15)


class EnsembleLIF(object):
    """Controller for an ensemble of LIF neurons."""
    def __init__(self, ensemble):
        """Create a new LIF ensemble controller."""
        self.ensemble = ensemble
        self.direct_input = np.zeros(ensemble.size_in)
        self.local_probes = list()


class SystemRegion(collections.namedtuple(
    "SystemRegion", "n_input_dimensions, n_output_dimensions, "
                    "machine_timestep, t_ref, t_rc, dt, probe_spikes")):
    """Region of memory describing the general parameters of a LIF ensemble."""

    def sizeof(self):
        """Get the number of bytes necessary to represent this region of
        memory.
        """
        return 8 * 4  # 8 words

    def write_subregion_to_file(self, vertex_slice, fp):
        """Write the system region for a specific vertex slice to a file-like
        object.
        """
        n_neurons = vertex_slice.stop - vertex_slice.start
        data = struct.pack(
            "<8I",
            self.n_input_dimensions,
            self.n_output_dimensions,
            n_neurons,
            self.machine_timestep,
            int(self.t_ref // self.dt),
            s1615(self.dt / self.t_rc),
            (0x1 if self.probe_spikes else 0x0),
            1
        )
        fp.write(data)
