"""LIF Ensemble

Takes an intermediate representation of a LIF ensemble and returns a vertex and
appropriate callbacks to load and prepare the ensemble for simulation on
SpiNNaker.  The build method also manages the partitioning of the ensemble into
appropriate sized slices.
"""

import collections
import nengo.neurons
from rig import type_casts
import struct

from ..netlist import VertexSlice

s1615 = type_casts.float_to_fix(True, 32, 15)


class LIF(object):
    """Vertex representing an ensemble constructed of LIF neurons.
    """
    @classmethod
    def from_annotations(cls, ens, ens_annotation, model, annotations):
        """Create a new LIF vertex from an annotated Nengo model.

        Parameters
        ----------
        ens : :py:class:`nengo.Ensemble`
            Original ensemble object
        ens_annotation : \
                :py:class:`~nengo_spinnaker.ensemble.AnnotatedEnsemble`
            Annotations for the ensemble.
        model : :py:class:`~nengo.builder.Model`
            Nengo constructed model.
        annotations : :py:class:`~nengo_spinnaker.annotations.Annotations`
            Nengo/SpiNNaker model annotations.

        Returns
        -------
        vertices, pre_load, pre_sim, post_sim
            A set of vertex slices and callables to load data to SpiNNaker, to
            prepare SpiNNaker for simulation and to retrieve data once
            simulation has occurred.
        """
        # Create the LIF vertex and then partition it.  Return the partitions
        # and references to the callables required by the vertex.
        raise NotImplementedError


class SystemRegion(collections.namedtuple(
    "SystemRegion", "n_input_dimensions, n_output_dimensions, "
                    "machine_timestep, t_ref, t_rc, dt, probe_spikes")):
    """Region of memory describing the general parameters of a LIF ensemble."""
    @classmethod
    def from_annotations(cls, ens, ens_annotation, model, annotations,
                         size_out):
        """Create a new LIF vertex from an annotated Nengo model.

        Parameters
        ----------
        ens : :py:class:`nengo.Ensemble`
            Original ensemble object
        ens_annotation : \
                :py:class:`~nengo_spinnaker.ensemble.AnnotatedEnsemble`
            Annotations for the ensemble.
        model : :py:class:`~nengo.builder.Model`
            Nengo constructed model.
        annotations : :py:class:`~nengo_spinnaker.annotations.Annotations`
            Nengo/SpiNNaker model annotations.
        size_out : int
            Total number of dimensions in the extended and compressed decoder.

        Returns
        -------
        :py:class:`~.SystemRegion`
            System region with appropriate parameters.
        """
        assert isinstance(ens.neuron_type, nengo.neurons.LIF)

        # Just case of extracting the correct details to create the region
        return cls(
            n_input_dimensions=ens.size_in,
            n_output_dimensions=size_out,
            machine_timestep=annotations.machine_timestep,
            t_ref=ens.neuron_type.tau_ref,
            t_rc=ens.neuron_type.tau_rc,
            dt=model.dt,
            probe_spikes=any(
                p.attr == "output" for p in ens_annotation.local_probes
            )
        )

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
