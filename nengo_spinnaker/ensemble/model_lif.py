"""LIF Ensemble

Takes an intermediate representation of a LIF ensemble and returns a vertex and
appropriate callbacks to load and prepare the ensemble for simulation on
SpiNNaker.  The build method also manages the partitioning of the ensemble into
appropriate sized slices.
"""

from ..netlist import VertexSlice


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
