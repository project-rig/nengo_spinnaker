"""LIF Ensemble

Takes an intermediate representation of a LIF ensemble and returns a vertex and
appropriate callbacks to load and prepare the ensemble for simulation on
SpiNNaker.  The build method also manages the partitioning of the ensemble into
appropriate sized slices.
"""

from ..netlist import Vertex, VertexSlice


class LIFVertex(Vertex):
    """Vertex representing an ensemble constructed of LIF neurons.
    """
    @classmethod
    def from_intermediate_representation(ens_intermediate, ens, irn):
        """Create a new LIF vertex from an intermediate representation.

        Parameters
        ----------
        ens_intermediate : IntermediateEnsemble
            Annotations for the ensemble.
        ens : :py:class:`nengo.Ensemble`
            Original ensemble object
        irn : IntermediateRepresentation
            The intermediate representation of the network this ensemble is a
            part of.

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
