"""Construct a set of vertices and nets which may be simulated on SpiNNaker.
"""
import collections
from itertools import chain
from six import iteritems, itervalues

from .netlist import Net
from .utils.collections import (
    mrolookupdict, noneignoringlist, registerabledict, flatinsertionlist)
from .utils.itertools import flatten


class SpiNNakerModel(object):
    """A set of vertices and nets which can be loaded onto a SpiNNaker machine
    for simulation.

    Attributes
    ----------
    nets : [:py:class:`~nengo_spinnaker.netlist.Net`, ...]
        List of nets for SpiNNaker.
    vertices : [:py:class:`~nengo_spinnaker.netlist.Vertex`, ...]
        List of vertices, each of which maps directly to an application running
        on a SpiNNaker core.
    groups : {vertex: int, ...}
        Map of vertices to an index which indicates which group they are a
        part of, if any.
    load_functions : [function, ...]
        Functions which will be called to load data required for any simulation
        on the SpiNNaker machine.
    before_simulation_functions : [function, ...]
        Functions which will be called to load data required for the next
        period of simulation.
    after_simulation_functions : [function, ...]
        Functions which will be called to retrieve data after a period of
        simulation.
    placements : {vertex : (x, y), ...}
        Map of vertices to the co-ordinates of chip it has been placed on.
    allocations : {vertex {resource : slice, ...}, ...}
        Map of vertices to the resources they have been assigned.
    application_memory : {vertex : MemoryIO, ...}
        Map of vertices to a file-like object which allows writing and reading
        from the SDRAM allocated to them.
    routes : {Net: RoutingTree, ...}
        Map of Nets to the routes they will take through a SpiNNaker machine.

    The last four attributes are only filled in when the model is placed and
    routed.
    """

    builders = registerabledict()
    """Builder methods for the model.

    Each builder should expect the original Nengo object, the annotation for
    the object, the model constructed by the Nengo builder and the annotations
    created by the previous build step.
    """

    def __init__(self, nets, vertices, groups, load_functions,
                 before_simulation_functions, after_simulation_functions):
        # Store all the parameters
        self.nets = list(nets)
        self.vertices = list(vertices)
        self.groups = dict(groups)
        self.load_functions = list(load_functions)
        self.before_simulation_functions = list(before_simulation_functions)
        self.after_simulation_functions = list(after_simulation_functions)

        # We have no placements, allocations, routes or maps yet
        self.placements = dict()
        self.allocations = dict()
        self.application_memory = dict()
        self.routes = dict()

    @classmethod
    def from_annotations(cls, model, annotations):
        """Create a model from a Nengo Model and a set of SpiNNaker
        annotations.

        Parameters
        ----------
        model : :py:class:`~nengo.builder.Model`
        annotations : :py:class:`~nengo_spinnaker.annotations.Annotations`

        Returns
        -------
        Model
            Set of nets and vertices which may be simulated on SpiNNaker.
        """
        # Build all the objects
        builders = mrolookupdict(cls.builders)
        vertices = flatinsertionlist()
        built_vertices = dict()
        load_functions = noneignoringlist()
        before_simulation_functions = noneignoringlist()
        after_simulation_functions = noneignoringlist()

        for obj, annotation in chain(
                iteritems(annotations.objects),
                [(None, x) for x in annotations.extra_objects]):
            # Perform the build
            try:
                builder = builders[type(annotation)]
            except KeyError:
                raise TypeError(
                    "No known builder for object of type {}".format(
                        type(annotation).__name__)
                )

            vertex, load_function, pre_function, post_function = \
                builder(obj, annotation, model, annotations)

            # Add these objects
            built_vertices[annotation] = vertex
            vertices.append(vertex)  # List is flattened on appending

            load_functions.append(load_function)
            before_simulation_functions.append(pre_function)
            after_simulation_functions.append(post_function)

        # Construct the groups dictionary
        # Vertex -> group index
        groups = dict()
        for i, vs in enumerate(x for x in itervalues(built_vertices) if
                               isinstance(x, collections.Iterable)):
            groups.update({v: i for v in vs})

        # Build all the nets
        nets = list()
        for annotation in chain(itervalues(annotations.connections),
                                annotations.extra_connections):
            # Get the source and sinks and work out which of the built vertices
            # these are.
            sources = built_vertices[annotation.source.object]
            if not isinstance(sources, collections.Iterable):
                sources = [sources]
            sinks = list(flatten(
                built_vertices[s.object] for s in annotation.sinks))

            # Build the new net(s)
            for source in sources:
                nets.append(
                    Net(source, sinks, annotation.weight,
                        annotation.keyspace)
                )

        # Create the model
        return cls(nets, vertices, groups, load_functions,
                   before_simulation_functions, after_simulation_functions)
