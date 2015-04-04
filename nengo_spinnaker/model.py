from itertools import chain
from six import iteritems, itervalues

from .netlist import Net
from .utils.collections import noneignoringlist, registerabledict
from .utils.itertools import flatten


class Model(object):
    """Represents a set of vertices and nets which may be placed and routed to
    simulate a neural network on a SpiNNaker machine.
    """

    builders = registerabledict()
    """Callables which can construct a vertex or vertices from an intermediate
    representation.

    Callables are registered against the type of the intermediate object and
    must accept the intermediate object, the original object (if present,
    otherwise None) and the whole intermediate representation of a network.

    For example, a `Spam` intermediate object could be built with::

        @Builder.builders.register(Spam)
        def build_spam(spam_intermediate, original_object, irn):
            # Do whatever
            return vertex, pre_load, pre_sim, post_sim

    Callables are expected to return:
     - (1) a vertex object
     - (2) a function (or `None`) to call to prepare the vertex for simulation
     - (3) a function (or `None`) to call immediately before a simulation
     - (4) a function (or `None`) to call immediately after a simulation
    """

    def __init__(self, vertex_map=dict(), net_map=dict(),
                 preload_callables=list(), presim_callables=list(),
                 postsim_callables=list()):
        """Create a new model.

        Parameters
        ----------
        vertex_map : {object : Vertex, object : [Vertex], ...}
            Mapping from intermediate objects to the vertex or vertices they
            instantiate
        net_map : {net : Net, net : [Net]}
            Mapping for intermediate Net to the Net or Nets they instantiate.
        preload_callables : list
            List of functions to call prior to loading the machine (used, e.g.,
            to load vertex data).
        presim_callables : list
            Functions called prior to the start of a simulation (used, e.g., to
            load on data required for just the next simulation steps).
        postsim_callables : list
            Callables called after the end of a run of simulation steps (used,
            e.g., to retrieve probe data).
        """
        self.vertex_map = dict(vertex_map)
        self.net_map = dict(net_map)
        self.preload_callables = list(preload_callables)
        self.presim_callables = list(presim_callables)
        self.postsim_callables = list(postsim_callables)

    @classmethod
    def from_intermediate_representation(cls, intermediate_representation,
                                         extra_builders=dict()):
        """Convert an intermediate representation into a new model.

        Parameters
        ----------
        intermediate_representation : `IntermediateRepresentation`
            An intermediary representation of the model to build.
        extra_builders : {type: callable, ...}
            Any additional build methods that are required.

        Returns
        -------
        :py:class:`.Model`
            A model constructed from the intermediate representation.
        """
        # Create a new empty model
        model = cls()

        # Construct the set of builders
        builders = dict(cls.builders)
        builders.update(extra_builders)

        pre_loads = noneignoringlist()  # Functions to call to load the machine
        pre_sims = noneignoringlist()  # ... to call before each simulation
        post_sims = noneignoringlist()  # ... to call after each simulation

        all_objects_mapped = chain(
            iteritems(intermediate_representation.object_map),
            ((None, o) for o in intermediate_representation.extra_objects)
        )
        for (orig, obj) in all_objects_mapped:
            # Try to get the builder
            if obj.__class__ not in builders:
                raise TypeError(
                    "No builder registered for intermediate object of type "
                    "{}.".format(obj.__class__.__name__)
                )

            # Build
            vertex, pre_load, pre_sim, post_sim = \
                builders[obj.__class__](obj, orig, intermediate_representation)

            model.vertex_map[obj] = vertex
            pre_loads.append(pre_load)  # `None` is not appended
            pre_sims.append(pre_sim)  # `None` is not appended
            post_sims.append(post_sim)  # `None` is not appended

        # Store the callables
        model.preload_callables = list(pre_loads)
        model.presim_callables = list(pre_sims)
        model.postsim_callables = list(post_sims)

        # Convert all of the intermediate nets into finalised nets
        all_nets = chain(
            itervalues(intermediate_representation.connection_map),
            intermediate_representation.extra_connections
        )
        for net in all_nets:
            # Get the source and sink(s)
            sources = model.vertex_map[net.source.object]
            sinks = model.vertex_map[net.sink.object]

            if not isinstance(sources, list):
                sources = [sources]

            # Create the new nets
            model.net_map[net] = [Net(source, sinks, 1, net.keyspace) for
                                  source in sources]

        # Return the built model
        return model

    @property
    def nets(self):
        """Return an iterable of the nets in the model."""
        return flatten(itervalues(self.net_map))

    @property
    def vertices(self):
        """Return an iterable of the vertices in the model."""
        return flatten(itervalues(self.vertex_map))
