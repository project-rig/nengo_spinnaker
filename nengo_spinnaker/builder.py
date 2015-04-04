from itertools import chain
from six import iteritems, itervalues

from .netlist import Net
from .utils.collections import noneignoringlist, registerabledict
from .utils.itertools import flatten


class Builder(object):
    """Converts objects from an intermediate representation into a set of
    vertices that can be placed and allocated.
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

    @classmethod
    def build(cls, intermediate_representation, extra_builders):
        """Convert an intermediate representation into a set of vertices and
        sets of callbacks.

        Parameters
        ----------
        intermediate_representation : `IntermediateRepresentation`
            An intermediary representation of the model to build.
        extra_builders : {type: callable, ...}
            Any additional build methods that are required.
        """
        # Construct the set of builders
        builders = dict(cls.builders)
        builders.update(extra_builders)

        # Build all of the objects in turn
        vertices = dict()  # Built vertices (mapped to build nets later)
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

            vertices[obj] = vertex
            pre_loads.append(pre_load)  # `None` is not appended
            pre_sims.append(pre_sim)  # `None` is not appended
            post_sims.append(post_sim)  # `None` is not appended

        # Convert all of the intermediate nets into finalised nets
        nets = list()
        all_nets = chain(
            itervalues(intermediate_representation.connection_map),
            intermediate_representation.extra_connections
        )
        for net in all_nets:
            # Get the source and sink(s)
            sources = vertices[net.source.object]
            sinks = vertices[net.sink.object]

            if not isinstance(sources, list):
                sources = [sources]

            # Create the new nets
            for source in sources:
                nets.append(Net(source, sinks, 1, net.keyspace))

        # Finally, return all vertices and nets
        return (
            list(flatten(vertices.values())), nets, list(pre_loads),
            list(pre_sims), list(post_sims)
        )
