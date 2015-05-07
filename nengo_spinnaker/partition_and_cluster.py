"""Tools for partitioning large vertices and then dealing with clusters of the
resulting subvertices.
"""
import collections
import math
from six import iteritems, itervalues

from .utils.keyspaces import is_nengo_keyspace


class Constraint(collections.namedtuple("Constraint",
                                        "maximum, target, max_usage")):
    """Constraint on a resource.

    Attributes
    ----------
    maximum : number
        Hard constraint on maximum usage of the resource.
    target : float
        Explicit cap on the amount of the maximum that may be used.
    max_usage : float
        `maximum * target`

    For example, to define a target of 90% 64KiB-DTCM usage::

        dtcm_constraint = Constraint(64 * 2**10, 0.9)
    """
    def __new__(cls, maximum, target=1.0):
        """Create a new constraint.

        Parameters
        ----------
        maximum : number
            Hard constraint on maximum usage of the resource.
        target : float (optional, default=1.0)
            Explicit cap on the amount of the maximum that may be used.
        """
        return super(Constraint, cls).__new__(
            cls, maximum, target, maximum*target
        )


def partition(initial_slice, constraints_and_getters):
    """Construct a list of slices which satisfy a set of constraints.

    Parameters
    ----------
    initial_slice : :py:class:`slice`
        Initial partition of the object, this should represent everything.
    constraints_and_getters : {:py:class:`~.Constraint`: func, ...}
        Dictionary mapping constraints to functions which will accept a slice
        and return the current usage of the resource for the given slice.

    ..note::
        It is assumed that the object being sliced is homogeneous, i.e., there
        is no difference in usage for `slice(0, 10)` and `slice(10, 20)`.

    Yields
    ------
    :py:class:`slice`
        Slices which meet satisfy all the constraints.

    Raises
    ------
    UnpartitionableError
        If the given problem cannot be solved by this partitioner.
    """
    def constraints_unsatisfied(slices, constraints):
        for s in slices:
            for constraint, usage in iteritems(constraints):
                yield constraint.max_usage < usage(s)

    # Normalise the slice
    if initial_slice.start is None:
        initial_slice = slice(0, initial_slice.stop)

    # While any of the slices fail to satisfy a constraint we partition further
    n_cuts = 1
    max_cuts = initial_slice.stop - initial_slice.start
    slices = [initial_slice]

    while any(constraints_unsatisfied(slices, constraints_and_getters)):
        if n_cuts == 1:
            # If we haven't performed any partitioning then we get the first
            # number of cuts to make.
            n_cuts = max(
                int(math.ceil(usage(initial_slice) / c.max_usage)) for
                c, usage in iteritems(constraints_and_getters)
            )
        else:
            # Otherwise just increment the number of cuts rather than honing in
            # on the expensive elements.
            n_cuts += 1

        if n_cuts > max_cuts:
            # We can't cut any further, so the problem can't be solved.
            raise UnpartitionableError

        # Partition
        slices = divide_slice(initial_slice, n_cuts)

    # Yield the partitioned slices
    for s in divide_slice(initial_slice, n_cuts):
        yield s


def divide_slice(initial_slice, n_slices):
    """Create a set of smaller slices from an original slice.

    Parameters
    ----------
    initial_slice : :py:class:`slice`
        A slice which must have `start` and `stop` set.
    n_slices : int
        Number of slices to produce.

    Yields
    ------
    :py:class:`slice`
        Slices which when combined would be equivalent to `initial_slice`.
    """
    start = initial_slice.start
    stop = initial_slice.stop
    chunk = int(math.ceil((stop - start) / n_slices))
    pos = start

    while pos < stop:
        yield slice(pos, min(pos + chunk, stop))
        pos += chunk


class UnpartitionableError(Exception):
    """Indicates that a given partitioning problem cannot be solved."""


def identify_clusters(placed_vertices, nets, groups):
    """Group vertex slices in clusters (based on chip assignation) and assign
    the cluster ID to nets which originate from these subvertices.

    A vertex may be partitioned into a number of vertex slices which are placed
    onto the cores of two SpiNNaker chips.  The vertex slices on these chips
    form two clusters; packets from these clusters need to be differentiated in
    order to route packets correctly.  For example::

       +--------+                      +--------+
       |        | ------- (a) ------>  |        |
       |   (A)  |                      |   (B)  |
       |        | <------ (b) -------  |        |
       +--------+                      +--------+

    Packets traversing `(a)` need to be differentiated from packets traversing
    `(b)`.  This can be done by including an additional field in the packet
    keys which indicates which chip the packet was sent from - in this case a
    single bit will suffice with packets from `(A)` using a key with the bit
    not set and packets from `(B)` setting the bit.

    This method will assign a unique ID to each cluster of vertex slices (e.g.,
    `(A)` and `(B)`) by assigning the index to the `cluster` attribute of each
    subvertex and will ensure the same cluster ID is present in the keyspace of
    all nets originating from the cluster (and using the standard Nengo
    keyspace).

    Parameters
    ----------
    placed_vertices : {vertex: (x, y), ...}
        Vertex placements (as produced by
        :py:func:`rig.place_and_route.place`).
    nets : [:py:class:`nengo_spinnaker.netlist.Net`, ...]
    groups : [{Vertex, ...}, ...]
        List of groups of vertices. This is used to identify objects which may
        be joined into a cluster.
    """
    class Cluster(object):
        """Internal representation of a cluster."""
        __slots__ = ["vertices", "nets"]

        def __init__(self):
            self.vertices = list()
            self.nets = list()

    # For each group of vertices perform clustering
    for group in groups:
        # Build a map of placements to vertices, this will be used to identify
        # clusters.
        # Maps co-ordinate to cluster
        clusters = collections.defaultdict(Cluster)

        # For each vertex in the group add it to a cluster
        for vertex in group:
            coord = placed_vertices[vertex]
            clusters[coord].vertices.append(vertex)

        # Look through all of the nets, add them to clusters based on their
        # originating vertex.
        for net in nets:
            # If the net uses the default keyspace
            if is_nengo_keyspace(net.keyspace):
                if net.source in group:
                    # Then include it in the cluster if its originating vertex
                    # is a vertex slice
                    coord = placed_vertices[net.source]
                    clusters[coord].nets.append(net)

        # Iterate through the clusters of the group
        for cluster_id, cluster in enumerate(itervalues(clusters)):
            # Assign the cluster ID to the vertex slices
            for vertex in cluster.vertices:
                vertex.cluster = cluster_id

            # Assign the cluster ID to the nets
            for net in cluster.nets:
                net.keyspace = net.keyspace(cluster=cluster_id)

    # Assign a cluster ID of zero to all other nets
    for net in nets:
        # If the net uses the default keyspace
        if is_nengo_keyspace(net.keyspace):
            if net.keyspace.cluster is None:
                net.keyspace = net.keyspace(cluster=0)
