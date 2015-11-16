"""Tools for partitioning large vertices.  """
import collections
import math
from six import iteritems


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
    # Extract current position, start and stop
    pos = start = initial_slice.start
    stop = initial_slice.stop

    # Determine the chunk sizes
    chunk = (stop - start) // n_slices
    n_larger = (stop - start) % n_slices

    # Yield the larger slices
    for _ in range(n_larger):
        yield slice(pos, pos + chunk + 1)
        pos += chunk + 1

    # Yield the standard sized slices
    for _ in range(n_slices - n_larger):
        yield slice(pos, pos + chunk)
        pos += chunk


class UnpartitionableError(Exception):
    """Indicates that a given partitioning problem cannot be solved."""
