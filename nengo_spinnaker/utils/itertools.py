from __future__ import absolute_import
import collections


def flatten(iterable):
    """Flatten an iterable that may contain multiple depths of other iterables.

    For example::

        >>> list(flatten([1, 2, [3, 4, 5], [[6]]]))
        [1, 2, 3, 4, 5, 6]

    .. warning::
        Strings are not treated any differently from other iterables!
    """
    for i in iterable:
        if isinstance(i, collections.Iterable):
            for j in flatten(i):
                yield j
        else:
            yield i
