"""Model Optimisations
"""
from __future__ import absolute_import

import itertools
import six

from nengo_spinnaker.operators import Filter


def remove_childless_filters(model):
    """Remove all Filter operators which do not transmit to anything from a
    :py:class:`~nengo_spinnaker.builder.Model`.

    Transmitting values to a filter which then doesn't forward them is a waste
    of network bandwidth. This method optimises out all filters which do not
    transmit to at least one other operator. This method will not remove cycles
    of Filters which have no output.
    """
    while True:
        # Childless Filters are those in EITHER the dictionary of object
        # operators or the set of extra operators which have no outgoing
        # signals.
        childless_filters = [
            (k, v) for k, v in
            itertools.chain(six.iteritems(model.object_operators),
                            ((None, v) for v in model.extra_operators)) if
            (isinstance(v, Filter) and  # It's a filter
             model.get_signals_connections_from_object(v) == {})  # Unconnected
        ]

        if not childless_filters:
            # If there are no childless filters then we have nothing to do
            break

        # Remove each of the childless filters in turn, this may require us to
        # remove yet further filters so we go through the loop again.
        for obj, filt in childless_filters:
            # Generate a list of all connections and signals to remove
            remove_conns = [
                (c, s) for c, s in
                itertools.chain(six.iteritems(model.connections_signals),
                                ((None, s) for s in model.extra_signals))
                if filt in [ss.obj for ss in s.sinks]
            ]

            # Remove all of these connections and signals
            for conn, sig in remove_conns:
                if conn is None:
                    model.extra_signals.remove(sig)
                else:
                    model.connections_signals.pop(conn)

            # Remove the Filter operator itself
            if obj is None:
                model.extra_operators.remove(filt)
            else:
                model.object_operators.pop(obj)
