"""Model Optimisations
"""
from __future__ import absolute_import

import itertools
from six import iteritems, itervalues

from nengo_spinnaker.operators import Filter


def remove_sinkless_signals(model):
    """Remove all Signals which do not have any sinks from a
    :py:class:`~nengo_spinnaker.builder.Model`.
    """
    # Create a list of signals to remove by iterating through the signals which
    # are related to connections and finding any with no sinks.
    sinkless_signals = [(c, s) for c, s in iteritems(model.connections_signals)
                        if len(s.sinks) == 0]

    # Now remove all sinkless signals
    for conn, sig in sinkless_signals:
        model.connections_signals.pop(conn)

    # Create a list of signals to remove by finding signals which are not
    # related to connections and which have no sinks.
    sinkless_signals = [s for s in model.extra_signals if len(s.sinks) == 0]

    # Now remove all sinkless signals
    for sig in sinkless_signals:
        model.extra_signals.remove(sig)


def remove_childless_filters(model):
    """Remove all Filter operators which do not transmit to anything from a
    :py:class:`~nengo_spinnaker.builder.Model`.

    Transmitting values to a filter which then doesn't forward them is a waste
    of network bandwidth. This method optimises out all filters which do not
    transmit to at least one other operator. This method will not remove cycles
    of Filters which have no output.
    """
    # We loop through removing filters that aren't connected to anything so
    # that we can deal with the case:
    #
    #     Filter -> Filter -> Filter -> Filter
    #
    # Which should result in the removal of all of the above filters. We break
    # as soon as there are no more childless filters.
    while True:
        # Childless Filters are those in EITHER the dictionary of object
        # operators or the set of extra operators which have no outgoing
        # signals.
        childless_filters = [
            (k, v) for k, v in
            itertools.chain(iteritems(model.object_operators),
                            ((None, v) for v in model.extra_operators)) if
            (isinstance(v, Filter) and  # It's a filter
             model.get_signals_connections_from_object(v) == {})  # Unconnected
        ]

        if not childless_filters:
            # If there are no childless filters then we have nothing to do
            break

        # Remove each of the childless filters in turn.
        for obj, filt in childless_filters:
            # Remove the filter from the list of sinks of each of the signals
            # which target it.
            for sig in itertools.chain(itervalues(model.connections_signals),
                                       model.extra_signals):
                sinks = [s for s in sig.sinks if s.obj is filt]
                for sink in sinks:
                    sig.sinks.remove(sink)

            # Remove the Filter operator itself
            if obj is None:
                model.extra_operators.remove(filt)
            else:
                model.object_operators.pop(obj)

        # Removing filters from the lists of sinks of signals may produce
        # signals which have no sinks, we should remove these as it will allow
        # us to find further filters with no children.
        remove_sinkless_signals(model)
