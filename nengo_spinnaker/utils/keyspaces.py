from __future__ import absolute_import

import collections
from rig.bitfield import BitField


def get_derived_keyspaces(keyspace, values, max_v=None,
                          field_identifier="index"):
    """Get a generator of keyspaces with the index field filled in with the
    given values.

    For example::

        >>> ks = BitField(length=32)
        >>> ks.add_field("index")
        >>> for x in get_derived_keyspaces(ks, (slice(0, 5), 5, 6, 7,
        ...                                     slice(8, 10))):
        ...     print(x)
        <32-bit BitField 'index':0>
        <32-bit BitField 'index':1>
        <32-bit BitField 'index':2>
        <32-bit BitField 'index':3>
        <32-bit BitField 'index':4>
        <32-bit BitField 'index':5>
        <32-bit BitField 'index':6>
        <32-bit BitField 'index':7>
        <32-bit BitField 'index':8>
        <32-bit BitField 'index':9>

    The minimum value for a slice is taken to be zero, but the max value
    can be specified:

        >>> for x in get_derived_keyspaces(ks, slice(None), max_v=3):
        ...     print(x)
        <32-bit BitField 'index':0>
        <32-bit BitField 'index':1>
        <32-bit BitField 'index':2>

    A `ValueError` is raised if these values are missing.

        >>> list(get_derived_keyspaces(ks, slice(None)))
        Traceback (most recent call last):
        ValueError: no stop value specified
    """
    def _get_with_indices(v):
        if isinstance(v, slice):
            if max_v is None and v.stop is None:
                raise ValueError("no stop value specified")

            start = 0 if v.start is None else v.start
            stop = max_v if v.stop is None else v.stop
            step = 1 if v.step is None else v.step

            for i in range(start, stop, step):
                yield keyspace(**{field_identifier: i})
        else:
            yield keyspace(**{field_identifier: v})

    if isinstance(values, collections.Iterable):
        for v in values:
            for x in _get_with_indices(v):
                yield x
    else:
        for x in _get_with_indices(values):
            yield x


class KeyspaceContainer(collections.defaultdict):
    """A container which can recall or allocate specific keyspaces to modules
    and users on request.

    A region of the keyspace ("user") is updated to indicate to which user the
    keyspace belongs.

    The default keyspace can be obtained by requesting the keyspace with the
    name "nengo". Additional keyspaces can be requested and are automagically
    created.

    ..warning::
        The `user` field is reserved for this container.
    """
    class _KeyspaceGetter(object):
        def __init__(self, ks):
            self._count = 0
            self._ks = ks

        def __call__(self):
            new_ks = self._ks(user=self._count)
            self._count += 1
            return new_ks

    def __init__(self, routing_tag="routing",
                 filter_routing_tag="filter_routing"):
        """Create a new keyspace container with the given tags for routing and
        filter routing.
        """
        # The tags
        self._routing_tag = routing_tag
        self._filter_routing_tag = filter_routing_tag

        # The keyspaces
        self._master_keyspace = _master_keyspace = BitField(length=32)
        _master_keyspace.add_field(
            "user", tags=[self.routing_tag, self.filter_routing_tag])

        # Initialise the defaultdict behaviour
        super(KeyspaceContainer, self).__init__(
            self._KeyspaceGetter(_master_keyspace))

        # Add the default keyspace
        nengo_ks = self["nengo"]
        nengo_ks.add_field("object", tags=[self.routing_tag,
                                           self.filter_routing_tag])
        nengo_ks.add_field("cluster", tags=[self.routing_tag])
        nengo_ks.add_field("connection", tags=[self.routing_tag,
                                               self.filter_routing_tag])
        nengo_ks.add_field("index", start_at=0)

    def assign_fields(self):
        """Call `assign_fields` on the master keyspace, forcing field
        assignation for all keyspaces.
        """
        self._master_keyspace.assign_fields()

    @property
    def routing_tag(self):
        """The tag used in creating routing table entries."""
        return self._routing_tag

    @property
    def filter_routing_tag(self):
        """The tag used in creating filter routing table entries."""
        return self._filter_routing_tag


def is_nengo_keyspace(keyspace):
    """Return True if the keyspace is the default Nengo keyspace.

    Example::

        >>> ksc = KeyspaceContainer()
        >>> is_nengo_keyspace(ksc["nengo"])
        True

        >>> is_nengo_keyspace(ksc["not_nengo"])
        False

    Parameters
    ----------
    keyspace : :py:class:`rig.bitfield.BitField`
        Bitfield representation of keyspace.

    Returns
    -------
    bool
        True if the bitspace is a member of the class of Nengo keyspaces.
    """
    return keyspace.user == 0
