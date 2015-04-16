import collections

from rig.bitfield import BitField


class KeyspaceContainer(collections.defaultdict):
    """A container which can recall or allocate specific keyspaces to modules
    and users on request.

    A region of the keyspace ("user") is updated to indicate to which user the
    keyspace belongs.

    The default keyspace can be obtained by requesting the keyspace with the
    name "nengo".

        >>> ksc = KeyspaceContainer()
        >>> default_ks = ksc["nengo"]
        >>> default_ks.get_tags('nengo_object') == {ksc.routing_tag,
        ...                                         ksc.filter_routing_tag}
        True
        >>> default_ks.get_tags('nengo_connection') == {ksc.routing_tag,
        ...                                             ksc.filter_routing_tag}
        True
        >>> default_ks.get_tags('nengo_cluster') == {ksc.routing_tag}
        True
        >>> default_ks.get_tags('nengo_dimension') == {ksc.dimension_tag}
        True
        >>> default_ks
        <32-bit BitField 'user':0, 'nengo_object':?, 'nengo_cluster':?, \
'nengo_connection':?, 'nengo_dimension':?>

    Additional keyspaces can be requested and are automagically created.

        >>> new_ks = ksc["new_user"]
        >>> new_ks
        <32-bit BitField 'user':1>

        >>> new_ks2 = ksc["new_user2"]
        >>> new_ks2
        <32-bit BitField 'user':2>

    ..warning::
        Namespacing should be used to avoid collisions between keyspaces.

    ..warning::
        The `user` field is reserved for this container.

    Re-requesting an existing keyspace simply returns the existing one.

        >>> ksc["new_user"]
        <32-bit BitField 'user':1>
        >>> ksc["new_user"] is new_ks
        True
        >>> ksc["new_user"] is not new_ks2
        True

    The routing and filter routing tags are also exposed through this interface
    as strings.

        >>> ksc.routing_tag
        'routing'
        >>> ksc.filter_routing_tag
        'filter_routing'
        >>> ksc.dimension_tag
        'dimension'

    Finally, field sizes may be fixed.

        >>> # Before fixing trying to get a mask fails
        >>> new_ks.get_mask(tag=ksc.routing_tag)
        Traceback (most recent call last):
        ValueError: Field 'user' does not have a fixed size/position.

        >>> # After fixing it works fine
        >>> ksc.assign_fields()
        >>> hex(new_ks.get_mask(tag=ksc.routing_tag))
        '0x30'

    The default fields and their tags:

    ==================   =======================
    Field                Tags
    ==================   =======================
    `nengo_object`       routing, filter routing
    `nengo_cluster`      routing
    `nengo_connection`   routing, filter routing
    `nengo_dimension`
    ==================   =======================

    `nengo_cluster` is only used for objects which are split across multiple
    chips to indicate which chip they are located on (all is needed is a simple
    count).
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
                 filter_routing_tag="filter_routing",
                 dimension_tag="dimension"):
        """Create a new keyspace container with the given tags for routing and
        filter routing.
        """
        # The tags
        self._routing_tag = routing_tag
        self._filter_routing_tag = filter_routing_tag
        self._dimension_tag = dimension_tag

        # The keyspaces
        self._master_keyspace = _master_keyspace = BitField(length=32)
        _master_keyspace.add_field(
            "user", tags=[self.routing_tag, self.filter_routing_tag])

        # Initialise the defaultdict behaviour
        super(KeyspaceContainer, self).__init__(
            self._KeyspaceGetter(_master_keyspace))

        # Add the default keyspace
        nengo_ks = self["nengo"]
        nengo_ks.add_field("nengo_object", tags=[self.routing_tag,
                                                 self.filter_routing_tag])
        nengo_ks.add_field("nengo_cluster", tags=[self.routing_tag])
        nengo_ks.add_field("nengo_connection", tags=[self.routing_tag,
                                                     self.filter_routing_tag])
        nengo_ks.add_field("nengo_dimension", tags=self.dimension_tag,
                           start_at=0)

    def assign_fields(self):
        """Call `assign_fields` on the master keyspace, forcing field
        assignation for all keyspaces.
        """
        self._master_keyspace.assign_fields()

    @property
    def dimension_tag(self):
        """The tag used in extracting dimension data from a key."""
        return self._dimension_tag

    @property
    def routing_tag(self):
        """The tag used in creating routing table entries."""
        return self._routing_tag

    @property
    def filter_routing_tag(self):
        """The tag used in creating filter routing table entries."""
        return self._filter_routing_tag


keyspaces = KeyspaceContainer()
"""The global set of keyspaces."""


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
