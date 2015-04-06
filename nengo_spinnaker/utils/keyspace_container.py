from rig.bitfield import BitField


class KeyspaceContainer(object):
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
        >>> default_ks.get_tags('nengo_dimension') == set([])
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

    Finally, field sizes may be fixed.

        >>> # Before fixing trying to get a mask fails
        >>> new_ks.get_mask(tag=ksc.routing_tag)
        Traceback (most recent call last):
        ValueError: Field 'user' does not have a fixed size/position.

        >>> # After fixing it works fine
        >>> ksc.assign_fields()
        >>> hex(new_ks.get_mask(tag=ksc.routing_tag))
        '0x3'

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

    def __init__(self, routing_tag="routing",
                 filter_routing_tag="filter_routing"):
        """Create a new keyspace container with the given tags for routing and
        filter routing.
        """
        # The tags
        self._routing_tag = routing_tag
        self._filter_routing_tag = filter_routing_tag

        # The keyspaces
        self._master_keyspace = BitField(length=32)
        self._master_keyspace.add_field(
            "user", tags=[self.routing_tag, self.filter_routing_tag])

        self._assigned_keyspaces = {}

        # Add the default keyspace
        nengo_ks = self._assigned_keyspaces["nengo"] = \
            self._master_keyspace(user=0)
        nengo_ks.add_field("nengo_object", tags=[self.routing_tag,
                                                 self.filter_routing_tag])
        nengo_ks.add_field("nengo_cluster", tags=[self.routing_tag])
        nengo_ks.add_field("nengo_connection", tags=[self.routing_tag,
                                                     self.filter_routing_tag])
        nengo_ks.add_field("nengo_dimension")

    def __getitem__(self, user):
        """Get the keyspace for a given user creating it if it doesn't already
        exist.

        Parameters
        ----------
        user : string
            Unique name to identify the user.  It is recommended that the same
            string be used to namespace fields within the returned keyspace.

        Returns
        -------
        :py:class:`rig.bitfields.BitField`
            A 32-bit bitfield which can be used to describe keyspaces.
        """
        if user not in self._assigned_keyspaces:
            # Create a new keyspace with a different user number
            self._assigned_keyspaces[user] = \
                self._master_keyspace(user=len(self._assigned_keyspaces))

        return self._assigned_keyspaces[user]

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
