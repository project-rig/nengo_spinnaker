from six import iteritems
import struct

from .region import Region


class KeyspacesRegion(Region):
    """A region of memory which represents data formed from a list of
    :py:class:`~rig.bitfield.BitField` instances representing SpiNNaker routing
    keys.

    Each "row" represents a keyspace, and each "column" is formed by getting
    the result of a function applied to the keyspace.  Each field will be one
    word long, and all keyspaces are expected to be 32-bit long.
    """
    def __init__(self, keyspaces, fields=list(), partitioned_by_atom=False,
                 prepend_num_keyspaces=False):
        """Create a new region representing keyspace information.

        Parameters
        ----------
        keyspaces : iterable
            An iterable containing instances of
            :py:class:`~rig.bitfield.BitField`
        fields : iterable
            An iterable of callables which will be called on each key and must
            return an appropriate sized bytestring representing the data to
            write to memory.  The appropriate size is the number of bytes
            required to represent a full key or mark (e.g., 4 bytes for 32 bit
            keyspaces).
        partitioned_by_atom : bool
            If True then one set of fields will be written out per atom, if
            False then fields for all keyspaces are written out regardless of
            the vertex slice.
        prepend_num_keyspaces : bool
            Prepend a word containing the number of keyspaces to the region
            data when it is written out.
        """
        # Can only support 32-bit keyspaces
        for ks in keyspaces:
            assert ks.length == 32

        # Save the keyspaces, fields and partitioned status
        self.keyspaces = keyspaces[:]
        self.fields = fields[:]
        self.partitioned = partitioned_by_atom
        self.prepend_num_keyspaces = prepend_num_keyspaces
        self.bytes_per_field = 4

    def sizeof(self, vertex_slice):
        """Get the size of a slice of this region in bytes.

        See :py:meth:`.region.Region.sizeof`
        """
        # Get the size from representing the fields
        if not self.partitioned:
            n_keys = len(self.keyspaces)
        else:
            assert vertex_slice.stop < len(self.keyspaces) + 1
            n_keys = vertex_slice.stop - vertex_slice.start

        pp_size = 0 if not self.prepend_num_keyspaces else 4

        return self.bytes_per_field * n_keys * len(self.fields) + pp_size

    def write_subregion_to_file(self, fp, vertex_slice=None, **field_args):
        """Write the data contained in a portion of this region out to file.
        """
        data = b''

        # Get a slice onto the keys
        if self.partitioned:
            assert vertex_slice.stop < len(self.keyspaces) + 1
        key_slice = vertex_slice if self.partitioned else slice(None)

        # Write the prepends
        if self.prepend_num_keyspaces:
            nks = len(self.keyspaces[key_slice])
            data += struct.pack("<I", nks)

        # For each key fill in each field
        for ks in self.keyspaces[key_slice]:
            for field in self.fields:
                data += struct.pack("<I", field(ks, **field_args))

        # Write out
        fp.write(data)


# NOTE: This closure intentionally tries to look like a class.
# TODO: Neaten this docstring.
def KeyField(maps={}, field=None, tag=None):
    """Create new field for a :py:class:`~KeyspacesRegion` that will fill in
    specified fields of the key and will then write out a key.

    Parameters
    ----------
    maps : dict
        A mapping from keyword-argument of the field to the field of the key
        that this value should be inserted into.
    field : string or None
        The field to get the key or None for all fields.

    For example:

        ks = Keyspace()
        ks.add_field(i)
        # ...

        kf = KeyField(maps={'subvertex_index': 'i'})
        k = Keyspace()
        kf(k, subvertex_index=11)

    Will return the key with the 'i' key set to 11.
    """
    key_field = field

    def key_getter(keyspace, **kwargs):
        # Build a set of fields to fill in
        fills = {}
        for (kwarg, field) in iteritems(maps):
            fills[field] = kwargs[kwarg]

        # Build the key with these fills made
        key = keyspace(**fills)

        return key.get_value(field=key_field, tag=tag)

    return key_getter


# NOTE: This closure intentionally tries to look like a class.
def MaskField(**kwargs):
    """Create a new field for a :py:class:`~.KeyspacesRegion` that will write
    out a mask value from a keyspace.

    Parameters
    ----------
    field : string
        The name of the keyspace field to store the mask for.
    tag : string
        The name of the keyspace tag to store the mask for.

    Raises
    ------
    TypeError
        If both or neither field and tag are specified.

    Returns
    -------
    function
        A function which can be used in the `fields` argument to
        :py:class:`~.KeyspacesRegion` that will include a specified mask in the
        region data.
    """
    # Process the arguments
    field = kwargs.get("field")
    tag = kwargs.get("tag")

    # Create the field method
    if field is not None and tag is None:
        def mask_getter(keyspace, **kwargs):
            return keyspace.get_mask(field=field)

        return mask_getter
    elif tag is not None and field is None:
        def mask_getter(keyspace, **kwargs):
            return keyspace.get_mask(tag=tag)

        return mask_getter
    else:
        raise TypeError("MaskField expects 1 argument, "
                        "either 'field' or 'tag'.")
