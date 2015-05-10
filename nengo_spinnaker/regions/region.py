class Region(object):
    """Represents a region of memory."""
    def sizeof(self, vertex_slice=None):  # pragma : no cover
        """Get the size of the region in bytes."""
        raise NotImplementedError

    def sizeof_padded(self, vertex_slice=None):
        """Get the size of the region in bytes when padded to take an integral
        number of words.
        """
        # Call to get the number of bytes, then pad if necessary and return
        n_bytes = self.sizeof(vertex_slice)
        return n_bytes + (0 if (n_bytes % 4 == 0) else (4 - n_bytes % 4))

    def write_subregion_to_file(self, fp, vertex_slice=None,
                                **kwargs):  # pragma : no cover
        """Write the region or a slice of the region to a file-like object.

        Parameters
        ----------
        fp : file-like
            The file-like object to write to.

        Other Parameters
        ----------------
        vertex_slice : :py:class:`slice` or None
            The slice of the region to write out.
        """
        raise NotImplementedError
