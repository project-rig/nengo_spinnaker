import enum
import numpy as np
import struct

from .region import Region


class MatrixPartitioning(enum.IntEnum):
    rows = 0
    columns = 1


class MatrixRegion(Region):
    """A region of memory which represents data from a matrix.

    The number of rows and columns may be prepended to the data as it is
    written out.

    Notes
    -----
    If the number of rows and columns are to be written out then they are
    always written in the order: rows, columns.
    """
    def __init__(self, matrix, prepend_n_rows=False, prepend_n_columns=False,
                 sliced_dimension=None):
        """Create a new region to represent a matrix data structure in memory.

        Parameters
        ----------
        matrix : :py:class:`numpy.ndarray`
            A matrix that will be stored in memory, or nothing to indicate that
            the data will be filled on SpiNNaker.  The matrix will be copied
            and made read-only, so provide the matrix as it is ready to go into
            memory.
        prepend_n_rows : bool
            Prepend the number of rows as a 4-byte integer to the matrix as it
            is written out in memory.
        prepend_n_columns : bool
            Prepend the number of columns as a 4-byte integer to the matrix as
            it is written out in memory.
        sliced_dimension : None or :py:class:`MatrixPartitioning` or int
            Indicates the dimension on which the matrix will be partitioned.
            None indicates no partitioning, 0 indicates partitioning of rows, 1
            of columns.  The :py:class:`MatrixPartitioning` enum can make for
            more readable code.
        """
        # Copy and store the matrix data
        self.matrix = np.copy(matrix)
        self.matrix.flags.writeable = False

        # Store the prepended values
        self.prepend_n_rows = prepend_n_rows
        self.prepend_n_columns = prepend_n_columns

        # Slicing
        assert sliced_dimension is None or sliced_dimension < self.matrix.ndim
        self.partition_index = sliced_dimension

    def expanded_slice(self, vertex_slice):
        if self.partition_index is None:
            return slice(None)

        return (
            tuple(slice(None) for _ in range(self.partition_index)) +
            (vertex_slice, ) +
            tuple(slice(None) for _ in range(self.partition_index + 1,
                                             self.matrix.ndim))
        )

    def sizeof(self, vertex_slice):
        """Get the size of a slice of this region in bytes.

        See :py:meth:`.region.Region.sizeof`
        """
        # Get the size of the prepends
        pp_size = 0
        if self.prepend_n_rows:
            pp_size += 4
        if self.prepend_n_columns:
            pp_size += 4

        return (pp_size +
                self.matrix[self.expanded_slice(vertex_slice)].nbytes)

    def write_subregion_to_file(self, fp, vertex_slice=slice(None),
                                **formatter_args):
        """Write the data contained in a portion of this region out to file.
        """
        # Partition the data
        data = self.matrix[self.expanded_slice(vertex_slice)]

        # Write the prepends
        if self.prepend_n_rows:
            fp.write(struct.pack('I', data.shape[0]))

        if self.prepend_n_columns:
            if self.matrix.ndim >= 2:
                fp.write(struct.pack('I', data.shape[1]))
            else:
                fp.write(struct.pack('I', 1))

        # Format the data and then write to file
        fp.write(data.tostring())
