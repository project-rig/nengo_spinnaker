import enum
import numpy as np
import struct


class MatrixPartitioning(enum.IntEnum):
    rows = 0
    columns = 1


class NpIntFormatter(object):
    def __init__(self, dtype):
        self.dtype = dtype
        self.bytes_per_element = {
            np.uint8: 1, np.int8: 1,
            np.uint16: 2, np.int16: 2,
            np.uint32: 4, np.int32: 4,
        }[dtype]

    def __call__(self, matrix, **kwargs):
        return matrix.astype(dtype=self.dtype)


class MatrixRegion(object):
    """A region of memory which represents data from a matrix.

    The number of rows and columns may be prepended to the data as it is
    written out.

    Notes
    -----
    If the number of rows and columns are to be written out then they are
    always written in the order: rows, columns.  By default they are
    written as 4-byte values.

    See also
    --------
     - :py:class:`NpIntFormatter` formats matrix elements as integers.
     - :py:class:`rig.fixed_point.FixedPointFormatter` formats matrix
       elements as fixed point values.
    """
    def __init__(self, matrix, prepend_n_rows=False, prepend_n_columns=False,
                 formatter=NpIntFormatter(np.uint32), sliced_dimension=None):
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
        formatter : callable
            A formatter which will be applied to the NumPy matrix before
            writing the value out.  The formatter must accept calls with a
            NumPy matrix and must report as `bytes_per_element` the number of
            bytes used to store each formatted element.
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

        # Store the formatter
        self.formatter = formatter

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

        return (pp_size + self.matrix[self.expanded_slice(vertex_slice)].size *
                self.formatter.bytes_per_element)

    def write_subregion_to_file(self, vertex_slice, fp, **formatter_args):
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
        formatted = self.formatter(data, **formatter_args)
        fp.write(formatted.reshape((formatted.size, 1)).data)
