import numpy as np
import pytest
import struct
import tempfile

from nengo_spinnaker.regions.matrix import (
    MatrixRegion, NpIntFormatter, MatrixPartitioning)


class TestNpIntFormatter(object):
    @pytest.mark.parametrize(
        "dtype, bytes_per_element",
        [(np.int8, 1),
         (np.uint8, 1),
         (np.int16, 2),
         (np.uint16, 2),
         (np.int32, 4),
         (np.uint32, 4),
         ])
    def test_correct(self, dtype, bytes_per_element):
        f = NpIntFormatter(dtype)
        assert f.bytes_per_element == bytes_per_element
        assert f.dtype == dtype
        assert f(np.zeros(100, dtype=float)).dtype == dtype


class TestMatrixRegion(object):
    @pytest.mark.parametrize("formatter", [NpIntFormatter(np.uint8),
                                           NpIntFormatter(np.uint16),
                                           NpIntFormatter(np.uint32)])
    @pytest.mark.parametrize(
        "prepend_n_rows, prepend_n_cols, n_prepend_bytes",
        [(False, False, 0),
         (True, False, 4),
         (False, True, 4),
         (True, True, 8)
         ])
    @pytest.mark.parametrize(
        "matrix, partition, sliced_dimension",
        [(np.zeros((4, 3)), slice(0, 4), None),
         (np.zeros((4, 3)), slice(0, 2), MatrixPartitioning.rows),
         (np.ones((4, 5)), slice(0, 2), MatrixPartitioning.columns),
         (np.zeros(100), slice(1), 0),
         (np.zeros((4, 3, 5)), slice(1), 2),
         ])
    def test_sizeof(self, matrix, prepend_n_rows, prepend_n_cols,
                    n_prepend_bytes, formatter, partition, sliced_dimension):
        # Create the matrix region
        mr = MatrixRegion(
            matrix, prepend_n_rows=prepend_n_rows,
            prepend_n_columns=prepend_n_cols, formatter=formatter,
            sliced_dimension=sliced_dimension
        )

        # Get the partitioned matrix
        if sliced_dimension is None:
            slices = slice(None)
        else:
            slices = (
                tuple(slice(None) for _ in range(sliced_dimension)) +
                (partition, ) +
                tuple(slice(None) for _ in range(sliced_dimension + 1,
                                                 matrix.ndim))
            )

        # Check the size is correct
        assert (
            mr.sizeof(partition) ==
            n_prepend_bytes + formatter.bytes_per_element * matrix[slices].size
        )

    @pytest.mark.parametrize(
        "prepend_n_rows, prepend_n_cols",
        [(False, False),
         (True, False),
         (False, True),
         (True, True)])
    @pytest.mark.parametrize(
        "matrix, partition, sliced_dimension",
        [(np.eye(4), slice(0, 2), MatrixPartitioning.rows),
         (np.eye(5), slice(1, 3), MatrixPartitioning.rows),
         ])
    @pytest.mark.parametrize(
        "formatter",
        [NpIntFormatter(np.uint8),
         NpIntFormatter(np.int8),
         NpIntFormatter(np.uint16),
         NpIntFormatter(np.int16),
         NpIntFormatter(np.uint32),
         NpIntFormatter(np.int32),
         ])
    def test_write_subregion_to_file(self, matrix,
                                     prepend_n_rows, prepend_n_cols,
                                     formatter, partition, sliced_dimension):
        # Create the matrix region
        mr = MatrixRegion(matrix, prepend_n_rows=prepend_n_rows,
                          prepend_n_columns=prepend_n_cols,
                          formatter=formatter,
                          sliced_dimension=sliced_dimension)

        partitioned_matrix = matrix[mr.expanded_slice(partition)]

        # Get the temporary file
        fp = tempfile.TemporaryFile()

        # Write the subregion
        mr.write_subregion_to_file(partition, fp)

        # Read and check
        # Check the prepended data
        fp.seek(0)
        if prepend_n_rows:
            n_rows = fp.read(4)
            assert (partitioned_matrix.shape[0] ==
                    struct.unpack('I', n_rows)[0])

        if prepend_n_cols:
            n_cols = fp.read(4)
            assert (partitioned_matrix.shape[1] ==
                    struct.unpack('I', n_cols)[0])

        # Check the actual data
        data = np.frombuffer(fp.read(), dtype=formatter.dtype).reshape(
            partitioned_matrix.shape)
        assert np.all(data == formatter(partitioned_matrix))

    def test_write_subregion_to_file_with_1d_array(self):
        matrix = np.ones(100)
        mr = MatrixRegion(matrix, True, True)

        # Get the temporary file
        fp = tempfile.TemporaryFile()

        # Write the subregion
        mr.write_subregion_to_file(slice(1, 1), fp)

        # Read and check
        # Check the prepended data
        fp.seek(0)
        n_rows = fp.read(4)
        assert 100 == struct.unpack('I', n_rows)[0]

        n_cols = fp.read(4)
        assert 1 == struct.unpack('I', n_cols)[0]

        # Check the actual data
        formatter = mr.formatter
        data = np.frombuffer(fp.read(), dtype=formatter.dtype).reshape(
            matrix.shape)
        assert np.all(data == formatter(matrix))

    def test_locks_matrix(self):
        """Check that the data stored in the region is copied and not editable.
        """
        # Create the data and the matrix region
        data = np.zeros((2, 3))
        mr = MatrixRegion(data)

        # Assert that writing to data doesn't modify the region data
        data[0][1] = 2.
        assert not np.all(data == np.zeros((2, 3)))
        assert np.all(mr.matrix == np.zeros((2, 3)))

        # Assert the region data can't be written to directly
        with pytest.raises(ValueError):
            mr.matrix[0][0] = 3.
        assert mr.matrix.flags.writeable is False
