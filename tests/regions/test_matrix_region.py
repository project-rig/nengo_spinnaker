import numpy as np
import pytest
import struct
import tempfile

from nengo_spinnaker.regions.matrix import (
    MatrixRegion, MatrixPartitioning)


class TestMatrixRegion(object):
    @pytest.mark.parametrize("dtype", [np.uint8, np.int16])
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
                    n_prepend_bytes, partition, sliced_dimension, dtype):
        matrix = np.asarray(matrix, dtype=dtype)

        # Create the matrix region
        mr = MatrixRegion(
            matrix, prepend_n_rows=prepend_n_rows,
            prepend_n_columns=prepend_n_cols,
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
        assert (mr.sizeof(partition) ==
                n_prepend_bytes + matrix[slices].data.nbytes)

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
    def test_write_subregion_to_file(self, matrix,
                                     prepend_n_rows, prepend_n_cols,
                                     partition, sliced_dimension):
        # Create the matrix region
        mr = MatrixRegion(matrix, prepend_n_rows=prepend_n_rows,
                          prepend_n_columns=prepend_n_cols,
                          sliced_dimension=sliced_dimension)

        partitioned_matrix = matrix[mr.expanded_slice(partition)]

        # Get the temporary file
        fp = tempfile.TemporaryFile()

        # Write the subregion
        mr.write_subregion_to_file(fp, partition)

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
        data = np.frombuffer(fp.read(), dtype=matrix.dtype).reshape(
            partitioned_matrix.shape)
        assert np.all(data == partitioned_matrix)

    def test_write_subregion_to_file_with_1d_array(self):
        matrix = np.ones(100)
        mr = MatrixRegion(matrix, True, True)

        # Get the temporary file
        fp = tempfile.TemporaryFile()

        # Write the subregion
        mr.write_subregion_to_file(fp, slice(1, 1))

        # Read and check
        # Check the prepended data
        fp.seek(0)
        n_rows = fp.read(4)
        assert 100 == struct.unpack('I', n_rows)[0]

        n_cols = fp.read(4)
        assert 1 == struct.unpack('I', n_cols)[0]

        # Check the actual data
        data = np.frombuffer(fp.read(), dtype=matrix.dtype).reshape(
            matrix.shape)
        assert np.all(data == matrix)
