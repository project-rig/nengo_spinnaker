import numpy as np
from nengo_spinnaker.builder.transmission_parameters import (
    Transform,
    EnsembleTransmissionParameters,
    PassthroughNodeTransmissionParameters,
    NodeTransmissionParameters
)
import pytest


class TestTransform(object):
    def test_equivalence_different_size_in(self):
        # With different size ins
        tp1 = Transform(size_in=1, size_out=3, transform=1)
        tp2 = Transform(size_in=2, size_out=3, transform=1)
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With the same size ins
        tp3 = Transform(size_in=1, size_out=3, transform=1)
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_equivalence_different_size_out(self):
        # With different size outs
        tp1 = Transform(size_in=1, size_out=1, transform=1)
        tp2 = Transform(size_in=1, size_out=2, transform=1)
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With same size outs
        tp3 = Transform(size_in=1, size_out=1, transform=1)
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_equivalence_slice_in(self):
        # With different slices
        tp1 = Transform(size_in=3, size_out=2, transform=1,
                        slice_in=slice(0, 2))
        tp2 = Transform(size_in=3, size_out=2, transform=1, 
                        slice_in=slice(1, 3))
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With equivalent but differently expressed slices
        tp3 = Transform(size_in=3, size_out=2, transform=1,
                        slice_in=(0, 1))
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_equivalence_slice_out(self):
        # With different slices
        tp1 = Transform(size_in=2, size_out=3, transform=1,
                        slice_out=slice(0, 2))
        tp2 = Transform(size_in=2, size_out=3, transform=1,
                        slice_out=slice(1, 3))
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With equivalent but differently expressed slices
        tp3 = Transform(size_in=2, size_out=3, transform=1,
                        slice_out=(0, 1))
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_equivalence_transform(self):
        # With different transforms
        tp1 = Transform(2, 2, np.ones((2, 2)))
        tp2 = Transform(2, 2, np.zeros((2, 2)))
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With the same transforms
        tp3 = Transform(2, 2, np.ones((2, 2)))
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_hash_with_sliced_input(self):
        tp = Transform(size_in=10, size_out=5, transform=1,
                       slice_in=slice(0, 10, 2))

        # Fails if the slicing is performed poorly
        hash(tp)

    def test_full_transform_scalar(self):
        """Test extracting the full transform when only a scalar is provided.
        """
        # Construct the expected matrix
        expected = np.zeros((4, 8))
        expected[np.array([1, 2, 3]), np.array([5, 6, 7])] = 1.5

        # Create the transform
        t = Transform(size_in=8, slice_in=slice(5, None),
                      transform=1.5,
                      size_out=4, slice_out=[1, 2, 3])

        # Check the full transform is correct
        assert np.array_equal(
            t.full_transform(slice_in=True, slice_out=False),
            expected[:, 5:]
        )

        assert np.array_equal(
            t.full_transform(slice_in=False, slice_out=True),
            expected[1:]
        )

        assert np.array_equal(
            t.full_transform(slice_in=False, slice_out=False),
            expected
        )

    def test_full_transform_vector(self):
        """Test extracting the full transform when only a vector is provided.
        """
        # Construct the expected matrix
        expected = np.zeros((4, 8))
        diag = np.array([1.0, 2.0, 3.0])
        expected[np.array([0, 1, 2]), np.array([3, 4, 5])] = diag

        # Create the transform
        t = Transform(size_in=8, slice_in=[3, 4, 5],
                      transform=diag,
                      size_out=4, slice_out=slice(3))

        # Check the full transform is correct
        assert np.array_equal(
            t.full_transform(slice_in=True, slice_out=False),
            expected[:, 3:6]
        )

        assert np.array_equal(
            t.full_transform(slice_in=False, slice_out=True),
            expected[:3]
        )

        assert np.array_equal(
            t.full_transform(slice_in=False, slice_out=False),
            expected
        )

    def test_full_transform_matrix(self):
        """Test extracting the full transform when a matrix is provided.
        """
        # Construct the expected matrix
        expected = np.zeros((4, 8))
        matrix = np.arange(9)
        matrix.shape = (3, 3)
        expected[:3, 3:6] = matrix

        # Create the transform
        t = Transform(size_in=8, slice_in=[3, 4, 5],
                      transform=matrix,
                      size_out=4, slice_out=slice(3))

        # Check the full transform is correct
        assert np.array_equal(
            t.full_transform(slice_in=True, slice_out=False),
            expected[:, 3:6]
        )

        assert np.array_equal(
            t.full_transform(slice_in=False, slice_out=True),
            expected[:3]
        )

        assert np.array_equal(
            t.full_transform(slice_in=False, slice_out=False),
            expected
        )

    @pytest.mark.parametrize(
        "a_params",
        [dict(slice_in=[1, 2, 3], transform=1, slice_out=[3, 4, 7]),  # Scalar
         dict(slice_in=[2, 3], transform=[2, 3], slice_out=[4, 7]),  # Vector
         dict(slice_in=[1, 2, 3], transform=np.arange(9).reshape((3, -1)),
              slice_out=[4, 6, 7])]  # Matrix
    )
    @pytest.mark.parametrize(
        "b_params",
        [dict(slice_in=[4, 5, 7], transform=2, slice_out=[0, 1, 2]),
         dict(slice_in=[4, 5, 7], transform=[1, 2, 3], slice_out=[0, 1, 2]),
         dict(slice_in=[4, 5, 7], transform=np.arange(9).reshape((3, -1)),
              slice_out=[0, 1, 3])]
    )
    def test_concat(self, a_params, b_params):
        # Build the transforms
        A = Transform(size_in=8, size_out=8, **a_params)
        B = Transform(size_in=8, size_out=4, **b_params)

        # Compute the expected combined transform
        expected = np.dot(B.full_transform(False, False),
                          A.full_transform(False, False))

        # Combine the transforms
        C = A.concat(B)

        # Test
        assert np.array_equal(expected, C.full_transform(False, False))

    def test_concat_empty_transform_mismatched_slicing(self):
        # Build the transforms
        A = Transform(size_in=1, transform=1, size_out=2, slice_out=[0])
        B = Transform(size_in=2, slice_in=[1], transform=1, size_out=1)

        # Check that the test is correct
        expected = np.dot(B.full_transform(False, False),
                          A.full_transform(False, False))
        assert not np.any(expected), "Test broken"

        # Combine the transforms, this should return None to indicate that the
        # transform is empty.
        assert A.concat(B) is None

    def test_concat_empty_transform_zero_transform_trivial(self):
        # Build the transforms
        A = Transform(size_in=2, transform=0, size_out=2)
        B = Transform(size_in=2, transform=1, size_out=2)

        # Check that the test is correct
        expected = np.dot(B.full_transform(False, False),
                          A.full_transform(False, False))
        assert not np.any(expected), "Test broken"

        # Combine the transforms, this should return None to indicate that the
        # transform is empty.
        assert A.concat(B) is None

    def test_concat_empty_transform_zero_transform(self):
        # Build the transforms
        transform_A = np.array([[1, -1], [-1, 1]])
        transform_B = np.array([[1, 1], [1, 1]])
        A = Transform(size_in=2, transform=transform_A, size_out=2)
        B = Transform(size_in=2, transform=transform_B, size_out=2)

        # Check that the test is correct
        expected = np.dot(B.full_transform(False, False),
                          A.full_transform(False, False))
        assert not np.any(expected), "Test broken"

        # Combine the transforms, this should return None to indicate that the
        # transform is empty.
        assert A.concat(B) is None


class TestPassthroughNodeTransmissionParameters(object):
    def test_concat(self):
        """Test that passthrough node connection parameters can be combined
        with later passthrough node connection parameters to build a new set of
        parameters.
        """
        # Check that these parameters are combined correctly
        a = PassthroughNodeTransmissionParameters(
                Transform(size_in=5, size_out=3, transform=2.0,
                          slice_in=slice(2), slice_out=slice(2))
        )
        b = PassthroughNodeTransmissionParameters(
                Transform(size_in=3, size_out=3, slice_in=slice(2),
                          slice_out=slice(1, 3), transform=[-1.0, 1.5])
        )

        # Combine the connections
        c = a.concat(b)

        # Check the new parameters
        assert c.size_in == a.size_in
        assert c.size_out == b.size_out
        assert np.array_equal(c.slice_in, a.slice_in)
        assert np.array_equal(c.slice_out, b.slice_out)

        assert np.array_equal(
            c.full_transform(False, False),
            np.dot(b.full_transform(False, False),
                   a.full_transform(False, False))
        )

    def test_concat_no_connection(self):
        """Test that None is returned if concatenating connections results in
        an empty transform.
        """
        a = PassthroughNodeTransmissionParameters(
                Transform(size_in=4, size_out=16, slice_out=slice(4),
                          transform=1.0)
        )
        b = PassthroughNodeTransmissionParameters(
                Transform(size_in=16, size_out=4, slice_in=slice(4, 8),
                          transform=1.0)
        )

        # Combine the connections
        assert a.concat(b) is None

    @pytest.mark.parametrize("method", ("scalar", "vector", "matrix"))
    def test_projects_to(self, method):
        """Test that the parameters correctly report if they transmit any
        values to the dimensions listed.
        """
        if method == "scalar":
            transform = 1.0
        elif method == "vector":
            transform = [1.0, 1.0, 1.0, 1.0]
        else:
            transform = np.eye(4)

        tp = PassthroughNodeTransmissionParameters(
                Transform(size_in=4, size_out=16, slice_out=slice(0, 4),
                          transform=transform)
        )

        assert tp.projects_to(slice(1))
        assert tp.projects_to(slice(5))
        assert not tp.projects_to(slice(4, 8))
        assert tp.projects_to((0, 1, 2, 4))

    def test_supports_global_inhibition(self):
        tp1 = PassthroughNodeTransmissionParameters(
                Transform(size_in=10, size_out=100,
                          transform=np.ones((100, 10)))
        )
        assert tp1.supports_global_inhibition

        tp2 = tp1.as_global_inhibition_connection
        assert tp2.size_in == tp1.size_in
        assert tp2.size_out == 1
        assert tp2.slice_out.size == 1
        assert np.array_equal(
            tp2.full_transform(),
            np.ones((1, 10))
        )


class TestEnsembleTransmissionParameters(object):
    def test_equivalence_decoders(self):
        """Parameters are only equivalent if they have the same decoders."""
        transform = Transform(1, 1, 1)
        tp1 = EnsembleTransmissionParameters(np.ones((3, 100)), transform)
        tp2 = EnsembleTransmissionParameters(np.zeros((3, 100)), transform)
        assert tp1 != tp2

        tp3 = EnsembleTransmissionParameters(np.ones((3, 100)), transform)
        assert hash(tp1) == hash(tp3)
        assert tp1 == tp3

    def test_equivalence_learning_rule(self):
        """Parameters are equivalent only if they both have no learning rule."""
        t = Transform(1, 1, 1)
        tp1 = EnsembleTransmissionParameters([[1]], t, learning_rule=None)
        tp2 = EnsembleTransmissionParameters([[1]], t, learning_rule=object())
        assert tp1 != tp2

        tp3 = EnsembleTransmissionParameters([[1]], t)
        assert hash(tp1) == hash(tp3)
        assert tp1 == tp3

        tp4 = EnsembleTransmissionParameters([[1]], t,
                                             learning_rule=tp2.learning_rule)
        assert tp4 != tp2

    def test_concat_no_learning_rule(self):
        a = EnsembleTransmissionParameters(
            decoders=[[1.0, 2.0, 3.0, 4.0],
                      [4.0, 3.0, 2.0, 1.0]],
            transform=Transform(2, 4, 1, slice_out=(1, 2))
        )
        b = PassthroughNodeTransmissionParameters(
                Transform(size_in=4, size_out=2, transform=1.0,
                          slice_in=(1, 2))
        )

        # Combine the parameters
        c = a.concat(b)

        # Check the results
        assert isinstance(c, EnsembleTransmissionParameters)
        assert c.learning_rule is None
        assert c.size_out == b.size_out
        assert np.array_equal(c.slice_out, b.slice_out)
        assert np.array_equal(c.decoders, a.decoders)

    def test_concat_no_connection(self):
        a = EnsembleTransmissionParameters(
            decoders=[[1.0, 2.0, 3.0, 4.0],
                      [4.0, 3.0, 2.0, 1.0]],
            transform=Transform(2, 4, 1, slice_out=(1, 2))
        )
        b = PassthroughNodeTransmissionParameters(
                Transform(size_in=4, size_out=2, transform=1.0,
                          slice_in=(0, 3))
        )

        # Combine the parameters
        assert a.concat(b) is None

    def test_concat_learning_rule(self):
        a = EnsembleTransmissionParameters(
            decoders=[[1.0, 2.0, 3.0, 4.0],
                      [4.0, 3.0, 2.0, 1.0]],
            transform=Transform(2, 4, 1, slice_out=(1, 2)),
            learning_rule=object()
        )
        b = PassthroughNodeTransmissionParameters(
                Transform(size_in=4, size_out=2, transform=1.0,
                          slice_in=(1, 2))
        )

        # Combine the parameters
        c = a.concat(b)

        # Check the results
        assert isinstance(c, EnsembleTransmissionParameters)
        assert c.learning_rule is a.learning_rule
        assert c.size_out == b.size_out
        assert np.array_equal(c.slice_out, b.slice_out)
        assert np.array_equal(c.decoders, a.decoders)

    def test_projects_to(self):
        """Test that the parameters correctly report if they transmit any
        values to the dimensions listed.
        """
        tp = EnsembleTransmissionParameters(
            decoders=np.ones((1, 4)),
            transform=Transform(size_in=1, size_out=16, transform=1.0,
                                slice_out=slice(4))
        )

        assert tp.projects_to(slice(1))
        assert tp.projects_to(slice(5))
        assert not tp.projects_to(slice(4, 8))
        assert tp.projects_to((0, 1, 2, 4))

    def test_global_inhibition(self):
        tp = EnsembleTransmissionParameters(
            decoders=np.random.normal(size=(10, 100)),
            transform=Transform(size_in=10, size_out=200,
                                transform=np.ones((200, 10)))
        )
        assert tp.supports_global_inhibition

        tp2 = tp.as_global_inhibition_connection
        assert tp2.size_out == 1 and tp2.slice_out.size == 1
        assert np.array_equal(
            tp2.full_transform(),
            np.ones((1, 10))
        )

    def test_full_decoders(self):
        """Test that the decoders and transform are combined correctly."""
        decoders = np.array([[0.5, 2.5, -.3, 1.0]])
        transform = Transform(size_in=1, size_out=2,
                              transform=[[-1.0], [ 1.0]])

        tp = EnsembleTransmissionParameters(decoders=decoders,
                                            transform=transform)

        assert np.array_equal(
            tp.full_decoders,
            np.array([[-0.5, -2.5, 0.3, -1.0],
                      [ 0.5,  2.5, -.3,  1.0]])
        )


class TestNodeTransmissionParameters(object):
    def test_equivalence_pre_slice(self):
        # NOTE: Slices can't be hashed
        # With different slices
        t = Transform(2, 2, 1)
        tp1 = NodeTransmissionParameters(t, pre_slice=slice(0, 2))
        tp2 = NodeTransmissionParameters(t, pre_slice=slice(1, 3))
        assert tp1 != tp2

        # With equivalent but differently expressed slices
        tp3 = NodeTransmissionParameters(t, pre_slice=slice(0, 2))
        assert tp1 == tp3

    def test_equivalence_transform(self):
        # With different transforms
        tp1 = NodeTransmissionParameters(Transform(2, 1, np.ones((2, 2))))
        tp2 = NodeTransmissionParameters(Transform(2, 1, np.zeros((2, 2))))
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With the same transforms
        tp3 = NodeTransmissionParameters(Transform(2, 1, np.ones((2, 2))))
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_concat(self):
        """Test concatenating Node transmission parameters with passthrough
        node transmission parameters.
        """
        a = NodeTransmissionParameters(
            Transform(size_in=2, size_out=5,
                      transform=[[2.0, 0.0], [0.0, 2.0]],
                      slice_out=(1, 3)),
            pre_slice=slice(0, 3),
            function=object()
        )
        b = PassthroughNodeTransmissionParameters(
                Transform(size_in=5, size_out=2, transform=[0.5, 0.25],
                          slice_in=(0, 3))
        )

        # Combine
        c = a.concat(b)
        assert c.size_in == a.size_in
        assert c.pre_slice == a.pre_slice
        assert c.function is a.function
        assert np.array_equal(
            c.full_transform(False, False),
            np.dot(b.full_transform(False, False),
                   a.full_transform(False, False))
        )

    def test_concat_no_connection(self):
        a = NodeTransmissionParameters(
                Transform(size_in=2, size_out=5,
                          transform=[[2.0, 0.0], [0.0, 2.0]],
                          slice_out=(1, 3)),
                pre_slice=slice(0, 3),
                function=object())
        b = PassthroughNodeTransmissionParameters(
                Transform(size_in=5, size_out=2, transform=[0.5, 0.25],
                          slice_in=(0, 2))
        )

        # Combine
        assert a.concat(b) is None

    def test_global_inhibition(self):
        tp = NodeTransmissionParameters(
                Transform(size_in=3, size_out=100,
                          transform=np.ones((100, 3))),
            pre_slice=slice(0, 100),
            function=object()
        )
        assert tp.supports_global_inhibition

        tp2 = tp.as_global_inhibition_connection
        assert tp2.function is tp.function
        assert tp2.pre_slice == tp.pre_slice
        assert tp2.size_out == 1
        assert np.array_equal(
            tp2.full_transform(),
            np.ones((1, 3))
        )
