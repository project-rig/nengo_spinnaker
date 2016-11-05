import numpy as np
from nengo_spinnaker.builder.transmission_parameters import (
    EnsembleTransmissionParameters,
    PassthroughNodeTransmissionParameters,
    NodeTransmissionParameters
)
import pytest


class TestPassthroughNodeTransmissionParameters(object):
    def test_equivalence_different_size_in(self):
        # With different size ins
        tp1 = PassthroughNodeTransmissionParameters(size_in=1,
                                                    size_out=3,
                                                    transform=1)
        tp2 = PassthroughNodeTransmissionParameters(size_in=2,
                                                    size_out=3,
                                                    transform=1)
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With the same size ins
        tp3 = PassthroughNodeTransmissionParameters(size_in=1,
                                                    size_out=3,
                                                    transform=1)
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_equivalence_different_size_out(self):
        # With different size outs
        tp1 = PassthroughNodeTransmissionParameters(size_in=1,
                                                    size_out=1,
                                                    transform=1)
        tp2 = PassthroughNodeTransmissionParameters(size_in=1,
                                                    size_out=2,
                                                    transform=1)
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With same size outs
        tp3 = PassthroughNodeTransmissionParameters(size_in=1,
                                                    size_out=1,
                                                    transform=1)
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_equivalence_slice_in(self):
        # With different slices
        tp1 = PassthroughNodeTransmissionParameters(
            size_in=3, size_out=2, transform=1, slice_in=slice(0, 2))
        tp2 = PassthroughNodeTransmissionParameters(
            size_in=3, size_out=2, transform=1, slice_in=slice(1, 3))
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With equivalent but differently expressed slices
        tp3 = PassthroughNodeTransmissionParameters(
            size_in=3, size_out=2, transform=1, slice_in=(0, 1))
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_equivalence_slice_out(self):
        # With different slices
        tp1 = PassthroughNodeTransmissionParameters(
            size_in=2, size_out=3, transform=1, slice_out=slice(0, 2))
        tp2 = PassthroughNodeTransmissionParameters(
            size_in=2, size_out=3, transform=1, slice_out=slice(1, 3))
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With equivalent but differently expressed slices
        tp3 = PassthroughNodeTransmissionParameters(
            size_in=2, size_out=3, transform=1, slice_out=(0, 1))
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_equivalence_transform(self):
        # With different transforms
        tp1 = PassthroughNodeTransmissionParameters(2, 2, np.ones((2, 2)))
        tp2 = PassthroughNodeTransmissionParameters(2, 2, np.zeros((2, 2)))
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With the same transforms
        tp3 = PassthroughNodeTransmissionParameters(2, 2, np.ones((2, 2)))
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_full_transform_slice_in(self):
        # Simple case with a pre-slice only
        tp = PassthroughNodeTransmissionParameters(
            size_in=2,
            size_out=2,
            transform=[[1.0], [-1.0]],
            slice_in=(1, )
        )

        # Check that we can extract a partial transform
        assert np.array_equal(
            tp.full_transform(),
            np.array([[1.0],
                      [-1.0]])
        )

        # Check that we can extract the full transform
        assert np.array_equal(
            tp.full_transform(slice_in=False),
            np.array([[0.0, 1.0],
                      [0.0, -1.0]])
        )

    def test_full_transform_slice_out(self):
        # Simple case with a post-slice only
        tp = PassthroughNodeTransmissionParameters(
            size_in=2,
            size_out=2,
            transform=[[1.0, -1.0]],
            slice_out=(1, )
        )

        # Check that we can extract a partial transform
        assert np.array_equal(
            tp.full_transform(),
            np.array([[1.0, -1.0]])
        )

        # Check that we can extract the full transform
        assert np.array_equal(
            tp.full_transform(slice_out=False),
            np.array([[0.0, 0.0],
                      [1.0, -1.0]])
        )

    def test_full_transform_with_scalar(self):
        # Simple case with a pre-slice, post-slice and a scalar transform
        tp = PassthroughNodeTransmissionParameters(
            size_in=2,
            size_out=2,
            transform=1.5
        )

        # Check that we can extract a full transform
        assert np.array_equal(
            tp.full_transform(slice_in=False, slice_out=False),
            np.eye(2) * 1.5
        )

    def test_full_transform_with_vector(self):
        # Simple case with a pre-slice, post-slice and a diagonal transform
        tp = PassthroughNodeTransmissionParameters(
            size_in=2,
            size_out=2,
            transform=[-1.0, -2.0]
        )

        # Check that we can extract a full transform
        assert np.array_equal(
            tp.full_transform(slice_in=False, slice_out=False),
            np.array([[-1.0, 0.0],
                      [0.0, -2.0]])
        )

    def test_concats(self):
        """Test that passthrough node connection parameters can be combined
        with later passthrough node connection parameters to build a new set of
        parameters.
        """
        # Check that these parameters are combined correctly
        a = PassthroughNodeTransmissionParameters(
            size_in=5, size_out=3, transform=2.0,
            slice_in=slice(2), slice_out=slice(2)
        )
        b = PassthroughNodeTransmissionParameters(
            size_in=3, size_out=3,
            slice_in=slice(2),
            slice_out=slice(1, 3),
            transform=[-1.0, 1.5]
        )

        # Combine the connections
        for c in a.concats([b]):
            # Check the new parameters
            assert c.size_in == a.size_in
            assert c.size_out == b.size_out
            assert np.array_equal(c.slice_in, a.slice_in)
            assert np.array_equal(c.slice_out, b.slice_out)

            assert np.array_equal(
                c.transform,
                np.array([[-1.0, 0.0],
                          [ 0.0, 1.5]]) * 2.0
            )

    def test_concats_no_connection(self):
        """Test that None is returned if concatenating connections results in
        an empty transform.
        """
        a = PassthroughNodeTransmissionParameters(
            size_in=4, size_out=16, slice_out=slice(4), transform=1.0)
        b = PassthroughNodeTransmissionParameters(
            size_in=16, size_out=4, slice_in=slice(4, 8), transform=1.0)

        # Combine the connections
        assert next(a.concats([b])) is None

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
            size_in=4, size_out=16, slice_out=slice(0, 4), transform=transform
        )

        assert tp.projects_to(slice(1))
        assert tp.projects_to(slice(5))
        assert not tp.projects_to(slice(4, 8))
        assert tp.projects_to((0, 1, 2, 4))

    def test_supports_global_inhibition(self):
        tp1 = PassthroughNodeTransmissionParameters(
            size_in=10, size_out=100, transform=np.ones((100, 10))
        )
        assert tp1.supports_global_inhibition

        tp2 = tp1.as_global_inhibition_connection
        assert tp2.size_in == tp1.size_in
        assert tp2.size_out == 1
        assert tp2.slice_out.size == 1
        assert np.array_equal(
            tp2.full_transform(1, 10),
            np.ones((1, 10))
        )


class TestEnsembleTransmissionParameters(object):
    def test_equivalence_decoders(self):
        """Parameters are only equivalent if they have the same decoders."""
        tp1 = EnsembleTransmissionParameters(np.ones((3, 100)), size_out=3)
        tp2 = EnsembleTransmissionParameters(np.zeros((3, 100)), size_out=3)
        assert tp1 != tp2

        tp3 = EnsembleTransmissionParameters(np.ones((3, 100)), size_out=3)
        assert hash(tp1) == hash(tp3)
        assert tp1 == tp3

    def test_equivalence_size_out(self):
        """Parameters are equivalent only if they have the same size out."""
        tp1 = EnsembleTransmissionParameters([[1]], size_out=3)
        tp2 = EnsembleTransmissionParameters([[1]], size_out=2)
        assert tp1 != tp2

        tp3 = EnsembleTransmissionParameters([[1]], size_out=3)
        assert hash(tp1) == hash(tp3)
        assert tp1 == tp3

    def test_equivalence_slice_out(self):
        """Parameters are equivalent only if they have the same output slice."""
        tp1 = EnsembleTransmissionParameters([[1]], size_out=3, slice_out=slice(None))
        tp2 = EnsembleTransmissionParameters([[1]], size_out=3, slice_out=(1, 2))
        assert tp1 != tp2

        tp3 = EnsembleTransmissionParameters([[1]], size_out=3, slice_out=slice(None))
        assert hash(tp1) == hash(tp3)
        assert tp1 == tp3

        tp4 = EnsembleTransmissionParameters([[1]], size_out=3, slice_out=(0, 1, 2))
        assert hash(tp1) == hash(tp4)
        assert tp1 == tp4

    def test_equivalence_size_out(self):
        """Parameters are equivalent only if they both have no learning rule."""
        tp1 = EnsembleTransmissionParameters([[1]], size_out=1, learning_rule=None)
        tp2 = EnsembleTransmissionParameters([[1]], size_out=1,
                                             learning_rule=object())
        assert tp1 != tp2

        tp3 = EnsembleTransmissionParameters([[1]], size_out=1)
        assert hash(tp1) == hash(tp3)
        assert tp1 == tp3

        tp4 = EnsembleTransmissionParameters([[1]], size_out=1,
                                             learning_rule=tp2.learning_rule)
        assert tp4 != tp2

    def test_full_transform(self):
        """Test that an expanded form of the transform/decoders can be
        extracted.
        """
        tp = EnsembleTransmissionParameters(
            decoders=np.zeros((3, 100)),
            transform=np.ones((5, 3)),
            size_out=6,
            slice_out=slice(0, 5)
        )

        # Extract the decoders with slicing included
        assert np.array_equal(tp.full_transform(), np.ones((5, 3)))

        # Extract with no slicing
        assert np.array_equal(
            tp.full_transform(slice_out=False),
            np.array([[1 for _ in range(3)] for _ in range(5)] +
                     [[0 for _ in range(3)]])
        )

    def test_concats_no_learning_rule(self):
        a = EnsembleTransmissionParameters(
            decoders=[[1.0, 2.0, 3.0, 4.0],
                      [4.0, 3.0, 2.0, 1.0]],
            size_out=4,
            slice_out=(1, 2)
        )
        b = PassthroughNodeTransmissionParameters(
            size_in=4, size_out=2, transform=1.0,
            slice_in=(1, 2)
        )

        # Combine the parameters
        for c in a.concats([b]):
            # Check the results
            assert isinstance(c, EnsembleTransmissionParameters)
            assert c.learning_rule is None
            assert c.size_out == b.size_out
            assert np.array_equal(c.slice_out, b.slice_out)
            assert np.array_equal(c.decoders, a.decoders)

    def test_concats_no_connection(self):
        a = EnsembleTransmissionParameters(
            decoders=[[1.0, 2.0, 3.0, 4.0],
                      [4.0, 3.0, 2.0, 1.0]],
            size_out=4,
            slice_out=(1, 2)
        )
        b = PassthroughNodeTransmissionParameters(
            size_in=4, size_out=2, transform=1.0,
            slice_in=(0, 3)
        )

        # Combine the parameters
        assert next(a.concats([b])) is None

    def test_concats_no_learning_rule(self):
        a = EnsembleTransmissionParameters(
            decoders=[[1.0, 2.0, 3.0, 4.0],
                      [4.0, 3.0, 2.0, 1.0]],
            size_out=2,
            learning_rule=object()
        )
        b = PassthroughNodeTransmissionParameters(
            size_in=2, size_out=2, transform=1.0
        )

        # Combine the parameters
        for c in a.concats([b]):
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
            size_out=16,
            slice_out=slice(4)
        )

        assert tp.projects_to(slice(1))
        assert tp.projects_to(slice(5))
        assert not tp.projects_to(slice(4, 8))
        assert tp.projects_to((0, 1, 2, 4))

    def test_global_inhibition(self):
        tp = EnsembleTransmissionParameters(
            decoders=np.random.normal(size=(10, 100)),
            size_out=200,
            transform=np.ones((200, 10))
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
        transform = np.array([[-1.0],
                              [ 1.0]])

        tp = EnsembleTransmissionParameters(decoders=decoders,
                                            size_out=2,
                                            transform=transform)

        assert np.array_equal(
            tp.full_decoders,
            np.array([[-0.5, -2.5, 0.3, -1.0],
                      [ 0.5,  2.5, -.3,  1.0]])
        )


class TestNodeTransmissionParameters(object):
    def test_equivalence_different_size_out(self):
        # With different size outs
        tp1 = NodeTransmissionParameters(1, size_out=1, transform=1)
        tp2 = NodeTransmissionParameters(1, size_out=2, transform=1)
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With same size outs
        tp3 = NodeTransmissionParameters(1, size_out=1, transform=1)
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_equivalence_pre_slice(self):
        # NOTE: Slices can't be hashed
        # With different slices
        tp1 = NodeTransmissionParameters(
            2, size_out=2, transform=1, pre_slice=slice(0, 2))
        tp2 = NodeTransmissionParameters(
            4, size_out=2, transform=1, pre_slice=slice(1, 3))
        assert tp1 != tp2

        # With equivalent but differently expressed slices
        tp3 = NodeTransmissionParameters(
            2, size_out=2, transform=1, pre_slice=slice(0, 2))
        assert tp1 == tp3

    def test_equivalence_slice_out(self):
        # With different slices
        tp1 = NodeTransmissionParameters(
            2, size_out=3, transform=1, slice_out=slice(0, 2))
        tp2 = NodeTransmissionParameters(
            2, size_out=3, transform=1, slice_out=slice(1, 3))
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With equivalent but differently expressed slices
        tp3 = NodeTransmissionParameters(
            2, size_out=3, transform=1, slice_out=(0, 1))
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_equivalence_transform(self):
        # With different transforms
        tp1 = NodeTransmissionParameters(2, 1, np.ones((2, 2)))
        tp2 = NodeTransmissionParameters(2, 1, np.zeros((2, 2)))
        assert tp1 != tp2
        assert hash(tp1) != hash(tp2)

        # With the same transforms
        tp3 = NodeTransmissionParameters(2, 1, np.ones((2, 2)))
        assert tp1 == tp3
        assert hash(tp1) == hash(tp3)

    def test_full_transform(self):
        """Test that an expanded form of the transform can be extracted.
        """
        tp = NodeTransmissionParameters(
            size_in=2,
            size_out=5,
            transform=[[1.0, 2.0],
                       [2.0, 3.0]],
            slice_out=slice(3, 5)
        )

        # Extract the decoders with slicing included
        assert np.array_equal(tp.full_transform(), tp.transform)

        # Extract with no slicing
        assert np.array_equal(
            tp.full_transform(slice_out=False),
            np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [0.0, 0.0],
                      [1.0, 2.0],
                      [2.0, 3.0]])
        )

    def test_concats(self):
        """Test concatenating Node transmission parameters with passthrough
        node transmission parameters.
        """
        a = NodeTransmissionParameters(size_in=2, size_out=5,
                                       transform=[[2.0, 0.0],
                                                  [0.0, 2.0]],
                                       slice_out=(1, 3),
                                       pre_slice=slice(0, 3),
                                       function=object())
        b = PassthroughNodeTransmissionParameters(
            size_in=5, size_out=2, transform=[0.5, 0.25],
            slice_in=(0, 3)
        )

        # Combine
        for c in a.concats([b]):
            assert c.size_in == a.size_in
            assert c.pre_slice == a.pre_slice
            assert c.function is a.function
            assert np.array_equal(
                c.transform,
                [[0.0, 0.0],
                 [0.0, 0.5]]
            )

    def test_concats_no_connection(self):
        a = NodeTransmissionParameters(size_in=2, size_out=5,
                                       transform=[[2.0, 0.0],
                                                  [0.0, 2.0]],
                                       slice_out=(1, 3),
                                       pre_slice=slice(0, 3),
                                       function=object())
        b = PassthroughNodeTransmissionParameters(
            size_in=5, size_out=2, transform=[0.5, 0.25],
            slice_in=(0, 2)
        )

        # Combine
        assert next(a.concats([b])) is None

    def test_projects_to(self):
        """Test that the parameters correctly report if they transmit any
        values to the dimensions listed.
        """
        tp = NodeTransmissionParameters(
            size_in=16,
            size_out=16,
            transform=1.0,
            slice_out=slice(4),
            pre_slice=slice(4)
        )

        assert tp.projects_to(slice(1))
        assert tp.projects_to(slice(5))
        assert not tp.projects_to(slice(4, 8))
        assert tp.projects_to((0, 1, 2, 4))

    def test_global_inhibition(self):
        tp = NodeTransmissionParameters(
            size_in=3, size_out=100, transform=np.ones((100, 3)),
            function=object()
        )
        assert tp.supports_global_inhibition

        tp2 = tp.as_global_inhibition_connection
        assert tp2.size_out == 1
        assert np.array_equal(
            tp2.full_transform(),
            np.ones((1, 3))
        )
