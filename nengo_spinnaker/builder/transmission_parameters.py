"""Transmission Parameters

A transmission parameter object contains the Nengo specific description of the
values that will be produced by SpiNNaker core(s) [this is in contrast to the
SpiNNaker-specific information required to route those values across the
network as multicast packets]. Each type of transmission parameter must be
*equatable* (it must have both `__ne__` and `__eq__` defined) and *hashable*
(it must have `__hash__` defined).

Moreover, each transmission parameter type must have a method called `concat`
which accepts a `PassthroughNodeTransmissionParameters` as an argument and
returns new transmission parameters representing the result of chaining the
first parameter with the other. Several sample implementations of this method
are presented within this document.

Each parameter must have a method called `projects_to` which accepts a valid
slice of dimensions and returns a boolean indicating whether any non-zero
values would be delivered to this slice of the output space. It is acceptable
to return True and to then not transmit to this space but it is not permissible
to return False and then transmit to this space.

Parameters must have a property called `supports_global_inhibition` which
indicates that every row of the transform they represent is equivalent. This
can be used to reduce the number of packets that is required to suppress the
firing of some neurons. An additional property called
`as_global_inhibition_connection` is required to return a modified version of
the parameters with a size out of `1` and containing only one row of the
transform.

Finally, each parameter type must have a method called `full_transform` which
allows the unsliced transform/decoder matrix to be extracted from the
parameter.
"""
import numpy as np

try:
    from xxhash import xxh64 as fasthash
except ImportError:  # pragma: no cover
    from hashlib import md5 as fasthash
    import warnings
    warnings.warn("xxhash not installed, falling back to md5. "
                  "Install xxhash to improve build performance.", UserWarning)


class Transform(object):
    __slots__ = ["size_in", "size_out", "transform", "slice_in", "slice_out"]

    def __init__(self, size_in, size_out, transform,
                 slice_in=slice(None), slice_out=slice(None)):
        self.size_in = size_in
        self.size_out = size_out

        # Transform the slices into an appropriate format
        self.slice_in = Transform._get_slice_as_ndarray(slice_in, size_in)
        self.slice_out = Transform._get_slice_as_ndarray(slice_out, size_out)

        # Copy the transform into a C-contiguous, read-only form
        self.transform = np.array(transform, order='C')
        self.transform.flags["WRITEABLE"] = False

    @staticmethod
    def _get_slice_as_ndarray(sl, size):
        """Return a slice as a read-only Numpy array."""
        if isinstance(sl, slice):
            sl = np.array(range(size)[sl])
        else:
            sl = np.array(sorted(set(sl)))

        sl.flags["WRITEABLE"] = False

        return sl

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        return (self.size_in == other.size_in and
                self.size_out == other.size_out and
                np.array_equal(self.slice_in, other.slice_in) and
                np.array_equal(self.slice_out, other.slice_out) and
                np.array_equal(self.transform, other.transform))

    def __hash__(self):
        # The hash is combination of all the elements of the tuple, but we use
        # a faster hashing mechanism to hash the array types.
        return hash((self.size_in,
                     self.size_out,
                     fasthash(self.slice_in).hexdigest(),
                     fasthash(self.slice_out).hexdigest(),
                     fasthash(self.transform).hexdigest()))

    def full_transform(self, slice_in=True, slice_out=True):
        """Get an expanded form of the transform."""
        # Determine the shape of the resulting matrix
        size_in = len(self.slice_in) if slice_in else self.size_in
        size_out = len(self.slice_out) if slice_out else self.size_out

        # Get the slices
        columns = np.arange(size_in) if slice_in else np.array(self.slice_in)
        rows = np.arange(size_out) if slice_out else np.array(self.slice_out)

        # Prepare the transform
        transform = np.zeros((size_out, size_in))

        if self.transform.ndim < 2:
            # For vectors and scalars
            transform[rows, columns] = self.transform
        elif self.transform.ndim == 2:
            # For matrices
            rows_transform = np.zeros_like(transform[rows, :])
            rows_transform[:, columns] = self.transform
            transform[rows] = rows_transform
        else:  # pragma: no cover
            raise NotImplementedError

        return transform

    def projects_to(self, space):
        """Indicate whether the output of the connection described by the
        connection will intersect with the specified range of dimensions.
        """
        space = set(Transform._get_slice_as_ndarray(space, self.size_out))

        if self.transform.ndim == 0:
            outputs = set(self.slice_out)
        elif self.transform.ndim == 1:
            outputs = set(self.slice_out[self.transform != 0])
        elif self.transform.ndim == 2:
            outputs = set(self.slice_out[np.any(self.transform != 0, axis=1)])
        else:  # pragma: no cover
            raise NotImplementedError

        return len(outputs.intersection(space)) != 0

    def concat(a, b):
        """Return a transform which is the result of concatenating this
        transform with another.
        """
        assert a.size_out == b.size_in

        # Determine where the output dimensions of this transform and the input
        # dimensions of the other intersect.
        out_sel = np.zeros(a.size_out, dtype=bool)
        out_sel[a.slice_out] = True

        in_sel = np.zeros(b.size_in, dtype=bool)
        in_sel[b.slice_in] = True

        mid_sel = np.logical_and(out_sel, in_sel)

        # If the slices do not intersect at all then return None to indicate
        # that the connection will be empty.
        if not np.any(mid_sel):
            return None

        # If the first transform is specified with either a scalar or a vector
        # (as a diagonal) then the slice in is modified by `mid_sel'.
        slice_in_sel = mid_sel[a.slice_out]
        if a.transform.ndim < 2:
            # Get the new slice in
            slice_in = a.slice_in[slice_in_sel]

            # Get the new transform
            if a.transform.ndim == 0:
                a_transform = a.transform
            else:
                a_transform = a.transform[slice_in_sel]
        else:
            # The slice in remains the same but the rows of the transform are
            # sliced.
            slice_in = a.slice_in
            a_transform = a.transform[slice_in_sel]

        # If the second transform is specified with either a scalar or a vector
        # (as a diagonal) then the output slice is modified by `mid_sel'
        slice_out_sel = mid_sel[b.slice_in]
        if b.transform.ndim < 2:
            # Get the new slice out
            slice_out = b.slice_out[slice_out_sel]

            # Get the new transform
            if b.transform.ndim == 0:
                b_transform = b.transform
            else:
                b_transform = b.transform[slice_out_sel]
        else:
            # The slice out remains the same but the columns of the transform
            # are sliced.
            slice_out = b.slice_out
            b_transform = b.transform[:, slice_out_sel]

        # Multiply the transforms together
        if a_transform.ndim < 2 or b_transform.ndim == 0:
            new_transform = b_transform * a_transform
        elif b_transform.ndim == 1:
            new_transform = (b_transform * a_transform.T).T
        else:
            new_transform = np.dot(b_transform, a_transform)

        # If the transform is filled with zeros then return None
        if not np.any(new_transform != 0.0):
            return None

        # Create the new Transform
        return Transform(size_in=a.size_in, slice_in=slice_in,
                         transform=new_transform,
                         size_out=b.size_out, slice_out=slice_out)

    def hstack(self, other):
        """Create a new transform as the result of stacking this transform with
        another.
        """
        if self.size_out != other.size_out:
            raise ValueError(
                "Cannot horizontally stack two transforms with different "
                "output sizes ({} and {})".format(
                    self.size_out, other.size_out)
            )

        # Compute the new input size and the new input slice
        size_in = self.size_in + other.size_in
        slice_in = np.hstack((self.slice_in, other.slice_in + self.size_in))

        # Determine which rows must be contained in the output matrix.
        slice_out = np.union1d(self.slice_out, other.slice_out)

        # Construct the new matrix
        n_rows = len(slice_out)
        n_cols = len(slice_in)
        matrix = np.zeros((n_rows, n_cols))

        # Write in the elements from ourself, and then the elements from the
        # other matrix.
        offset = 0  # Used to perform the stacking
        for t in (self, other):
            # Select the rows which should be written
            selected_rows = np.array([i in t.slice_out for i in slice_out])
            rows = np.arange(n_rows)[selected_rows]

            # Select the columns to be written, note that the offset is used
            # for stacking.
            n_cols = len(t.slice_in)
            cols = np.arange(offset, offset + n_cols)
            offset += n_cols

            if t.transform.ndim < 2:
                # If the transform was specified as either a scalar or a
                # diagonal.
                matrix[rows, cols] = t.transform
            elif t.transform.ndim == 2:
                # If the transform is a matrix
                rows_transform = np.zeros_like(matrix[rows, :])
                rows_transform[:, cols] = t.transform
                matrix[rows] += rows_transform
            else:  # pragma: no cover
                raise NotImplementedError

        # Return the new transform
        return Transform(size_in, self.size_out, matrix, slice_in, slice_out)


class TransmissionParameters(object):
    __slots__ = ["_transform"]

    def __init__(self, transform):
        # Store the transform
        self._transform = transform

    def __getattr__(self, attr):
        # Forward the request to the transform
        return getattr(self._transform, attr)

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        return (type(self) is type(other) and
                self._transform == other._transform)

    def __hash__(self):
        return hash((type(self), self._transform))

    @property
    def supports_global_inhibition(self):
        """Indicates whether this transform supports being optimised out as a
        global inhibition connection.
        """
        # True iff. every row of the transform is the same
        transform = self.full_transform(slice_out=False)
        return np.all(transform[0, :] == transform[1:, :])

    @property
    def as_global_inhibition_connection(self):  # pragma: no cover
        raise NotImplementedError


class PassthroughNodeTransmissionParameters(TransmissionParameters):
    """Parameters describing information transmitted by a passthrough node.
    """
    def concat(self, other):
        """Create new connection parameters which are the result of
        concatenating this connection several others.

        Parameters
        ----------
        other : PassthroughNodeTransmissionParameters
            Connection parameters to add to the end of this connection.

        Returns
        -------
        PassthroughNodeTransmissionParameters or None
            Either a new set of transmission parameters, or None if the
            resulting transform contained no non-zero values.
        """
        # Combine the transforms
        new_transform = self._transform.concat(other._transform)

        # Create a new connection (unless the resulting transform is empty,
        # in which case don't)
        if new_transform is not None:
            return PassthroughNodeTransmissionParameters(new_transform)
        else:
            # The transform consisted entirely of zeros so return None.
            return None

    def hstack(self, *others):
        """Create new connection parameters which are the result of stacking
        these connection parameters with other connection parameters.

        Parameters
        ----------
        *others : PassthroughNodeTransmissionParameters
            Additional connection parameters to stack against these parameters.

        Returns
        -------
        PassthroughNodeTransmissionParameters
            A new set of transmission parameters resulting from stacking the
            provided parameters together.
        """
        # Horizontally stack the parameters
        stacked_transform = self._transform
        for other in others:
            stacked_transform = stacked_transform.hstack(other._transform)

        # Create and return the new connection
        return PassthroughNodeTransmissionParameters(stacked_transform)

    @property
    def as_global_inhibition_connection(self):
        """Construct a copy of the connection with the optimisation for global
        inhibition applied.
        """
        assert self.supports_global_inhibition
        transform = self.full_transform(slice_out=False)[0, :]

        return PassthroughNodeTransmissionParameters(
            Transform(size_in=self.size_in, size_out=1, transform=transform,
                      slice_in=self.slice_in)
        )


class EnsembleTransmissionParameters(TransmissionParameters):
    """Parameters describing information transmitted by an ensemble.

    Attributes
    ----------
    decoders : ndarray
        A matrix describing a decoding of the ensemble (sized N x D).
    learning_rule :
        Learning rule associated with the decoding.
    """
    __slots__ = TransmissionParameters.__slots__ + [
        "decoders", "learning_rule"
    ]

    def __init__(self, decoders, transform, learning_rule=None):
        super(EnsembleTransmissionParameters, self).__init__(transform)

        # Copy the decoders into a C-contiguous, read-only array
        self.decoders = np.array(decoders, order='C')
        self.decoders.flags["WRITEABLE"] = False

        # Store the learning rule
        self.learning_rule = learning_rule

    def __eq__(self, other):
        # Two parameters are equal only if they are of the same type, both have
        # no learning rule and are equivalent in all other fields.
        return (super(EnsembleTransmissionParameters, self).__eq__(other) and
                np.array_equal(self.decoders, other.decoders) and
                self.learning_rule is None and
                other.learning_rule is None)

    def __hash__(self):
        return hash((type(self), self.learning_rule, self._transform,
                     fasthash(self.decoders).hexdigest()))

    def concat(self, other):
        """Create new connection parameters which are the result of
        concatenating this connection with others.

        Parameters
        ----------
        other : PassthroughNodeTransmissionParameters
            Connection parameters to add to the end of this connection.

        Returns
        -------
        EnsembleTransmissionParameters or None
            Either a new set of transmission parameters, or None if the
            resulting transform contained no non-zero values.
        """
        # Get the outgoing transformation
        new_transform = self._transform.concat(other._transform)

        # Create a new connection (unless the resulting transform is empty,
        # in which case don't)
        if new_transform is not None:
            return EnsembleTransmissionParameters(
                self.decoders, new_transform, self.learning_rule
            )
        else:
            # The transform consisted entirely of zeros so return None.
            return None

    @property
    def as_global_inhibition_connection(self):
        """Construct a copy of the connection with the optimisation for global
        inhibition applied.
        """
        assert self.supports_global_inhibition
        transform = self.full_transform(slice_out=False)[0, :]

        return EnsembleTransmissionParameters(
            self.decoders,
            Transform(size_in=self.decoders.shape[0], size_out=1,
                      transform=transform, slice_in=self._transform.slice_in)
        )

    @property
    def full_decoders(self):
        """Get the matrix corresponding to a combination of the decoders and
        the transform applied by the connection.
        """
        return np.dot(self.full_transform(slice_in=False, slice_out=False),
                      self.decoders)


class NodeTransmissionParameters(TransmissionParameters):
    __slots__ = TransmissionParameters.__slots__ + ["pre_slice", "function"]

    def __init__(self, transform, pre_slice=slice(None), function=None):
        """
        Parameters
        ----------
        size_in : int
            Either the size out of the transmitting object (if the function is
            `None`) or the size out of the function.
        """
        super(NodeTransmissionParameters, self).__init__(transform)
        self.pre_slice = pre_slice
        self.function = function

    def __eq__(self, other):
        return (super(NodeTransmissionParameters, self).__eq__(other) and
                self.pre_slice == other.pre_slice and
                self.function is other.function)

    def __hash__(self):
        return hash((type(self), self.function, self._transform))

    def concat(self, other):
        """Create new connection parameters which are the result of
        concatenating this connection another.

        Parameters
        ----------
        other : NodeTransmissionParameters
            Connection parameters to add to the end of this connection.

        Returns
        -------
        NodeTransmissionParameters or None
            Either a new set of transmission parameters, or None if the
            resulting transform contained no non-zero values.
        """
        # Get the outgoing transformation
        new_transform = self._transform.concat(other._transform)

        # Create a new connection (unless the resulting transform is empty,
        # in which case don't)
        if new_transform is not None:
            return NodeTransmissionParameters(
                    new_transform, self.pre_slice, self.function
            )
        else:
            # The transform consisted entirely of zeros so return None.
            return None

    @property
    def as_global_inhibition_connection(self):
        """Construct a copy of the connection with the optimisation for global
        inhibition applied.
        """
        assert self.supports_global_inhibition
        transform = self.full_transform(slice_out=False)[0, :]

        return NodeTransmissionParameters(
            Transform(size_in=self.size_in, size_out=1, transform=transform,
                      slice_in=self.slice_in),
            pre_slice=self.pre_slice,
            function=self.function
        )
