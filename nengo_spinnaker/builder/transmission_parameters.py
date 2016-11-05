"""Transmission Parameters

A transmission parameter object contains the Nengo specific description of the
values that will be produced by SpiNNaker core(s) [this is in contrast to the
SpiNNaker-specific information required to route those values across the
network as multicast packets]. Each type of transmission parameter must be
*equatable* (it must have both `__ne__` and `__eq__` defined) and *hashable*
(it must have `__hash__` defined).

Moreover, each transmission parameter type must have a method called `concats`
which accepts a list of `PassthroughNodeTransmissionParameters` as an argument
and yields new transmission parameters which representing the result of
chaining the first parameter with each of the other parameters. Several sample
implementations of this method are presented within this document.

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
the parameters with a size out of `1` and only containing one row of the
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


class TransmissionParameters(object):
    __slots__ = ["size_in", "size_out", "transform", "slice_in", "slice_out"]

    def __init__(self, size_in, size_out, transform,
                 slice_in=slice(None), slice_out=slice(None)):
        self.size_in = size_in
        self.size_out = size_out

        # Transform the slices into an appropriate format
        self.slice_in = _get_slice_as_ndarray(slice_in, size_in)
        self.slice_out = _get_slice_as_ndarray(slice_out, size_out)

        # Copy the transform into a C-contiguous, read-only form
        self.transform = np.array(transform, order='C')
        self.transform.flags["WRITEABLE"] = False

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.size_in == other.size_in and
                self.size_out == other.size_out and
                np.array_equal(self.slice_in, other.slice_in) and
                np.array_equal(self.slice_out, other.slice_out) and
                np.array_equal(self.transform, other.transform))

    @property
    def _hashables(self):
        return ((type(self),
                 self.size_in,
                 self.size_out,
                 fasthash(self.slice_in).hexdigest(),
                 fasthash(self.slice_out).hexdigest(),
                 fasthash(self.transform).hexdigest()))

    def __hash__(self):
        # The hash is combination of all the elements of the tuple, but we use
        # a faster hashing mechanism to hash the array types.
        return hash(self._hashables)

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

    def projects_to(self, space):
        """Indicate whether the output of the connection described by the
        connection will intersect with the specified range of dimensions.
        """
        space = set(_get_slice_as_ndarray(space, self.size_out))

        if self.transform.ndim == 0:
            outputs = set(self.slice_out)
        elif self.transform.ndim == 1:
            outputs = set(self.slice_out[self.transform != 0])
        elif self.transform.ndim == 2:
            outputs = set(self.slice_out[np.any(self.transform != 0, axis=1)])
        else:  # pragma: no cover
            raise NotImplementedError

        return len(outputs.intersection(space)) != 0


class PassthroughNodeTransmissionParameters(TransmissionParameters):
    """Parameters describing information transmitted by a passthrough node.
    """
    def concats(self, others):
        """Create new connection parameters which are the result of
        concatenating this connection several others.

        Parameters
        ----------
        others : [PassthroughNodeTransmissionParameters, ...]
            Another set of connection parameters to add to the end of this
            connection.

        Yields
        ------
        PassthroughNodeTransmissionParameters or None
            Either a new set of transmission parameters, or None if the
            resulting transform contained no non-zero values.
        """
        # Get the transform from this connection
        A = self.full_transform(slice_out=False)

        for other in others:
            # Combine the transforms
            new_transform = np.dot(other.full_transform(slice_in=False), A)

            # Create a new connection (unless the resulting transform is empty,
            # in which case don't)
            if np.any(new_transform != 0):
                yield PassthroughNodeTransmissionParameters(
                    size_in=self.size_in,
                    size_out=other.size_out,
                    slice_in=self.slice_in,
                    slice_out=other.slice_out,
                    transform=new_transform
                )
            else:
                # The transform consisted entirely of zeros so return None.
                yield None

    @property
    def as_global_inhibition_connection(self):
        """Construct a copy of the connection with the optimisation for global
        inhibition applied.
        """
        assert self.supports_global_inhibition
        transform = self.full_transform(slice_out=False)[0, :]
        return PassthroughNodeTransmissionParameters(size_in=self.size_in,
                                                     size_out=1,
                                                     transform=transform,
                                                     slice_in=self.slice_in)


class EnsembleTransmissionParameters(TransmissionParameters):
    """Parameters describing information transmitted by an ensemble.

    Attributes
    ----------
    decoders : ndarray
        A matrix describing a decoding of the ensemble (sized N x D).
    size_out : int
        Size of the space that the ensemble is being decoded into (may be
        greater than D if a slice is provided).
    slice_out :
        Slice of the output space which the decoder targets.
    learning_rule :
        Learning rule associated with the decoding.
    """
    __slots__ = TransmissionParameters.__slots__ + [
        "decoders", "learning_rule"
    ]

    def __init__(self, decoders, size_out, slice_out=slice(None),
                 learning_rule=None, transform=1):
        # Copy the decoders into a C-contiguous, read-only array
        self.decoders = np.array(decoders, order='C')
        self.decoders.flags["WRITEABLE"] = False

        # Store the learning rule
        self.learning_rule = learning_rule

        super(EnsembleTransmissionParameters, self).__init__(
            size_in=self.decoders.shape[0],
            size_out=size_out,
            transform=transform,
            slice_in=slice(None),
            slice_out=slice_out
        )

    def __eq__(self, other):
        # Two parameters are equal only if they are of the same type, both have
        # no learning rule and are equivalent in all other fields.
        return (super(EnsembleTransmissionParameters, self).__eq__(other) and
                np.array_equal(self.decoders, other.decoders) and
                self.learning_rule is None and
                other.learning_rule is None)

    @property
    def _hashables(self):
        return super(EnsembleTransmissionParameters, self)._hashables + (
            fasthash(self.decoders).hexdigest(), self.learning_rule
        )

    def __hash__(self):
        return hash(self._hashables)

    def concats(self, others):
        """Create new connection parameters which are the result of
        concatenating this connection with others.

        Parameters
        ----------
        others : [PassthroughNodeTransmissionParameters, ...]
            Another set of connection parameters to add to the end of this
            connection.

        Yields
        ------
        EnsembleTransmissionParameters or None
            Either a new set of transmission parameters, or None if the
            resulting transform contained no non-zero values.
        """
        # Get the outgoing transformation
        A = self.full_transform(slice_out=False)

        for other in others:
            # Combine the transforms
            new_transform = np.dot(other.full_transform(slice_in=False), A)

            # Create a new connection (unless the resulting transform is empty,
            # in which case don't)
            if np.any(new_transform != 0):
                yield EnsembleTransmissionParameters(
                    decoders=self.decoders,
                    size_out=other.size_out,
                    slice_out=other.slice_out,
                    learning_rule=self.learning_rule,
                    transform=new_transform
                )
            else:
                # The transform consisted entirely of zeros so return None.
                yield None

    @property
    def as_global_inhibition_connection(self):
        """Construct a copy of the connection with the optimisation for global
        inhibition applied.
        """
        assert self.supports_global_inhibition
        transform = self.full_transform(slice_out=False)[0, :]
        return EnsembleTransmissionParameters(self.decoders,
                                              size_out=1,
                                              learning_rule=self.learning_rule,
                                              transform=transform)

    @property
    def full_decoders(self):
        """Get the matrix corresponding to a combination of the decoders and
        the transform applied by the connection.
        """
        return np.dot(self.full_transform(slice_in=False, slice_out=False),
                      self.decoders)


class NodeTransmissionParameters(TransmissionParameters):
    __slots__ = TransmissionParameters.__slots__ + ["pre_slice", "function"]

    def __init__(self, size_in, size_out, transform, slice_out=slice(None),
                 pre_slice=slice(None), function=None):
        """
        Parameters
        ----------
        size_in : int
            Either the size out of the transmitting object (if the function is
            `None`) or the size out of the function.
        """
        super(NodeTransmissionParameters, self).__init__(
            size_in=size_in,
            size_out=size_out,
            transform=transform,
            slice_out=slice_out
        )
        self.pre_slice = pre_slice
        self.function = function

    def __eq__(self, other):
        return (super(NodeTransmissionParameters, self).__eq__(other) and
                self.function is other.function)

    @property
    def _hashables(self):
        return super(NodeTransmissionParameters, self)._hashables + (
            self.function,
        )

    def __hash__(self):
        return hash(self._hashables)

    def concats(self, others):
        """Create new connection parameters which are the result of
        concatenating this connection with others.

        Parameters
        ----------
        others : [NodeTransmissionParameters, ...]
            Another set of connection parameters to add to the end of this
            connection.

        Yields
        ------
        NodeTransmissionParameters or None
            Either a new set of transmission parameters, or None if the
            resulting transform contained no non-zero values.
        """
        # Get the outgoing transformation
        A = self.full_transform(slice_out=False)

        for other in others:
            # Combine the transforms
            new_transform = np.dot(other.full_transform(slice_in=False), A)

            # Create a new connection (unless the resulting transform is empty,
            # in which case don't)
            if np.any(new_transform != 0):
                yield NodeTransmissionParameters(
                    size_in=self.size_in,
                    size_out=other.size_out,
                    transform=new_transform,
                    pre_slice=self.pre_slice,
                    function=self.function
                )
            else:
                # The transform consisted entirely of zeros so return None.
                yield None

    @property
    def as_global_inhibition_connection(self):
        """Construct a copy of the connection with the optimisation for global
        inhibition applied.
        """
        assert self.supports_global_inhibition
        transform = self.full_transform(slice_out=False)[0, :]
        return NodeTransmissionParameters(size_in=self.size_in,
                                          size_out=1,
                                          transform=transform,
                                          pre_slice=self.pre_slice,
                                          function=self.function)


def _get_slice_as_ndarray(sl, size):
    """Return a slice as a read-only Numpy array."""
    if isinstance(sl, slice):
        sl = np.arange(size)[sl]
    else:
        sl = np.array(sorted(set(sl)))

    sl.flags["WRITEABLE"] = False

    return sl
