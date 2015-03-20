"""Descriptors.
"""
import six
import weakref


class IntParam(object):
    """Descriptor for integer parameters.
    """
    def __init__(self, min=None, max=None, allow_none=False, default=0):
        """Create a new parameter instance.

        Parameters
        ----------
        min : int or None
            Minimum value accepted by the descriptor.  None means unbounded.
        max : int or None
            Maximum value accepted by the descriptor.  None means unbounded.
        allow_none : bool
            Whether the descriptor will accept None as a value.
        default : int or None
            The default value to represent.
        """
        self.min = min
        self.max = max
        self.allow_none = allow_none
        self.default = default
        self.data = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        if value is None:
            if not self.allow_none:
                raise ValueError("Value may not be None")
        elif not isinstance(value, six.integer_types):
            raise TypeError(
                "Value must be an integer (not {})".format(
                    type(value).__name__)
            )
        else:
            if self.min is not None and value < self.min:
                raise ValueError("Value must be >= {}".format(self.min))
            if self.max is not None and value > self.max:
                raise ValueError("Value must be <= {}".format(self.max))

        self.data[instance] = value
