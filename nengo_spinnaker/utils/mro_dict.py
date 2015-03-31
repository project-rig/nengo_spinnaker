class MRODict(dict):
    """A dictionary with classes as keys and MRO-searching look-up behaviour.

    Each entry in the dictionary has a class as its key.  When a look-up is
    performed the MRO of the class being looked up is used to determine which
    entries are relevant and the first matching entry is returned.  A decorator
    is provided to allow easy entering of functions as values in the
    dictionary.

    For example::

        >>> class SimpleObject(object):
        ...     pass
        >>>
        >>> class DerivedObject(SimpleObject):
        ...     pass
        >>>
        >>>
        >>> mrodict = MRODict()
        >>> mrodict[SimpleObject] = 5
        >>>
        >>> mrodict[SimpleObject]
        5
        >>> mrodict[DerivedObject]  # Uses the MRO to find a value
        5

    `KeyError`s are raised if a suitable class can't be found in the
    dictionary::

        >>> class NotDerivedObject(object):
        ...     pass
        >>>
        >>> mrodict[NotDerivedObject]
        Traceback (most recent call last):
        KeyError: <class 'nengo_spinnaker.utils.mro_dict.NotDerivedObject'>
    """
    def __getitem__(self, cls):
        """Get the first matching entry from the class MRO."""
        for t in cls.__mro__:
            if t in self:
                return super(MRODict, self).__getitem__(t)
        else:
            raise KeyError(cls)

    def register(self, key, allow_overrides=False):
        """Decorator which allows registering of functions as entries in the
        dictionary.
        """
        if key in self and not allow_overrides:
            raise KeyError(
                "class {} already in dictionary".format(key.__name__))

        def decorator(f):
            self[key] = f
            return f

        return decorator
