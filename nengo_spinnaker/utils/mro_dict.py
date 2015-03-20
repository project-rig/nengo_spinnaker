class MRODict(object):
    """A dictionary with classes as keys and MRO-searching look-up behaviour.

    Each entry in the dictionary has a class as its key.  When a look-up is
    performed the MRO of the class being looked up is used to determine which
    entries are relevant and the first matching entry is returned.  A decorator
    is provided to allow easy entering of functions as values in the
    dictionary.
    """
    def __init__(self):
        self._dict = dict()

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, cls):
        """Get the first matching entry from the class MRO."""
        for t in cls.__mro__:
            if t in self._dict:
                return self._dict[t]
        else:
            raise KeyError(cls)

    def register(self, key, allow_overrides=False):
        """Decorator which allows registering of functions as entries in the
        dictionary.
        """
        if key in self._dict and not allow_overrides:
            raise KeyError(
                "class {} already in dictionary".format(key.__name__))

        def decorator(f):
            self._dict[key] = f
            return f

        return decorator
