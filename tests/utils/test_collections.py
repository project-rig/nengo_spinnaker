import pytest

from nengo_spinnaker.utils import collections as nscollections


def test_noneignoringlist():
    """Test a list which will not append None."""
    nil = nscollections.noneignoringlist()

    # We can append items normally
    nil.append(123)
    assert nil == [123]

    # Unless they're None
    nil.append(None)
    assert nil == [123]


def test_flatinsertionlist():
    """Test a list which will always flatten the items it inserts."""
    fil = nscollections.flatinsertionlist()

    # We can append items normally
    fil.append(123)
    assert fil == [123]

    # Unless they're lists
    fil.append([1, 2, 3])
    assert fil == [123, 1, 2, 3]


def test_registerabledict():
    """Test a dictionary that allows functions to be registered against it."""
    rd = nscollections.registerabledict()

    @rd.register("ABCD")
    def test_a():
        pass  # pragma : no cover

    assert rd["ABCD"] is test_a

    # Registering twice raises an error
    with pytest.raises(Exception) as excinfo:
        @rd.register("ABCD")
        def test_b():
            pass  # pragma : no cover
    assert "ABCD" in str(excinfo.value)

    # But this can be overridden
    @rd.register("ABCD", allow_overrides=True)
    def test_c():
        pass  # pragma : no cover


def test_mrolookupdict():
    """Test a dictionary which will look up items by going through their
    MROs.
    """
    class ParentA(object):
        pass

    class ChildA(ParentA):
        pass

    mdict = nscollections.mrolookupdict()
    mdict[ParentA] = 5

    assert mdict[ChildA] == 5

    mdict[ChildA] = 10
    assert mdict[ChildA] == 10

    # Objects not in the dictionary raise KeyErrors
    with pytest.raises(KeyError):
        mdict[object]
