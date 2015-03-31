import pytest
from nengo_spinnaker.utils.mro_dict import MRODict


class TestMRODict(object):
    """Tests for a dictionary which looks up items by iterating through their
    MRO.
    """
    def test_get_item_success(self):
        class A(object):
            pass

        class B(A):
            pass

        class C(object):
            pass

        d = MRODict()

        d[A] = 1
        d[C] = 2

        # Retrieving B should end up getting the value mapped to A
        assert d[B] == d[A] == 1
        assert d[C] == 2

        d[B] = 3
        assert d[B] == 3
        assert d[A] == 1

    def test_get_item_failure(self):
        class ThisClassNotPresent(object):
            pass

        d = MRODict()

        with pytest.raises(KeyError) as excinfo:
            d[ThisClassNotPresent]
        assert ThisClassNotPresent.__name__ in str(excinfo.value)

    def test_register_success(self):
        class A(object):
            pass

        mrodict = MRODict()

        @mrodict.register(A)
        def f():  # pragma : no cover
            pass

        assert mrodict[A] is f

        @mrodict.register(A, allow_overrides=True)
        def g():  # pragma : no cover
            pass

        assert mrodict[A] is g

    def test_register_collision(self):
        class A(object):
            pass

        mrodict = MRODict()

        @mrodict.register(A)
        def f():  # pragma : no cover
            pass

        assert mrodict[A] is f

        with pytest.raises(KeyError) as excinfo:
            @mrodict.register(A)
            def g():  # pragma : no cover
                pass
        assert A.__name__ in str(excinfo.value)
