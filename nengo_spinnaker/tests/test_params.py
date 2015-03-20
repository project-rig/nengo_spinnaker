"""Tests for internally used descriptor/parameter types.
"""
import pytest
from nengo_spinnaker import params


class TestIntParam(object):
    @pytest.mark.parametrize(
        "fail_value, exc_type, reason",
        [(-1, ValueError, "Value must be >= 0"),
         (101, ValueError, "Value must be <= 100"),
         (None, ValueError, "Value may not be None"),
         (1.0, TypeError, "Value must be an integer (not float)"),
         ]
    )
    def test_exceptions(self, fail_value, exc_type, reason):
        # Create the object to test with
        class P(object):
            x = params.IntParam(min=0, max=100)

        p = P()

        with pytest.raises(exc_type) as excinfo:
            p.x = fail_value
        assert reason in str(excinfo.value)

    def test_success(self):
        # Create the object to test with
        class P(object):
            x = params.IntParam(min=0, max=100, default=5, allow_none=True)

        p = P()
        assert p.x == 5
        p.x = None
        assert p.x is None
        p.x = 33
        assert p.x == 33
