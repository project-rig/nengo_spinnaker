import numpy as np
import pytest

from nengo_spinnaker.operators import EnsembleLIF


class TestEnsembleLIF(object):
    @pytest.mark.parametrize("size_in", [1, 4, 5])
    def test_init(self, size_in):
        """Test that creating an Ensemble LIF creates an empty list of local
        probes and an empty input vector.
        """
        lif = EnsembleLIF(size_in)
        assert np.all(lif.direct_input == np.zeros(size_in))
        assert lif.local_probes == list()
