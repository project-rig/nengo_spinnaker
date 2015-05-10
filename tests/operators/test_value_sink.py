import mock
from nengo_spinnaker.operators import ValueSink


def test_value_sink_init():
    probe = mock.Mock(name="Probe")
    probe.size_in = 3
    probe.sample_every = 0.0043

    v = ValueSink(probe, 0.001)
    assert v.probe is probe
    assert v.size_in == 3
    assert v.sample_every == 4
