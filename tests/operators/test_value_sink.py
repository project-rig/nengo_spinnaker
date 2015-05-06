from nengo_spinnaker.operators import ValueSink


def test_value_sink_init():
    v = ValueSink(3, 4)
    assert v.size_in == 3
    assert v.sample_every == 4
