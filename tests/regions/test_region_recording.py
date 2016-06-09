import mock
import numpy as np
import pytest
import struct

from rig.type_casts import NumpyFloatToFixConverter, NumpyFixToFloatConverter

from nengo_spinnaker.regions import recording as rr


class TestWordRecordingRegion(object):
    @pytest.mark.parametrize(
        "n_steps, vertex_slice, words_per_frame",
        [(1, slice(0, 2), 2),
         (100, slice(0, 32), 32),
         (1000, slice(0, 33), 33),
         ]
    )
    def test_sizeof(self, n_steps, vertex_slice, words_per_frame):
        # Create the region
        sr = rr.WordRecordingRegion(n_steps)

        # Check that the size is reported correctly
        assert sr.sizeof(vertex_slice) == 4 * words_per_frame * n_steps


class TestSpikeRegion(object):
    """Spike regions use 1 bit per neuron per timestep but pad each frame to a
    multiple of words.
    """
    @pytest.mark.parametrize(
        "n_steps, vertex_slice, words_per_frame",
        [(1, slice(0, 2), 1),
         (100, slice(0, 32), 1),
         (1000, slice(0, 33), 2),
         ]
    )
    def test_sizeof(self, n_steps, vertex_slice, words_per_frame):
        # Create the region
        sr = rr.SpikeRecordingRegion(n_steps)

        # Check that the size is reported correctly
        assert sr.sizeof(vertex_slice) == 4 * words_per_frame * n_steps

    def test_to_array(self):
        """Check that data can be read back from a memory."""
        # Data to reconstruct; this is two frames of two words each
        data = struct.pack(
            "<4I",
            0b00000000000000000000000000000010,
            0b11111111111111111111111111100001,  # Ignore 27MSB
            0b00000000000000000000000000000100,
            0b11111111111111111111111111100010   # Ignore 27MSB
        )

        # Construct a memory to read from
        mem = mock.Mock()
        mem.read.return_value = data

        # Get the array
        sr = rr.SpikeRecordingRegion(100)
        array = sr.to_array(mem, slice(0, 37), 2)

        # Check that an appropriate read was made
        mem.seek.assert_called_once_with(0)
        mem.read.assert_called_once_with(16)

        # Check that the return array is of an appropriate shape
        assert array.shape == (2, 37)

        # Check that the right neurons have fired
        expected = np.array([[False] * 37] * 2)
        expected[0][1] = True
        expected[0][32] = True
        expected[1][2] = True
        expected[1][33] = True

        assert np.all(array == expected)


class TestVoltageRegion(object):
    """Voltage regions use 1 short per neuron per timestep but pad each frame
    to a multiple of words.
    """
    @pytest.mark.parametrize(
        "n_steps, vertex_slice, words_per_frame",
        [(1, slice(0, 2), 1),
         (100, slice(0, 32), 16),
         (1000, slice(0, 33), 17),
         ]
    )
    def test_sizeof(self, n_steps, vertex_slice, words_per_frame):
        # Create the region
        sr = rr.VoltageRecordingRegion(n_steps)

        # Check that the size is reported correctly
        assert sr.sizeof(vertex_slice) == 4 * words_per_frame * n_steps

    def test_to_array(self):
        """Check that the data can be read back from a memory."""
        # Data to reconstruct
        float_to_u1_15 = NumpyFloatToFixConverter(False, 16, 15)
        voltages = np.random.uniform(0.0, 1.0, size=(3, 38))
        voltages_fp = float_to_u1_15(voltages)
        data = voltages_fp.tostring()

        # Construct a memory to read from
        mem = mock.Mock()
        mem.read.return_value = data

        # Get the array
        vr = rr.VoltageRecordingRegion(100)
        array = vr.to_array(mem, slice(5, 5 + 37), 3)

        # Check that an appropriate read was made
        mem.seek.assert_called_once_with(0)
        mem.read.assert_called_once_with(19*4*3)

        # Check that the return array is of an appropriate shape
        assert array.shape == (3, 37)

        # Check that the data is correct
        expected = NumpyFixToFloatConverter(15)(voltages_fp)
        assert np.all(array == expected[:,:-1])

        
class TestEncoderRegion(object):
    """Encoder learning regions use 1 word per neuron per dimension per timestep"""
    @pytest.mark.parametrize(
        "vertex_slice, n_steps, n_dimensions",
        [(slice(0, 800), 0, 16),
         (slice(0, 800), 2000, 16),
         (slice(800, 1600), 2000, 16)]
    )
    def test_sizeof(self, vertex_slice, n_steps, n_dimensions):
        # Create the region
        sr = rr.EncoderRecordingRegion(n_steps, n_dimensions)

        # Check that the size is reported correctly
        assert sr.sizeof(vertex_slice) == 4 * n_steps * n_dimensions *\
            (vertex_slice.stop - vertex_slice.start)

    def test_to_array(self):
        """Check that the data can be read back from a memory."""
        # Data to reconstruct
        n_steps = 100
        n_dimensions = 16
        n_neurons = 100
        n_words = n_steps * n_dimensions * n_neurons

        float_to_s16_15 = NumpyFloatToFixConverter(True, 32, 15)
        encoders = np.random.uniform(0.0, 128.0, size=n_words)
        encoders_fp = float_to_s16_15(encoders)
        data = encoders_fp.tostring()

        # Construct a memory to read from
        mem = mock.Mock()
        mem.read.return_value = data

        # Get the array
        er = rr.EncoderRecordingRegion(n_steps, n_dimensions)
        array = er.to_array(mem, slice(0, n_neurons), n_steps)

        # Check that an appropriate read was made
        mem.seek.assert_called_once_with(0)
        mem.read.assert_called_once_with(n_words * 4)

        # Check that the return array is of an appropriate shape
        assert array.shape == (n_steps, n_neurons, n_dimensions)

        # Check that the data is correct
        expected = NumpyFixToFloatConverter(15)(encoders_fp)
        assert np.all(array.reshape(n_words) == expected)
