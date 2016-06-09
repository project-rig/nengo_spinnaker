from bitarray import bitarray
import numpy as np

from rig.type_casts import NumpyFixToFloatConverter

from .region import Region
from nengo_spinnaker.utils.type_casts import fix_to_np


class RecordingRegion(Region):
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def sizeof(self, vertex_slice):
        # Get the number of words per frame
        n_atoms = vertex_slice.stop - vertex_slice.start
        return self.bytes_per_frame(n_atoms) * self.n_steps

    def _read(self, mem, vertex_slice, n_steps):
        """Read a suitable amount of data out of the memory view."""
        mem.seek(0)

        # Determine how many bytes to read, then read
        width = vertex_slice.stop - vertex_slice.start
        framelength = self.bytes_per_frame(width)
        data = mem.read(n_steps * framelength)

        return data, framelength, width

    def write_subregion_to_file(self, *args, **kwargs):  # pragma: no cover
        pass  # Nothing to do


class WordRecordingRegion(RecordingRegion):
    """Record 1 word per atom per time step."""
    def bytes_per_frame(self, n_atoms):
        return 4*n_atoms

    def to_array(self, mem, vertex_slice, n_steps):
        # Read from the memory
        data, _, _ = self._read(mem, vertex_slice, n_steps)

        # Convert the data into the correct format
        data = np.fromstring(data, dtype=np.int32)
        data.shape = (n_steps, -1)

        # Recast back to float and return
        return fix_to_np(data)


class SpikeRecordingRegion(RecordingRegion):
    """Region used to record spikes.

    Spike regions use 1 bit per neuron per timestep but pad each frame to a
    multiple of words.
    """
    def bytes_per_frame(self, n_neurons):
        words_per_frame = n_neurons//32 + (1 if n_neurons % 32 else 0)
        return 4 * words_per_frame

    def to_array(self, mem, vertex_slice, n_steps):
        """Read the memory and return an appropriately formatted array of the
        results.
        """
        # Read from the memory
        data, framelength, _ = self._read(mem, vertex_slice, n_steps)

        # Format the data as a bitarray
        spikes = bitarray(endian="little")
        spikes.frombytes(data)

        # Break this into timesteps
        steps = [spikes[i * framelength * 8:(i+1) * framelength * 8] for
                 i in range(n_steps)]

        # Convert this into a NumPy array
        array = np.array(
            [[x for x in step[vertex_slice]] for step in steps],
            dtype=np.bool
        )
        return array


class VoltageRecordingRegion(RecordingRegion):
    """Region used to record neuron input voltages.

    Voltage regions use 1 short per neuron per timestep but pad each frame to a
    multiple of words.
    """
    def bytes_per_frame(self, n_neurons):
        words_per_frame = n_neurons // 2 + n_neurons % 2
        return 4 * words_per_frame

    def to_array(self, mem, vertex_slice, n_steps):
        # Read from the memory
        data, _, n_neurons = self._read(mem, vertex_slice, n_steps)

        # Convert the data into the correct format
        data = np.fromstring(data, dtype=np.uint16)
        data.shape = (n_steps, -1)

        # Recast back to float
        data_fp = NumpyFixToFloatConverter(15)(data[:, 0:n_neurons])
        return data_fp


class EncoderRecordingRegion(RecordingRegion):
    """Region used to record learnt encoders."""
    def __init__(self, n_steps, n_dimensions):
        # Superclass
        super(EncoderRecordingRegion, self).__init__(n_steps)

        self.n_dimensions = n_dimensions

    def bytes_per_frame(self, n_neurons):
        words_per_frame = n_neurons * self.n_dimensions
        return 4 * words_per_frame

    def to_array(self, mem, vertex_slice, n_steps):
        # Read from the memory
        data, _, n_neurons = self._read(mem, vertex_slice, n_steps)

        # Convert the data into the correct format
        data = np.fromstring(data, dtype=np.int32)

        # Convert the data into the correct format
        slice_encoders = NumpyFixToFloatConverter(15)(data)

        # Reshape and return
        slice_encoders = np.reshape(
            slice_encoders,
            (
                n_steps,
                n_neurons,
                self.n_dimensions
            )
        )
        return slice_encoders
