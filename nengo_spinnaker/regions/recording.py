from bitarray import bitarray
import numpy as np

from rig.type_casts import NumpyFixToFloatConverter

from .region import Region


class RecordingRegion(Region):
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def sizeof(self, vertex_slice):
        # Get the number of words per frame
        n_atoms = vertex_slice.stop - vertex_slice.start
        return self.bytes_per_frame(n_atoms) * self.n_steps

    def write_subregion_to_file(self, *args, **kwargs):  # pragma: no cover
        pass  # Nothing to do


class WordRecordingRegion(RecordingRegion):
    """Record 1 word per atom per time step."""
    def bytes_per_frame(self, n_atoms):
        return 4*n_atoms


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
        mem.seek(0)

        # Determine how many bytes to read, then read
        n_neurons = vertex_slice.stop - vertex_slice.start
        framelength = self.bytes_per_frame(n_neurons)
        data = mem.read(n_steps * framelength)

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
        mem.seek(0)

        # Determine how many bytes to read, then read
        n_neurons = vertex_slice.stop - vertex_slice.start
        framelength = self.bytes_per_frame(n_neurons)
        data = mem.read(n_steps * framelength)

        # Convert the data into the correct format
        data = np.fromstring(data, dtype=np.uint16)
        data.shape = (n_steps, -1)

        # Recast back to float
        data_fp = NumpyFixToFloatConverter(15)(data[:, 0:n_neurons])
        return data_fp
