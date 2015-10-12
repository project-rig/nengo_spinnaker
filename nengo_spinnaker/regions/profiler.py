import logging
import numpy as np
import struct

from .region import Region

MS_SCALE = (1.0 / 200032.4)

logger = logging.getLogger(__name__)


class Profiler(Region):
    """Region used to record profiling data."""
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sizeof(self, *args, **kwargs):
        # 1 word counter and 2 words per sample
        return 4 + (8 * self.n_samples)

    def write_subregion_to_file(self, *args, **kwargs):  # pragma: no cover
        pass  # Nothing to do

    def read_from_mem(self, mem, tag_names):
        # Read number of words written by profiler
        word_written = struct.unpack("I", mem.read(4))[0]

        # Read these from memory
        data = np.fromstring(mem.read(word_written * 4), dtype=np.uint32)
        assert len(data) % 2 == 0

        # Slice data into seperate times, tags and flags
        sample_times = data[::2]
        sample_tags_and_flags = data[1::2]

        # Further split the tags and flags word
        # into seperate arrays of tags and flags
        sample_tags = np.bitwise_and(sample_tags_and_flags, 0x7FFFFFFF)
        sample_flags = np.right_shift(sample_tags_and_flags, 31)

        # Find indices of samples relating to entries and exits
        sample_entry_indices = np.where(sample_flags == 1)
        sample_exit_indices = np.where(sample_flags == 0)

        # Convert count-down times to count up times from 1st sample
        sample_times = np.subtract(sample_times[0], sample_times)
        sample_times_ms = np.multiply(sample_times, MS_SCALE, dtype=np.float)

        # Slice tags and times into entry and exits
        entry_tags = sample_tags[sample_entry_indices]
        entry_times_ms = sample_times_ms[sample_entry_indices]
        exit_tags = sample_tags[sample_exit_indices]
        exit_times_ms = sample_times_ms[sample_exit_indices]

        # Loop through unique tags
        tag_dictionary = dict()
        unique_tags = np.unique(sample_tags)
        for tag in unique_tags:
            # Check we have a name for this tag
            assert tag in tag_names

            # Get indices where these tags occur
            tag_entry_indices = np.where(entry_tags == tag)
            tag_exit_indices = np.where(exit_tags == tag)

            # Use these to get subset for this tag
            tag_entry_times_ms = entry_times_ms[tag_entry_indices]
            tag_exit_times_ms = exit_times_ms[tag_exit_indices]

            # If both these subsets aren't empty
            num_entry_times = len(tag_entry_times_ms)
            num_exit_times = len(tag_exit_times_ms)
            if num_entry_times > 0 and num_exit_times > 0:
                # If the first exit is before the first
                # Entry, add a dummy entry at beginning
                if tag_exit_times_ms[0] < tag_entry_times_ms[0]:
                    logger.warn("Profile starts mid-tag")
                    tag_entry_times_ms = np.append(0.0, tag_entry_times_ms)

                if len(tag_entry_times_ms) > len(tag_exit_times_ms):
                    logger.warn("Profile finishes mid-tag")
                    tag_entry_times_ms =\
                        tag_entry_times_ms[:num_exit_times - num_entry_times]

                # Subtract entry times from exit times
                # to get durations of each call
                tag_durations_ms = np.subtract(
                    tag_exit_times_ms, tag_entry_times_ms)

                # Add entry times and durations to dictionary
                tag_dictionary[tag_names[tag]] = (
                    tag_entry_times_ms, tag_durations_ms)

        return tag_dictionary
