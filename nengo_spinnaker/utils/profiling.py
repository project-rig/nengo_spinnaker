import numpy as np


def print_summary(profiling_data, duration):
    ms_time_bins = np.arange(duration * 1000.0)

    # Summarise data for all tags
    for tag_name, times in profiling_data.iteritems():
        print("Tag:%s" % (tag_name))

        print("\tMean time:%fms" % (np.average(times[1])))

        # Digitize the sample entry times into these bins
        sample_timestep_indices = np.digitize(times[0], ms_time_bins)
        assert len(sample_timestep_indices) == len(times[1])

        # Calculate the average number of samples in each bin
        print("\tMean samples per timestep:%f" %
              (np.average(np.bincount(sample_timestep_indices))))

        # Determine the last sample time (if profiler runs out
        # Of space to write samples it may not be duration)
        last_sample_time = np.amax(sample_timestep_indices) + 1
        print("\tLast sample time:%fms" % (last_sample_time))

        # Create bins to hold total time spent in this tag during each
        # Timestep and add duration to total in corresponding bin
        total_sample_duration_per_timestep = np.zeros(last_sample_time)
        for sample_duration, index in zip(times[1], sample_timestep_indices):
            total_sample_duration_per_timestep[index] += sample_duration

        print("\tMean time per timestep:%fms" %
              (np.average(total_sample_duration_per_timestep)))


def write_csv_header(profiling_data, csv_writer, extra_column_headers):
    # Write header row with extra column headers
    # followed by tag names found in profiling_data
    csv_writer.writerow(extra_column_headers + list(profiling_data.iterkeys()))


def write_csv_row(profiling_data, csv_writer, extra_column_values):
    # Calculate mean of all profiling tags
    mean_times = [np.average(t[1]) for t in profiling_data.itervalues()]

    # Write extra column followed by means
    csv_writer.writerow(extra_column_values + mean_times)
