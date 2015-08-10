import numpy as np

def display_tag_summary(times, duration):
    print("\t\tMean time:%f" % (np.average(times[1])))

    # Digitize the sample entry times into these bins
    sample_timestep_indices = np.digitize(times[0], np.arange(duration))
    assert len(sample_timestep_indices) == len(times[1])

    # Calculate the average number of samples in each bin
    print("\t\tMean samples per timestep:%f" % (np.average(np.bincount(sample_timestep_indices))))

    # Determine the last sample time (if profiler runs out
    # Of space to write samples it may not be duration)
    last_sample_time = np.amax(sample_timestep_indices) + 1
    print("\t\tLast sample time:%fms" % (last_sample_time))

    # Create bins to hold total time spent in this tag during each
    # Timestep and add duration to total in corresponding bin
    total_sample_duration_per_timestep = np.zeros(last_sample_time)
    for sample_duration, index in zip(times[1], sample_timestep_indices):
        total_sample_duration_per_timestep[index] += sample_duration

    print("\t\tMean time per timestep:%f" % (np.average(total_sample_duration_per_timestep)))

def display_summary(profiling_data, duration):
    # Summarise data for all tags
    for tag_name, times in profiling_data.iteritems():
        print("\tTag:%s" % (tag_name))

        display_tag_summary(times, duration)