def get_bytes_for_unpacked_spike_vector(slices):
    """Compute the number of bytes necessary to store the unpacked spike vector
    for the given ranges of neurons.
    """
    words = 0

    for neurons in slices:
        # Get the number of neurons
        n_neurons = neurons.stop - neurons.start

        # Update the word count, padding to allow an integral number of words
        # for each slice.
        words += n_neurons // 32
        words += 1 if n_neurons % 32 else 0

    # Multiply to get bytes
    return words * 4
