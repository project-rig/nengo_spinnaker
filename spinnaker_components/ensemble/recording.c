#include "recording.h"

bool record_buffer_initialise(recording_buffer_t *buffer, address_t region,
                              uint n_blocks, uint n_neurons) {
  // Generate and store buffer parameters
  buffer->block_length = (n_neurons >> 5) + (n_neurons & 0x1f ? 1 : 0);
  buffer->n_blocks = n_blocks;
  buffer->_sdram_start = (uint *) region;
  record_buffer_reset(buffer);

  // Create the local buffer
  MALLOC_FAIL_FALSE(buffer->buffer, buffer->block_length * sizeof(uint));

  // Zero the local buffers
  for (uint i = 0; i < buffer->block_length; i++) {
    buffer->buffer[i] = 0x0;
  }

  return true;
}

void record_buffer_reset(recording_buffer_t *buffer)
{
  // Reset the position of the recording region
  buffer->_sdram_current = buffer->_sdram_start;
}
