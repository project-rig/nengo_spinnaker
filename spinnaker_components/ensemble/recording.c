#include "recording.h"

bool record_buffer_initialise(recording_buffer_t *buffer, address_t region,
                              uint n_frames, uint n_neurons) {
  // Generate and store buffer parameters
  buffer->frame_length = (n_neurons >> 5) + (n_neurons & 0x1f ? 1 : 0);
  buffer->n_frames = n_frames;
  buffer->_sdram_start = (uint *) region;
  buffer->_sdram_current = (uint *) region;

  buffer->current_frame = UINT32_MAX;  // To cause overflow on first tick

  // Create the local buffer
  MALLOC_FAIL_FALSE(buffer->buffer, buffer->frame_length * sizeof(uint));

  // Zero the local buffers
  for (uint i = 0; i < buffer->frame_length; i++) {
    buffer->buffer[i] = 0x0;
  }

  return true;
}
