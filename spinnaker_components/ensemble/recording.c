#include "recording.h"

// Generic buffer initialisation
bool record_buffer_initialise(recording_buffer_t *buffer, address_t region,
                              uint32_t n_blocks, uint32_t block_length)
{
  // Store buffer parameters
  buffer->n_blocks = n_blocks;
  buffer->block_length = block_length;
  buffer->_sdram_start = (uint32_t *) region;
  record_buffer_reset(buffer);

  // Create the local buffer
  MALLOC_FAIL_FALSE(buffer->buffer, buffer->block_length * sizeof(uint));

  // Zero the local buffers
  memset(buffer->buffer, 0x0, buffer->block_length * sizeof(uint32_t));

  return true;
}

void record_buffer_reset(recording_buffer_t *buffer)
{
  // Reset the position of the recording region
  buffer->_sdram_current = buffer->_sdram_start;
}

/*****************************************************************************/
/* Spike specific functions.
 */

/*!\brief Initialise a new recording buffer for recording spikes
 */
bool record_buffer_initialise_spikes(
  recording_buffer_t *buffer,
  address_t region,
  uint n_blocks,
  uint n_neurons
)
{
  // Compute the block length (an integral number of words allowing for 1 bit
  // per neuron).
  uint32_t block_length = (n_neurons >> 5) + (n_neurons & 0x1f ? 1 : 0);

  // Use this to create the recording buffer
  return record_buffer_initialise(buffer, region, n_blocks, block_length);
};
