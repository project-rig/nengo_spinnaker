#include "record_learnt_encoders.h"

//-----------------------------------------------------------------------------
// Global functions
//-----------------------------------------------------------------------------
bool record_learnt_encoders_initialise(encoder_recording_buffer_t *buffer, address_t region)
{
  buffer->sdram_start = (value_t*)region;
  record_learnt_encoders_reset(buffer);
  return true;
}
//-----------------------------------------------------------------------------
void record_learnt_encoders_reset(encoder_recording_buffer_t *buffer)
{
  // Reset the position of the recording region
  buffer->sdram_current = buffer->sdram_start;
}