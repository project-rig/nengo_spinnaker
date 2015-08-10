#include "profiler.h"

#ifdef PROFILER_ENABLED

//---------------------------------------
// Globals
//---------------------------------------
uint32_t *profiler_count = NULL;
uint32_t profiler_samples_remaining = 0;
uint32_t *profiler_output = NULL;

//---------------------------------------
// Functions
//---------------------------------------
void profiler_read_region(uint32_t* address)
{
  profiler_count = &address[0];
  profiler_output = &address[1];
}
//---------------------------------------
void profiler_finalise()
{
  uint32_t words_written = (profiler_output - profiler_count) - 1;
  *profiler_count = words_written;
  io_printf(IO_BUF, "Profiler wrote %u bytes to %08x\n.", (words_written * 4) + 4, profiler_count);
}
//---------------------------------------
void profiler_init(uint32_t num_samples)
{
  io_printf(IO_BUF, "Initialising profiler with storage for %u samples\n", num_samples);
  
  // Initialize number of samples remaining
  profiler_samples_remaining = num_samples;
  
  // If profiler is turned on, start timer 2 with no clock divider
  if(profiler_samples_remaining > 0)
  {
    tc[T2_CONTROL] = 0x82;
    tc[T2_LOAD] = 0;
  }
}

#endif  // PROFILER_ENABLED