#include "value_sink.h"

address_t rec_start, rec_curr;
uint n_dimensions;
value_t *input;

if_collection_t g_input;

void sink_update(uint ticks, uint arg1) {
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks > simulation_ticks) {
    spin1_exit(0);
    return;
  }

  // Filter inputs, write the latest value to SRAM
  input_filtering_step(&g_input);
  spin1_memcpy(rec_curr, input, n_dimensions * sizeof(value_t));
  rec_curr = &rec_curr[n_dimensions];
}

void mcpl_callback(uint key, uint payload) {
  input_filtering_input(&g_input, key, payload);
}

void c_main(void)
{
  address_t address = system_load_sram();

  // Load parameters and filters
  region_system_t *pars = (region_system_t *) region_start(1, address);
  n_dimensions = pars->n_dimensions;
  input_filtering_initialise_output(&g_input, n_dimensions);
  input = g_input.output;

  if (input == NULL) {
    return;
  }

  input_filtering_get_filters(&g_input, region_start(2, address));
  input_filtering_get_routes(&g_input, region_start(3, address));
  rec_start = region_start(15, address);

  // Set up callbacks, start
  spin1_set_timer_tick(pars->timestep);
  spin1_callback_on(MCPL_PACKET_RECEIVED, mcpl_callback, -1);
  spin1_callback_on(TIMER_TICK, sink_update, 2);

  while(true)
  {
    // Wait for data loading, etc.
    event_wait();

    // Determine how long to simulate for
    config_get_n_ticks();

    // Reset the recording region location
    rec_curr = rec_start;

    // Perform the simulation
    spin1_start(SYNC_WAIT);
  }
}
