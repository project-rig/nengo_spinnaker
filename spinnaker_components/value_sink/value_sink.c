#include "value_sink.h"

address_t rec_start, rec_curr;
uint n_dimensions;
value_t *input;

input_filter_t g_input;

void sink_update(uint ticks, uint arg1) {
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks) {
    spin1_exit(0);
  }

  // Filter inputs, write the latest value to SRAM
  input_filter_step(&g_input, true);
  spin1_memcpy(rec_curr, input, n_dimensions * sizeof(value_t));
  rec_curr = &rec_curr[n_dimensions];
}

void mcpl_callback(uint key, uint payload) {
  input_filter_mcpl_rx(&g_input, key, payload);
}

void c_main(void)
{
  address_t address = system_load_sram();

  // Load parameters and filters
  region_system_t *pars = (region_system_t *) region_start(1, address);
  n_dimensions = pars->n_dimensions;
  input = input_filter_initialise(&g_input, n_dimensions);

  if (input == NULL) {
    return;
  }

  if (!input_filter_get_filters(&g_input, region_start(2, address)) ||
      !input_filter_get_filter_routes(&g_input, region_start(3, address))
  ) {
    io_printf(IO_BUF, "[Value Sink] Failed to start.\n");
    return;
  }
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
