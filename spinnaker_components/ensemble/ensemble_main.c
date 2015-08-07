#include "ensemble.h"
#include "ensemble_data.h"
#include "ensemble_pes.h"

void c_main(void) {
  // Set the system up
  io_printf(IO_BUF, "[Ensemble] C_MAIN\n");
  address_t address = system_load_sram();
  if (!data_system(region_start(1, address))) {
    io_printf(IO_BUF, "[Ensemble] Failed to start.\n");
    return;
  }

  // Get data
  data_get_bias(region_start(2, address), g_ensemble.n_neurons);
  data_get_encoders(region_start(3, address), g_ensemble.n_neurons, g_input.n_dimensions);
  data_get_decoders(region_start(4, address), g_ensemble.n_neurons, g_n_output_dimensions);
  data_get_keys(region_start(5, address), g_n_output_dimensions);

  // Get the inhibitory gains
  g_ensemble.inhib_gain = spin1_malloc(g_ensemble.n_neurons * sizeof(value_t));
  if (g_ensemble.inhib_gain == NULL) {
    io_printf(IO_BUF, "[Ensemble] Failed to malloc inhib gains.\n");
    return;
  }
  spin1_memcpy(g_ensemble.inhib_gain, region_start(10, address),
               g_ensemble.n_neurons * sizeof(value_t));
  for (uint n = 0; n < g_ensemble.n_neurons; n++) {
    io_printf(IO_BUF, "Inhib gain[%d] = %k\n", n, g_ensemble.inhib_gain[n]);
  }

  // Load subcomponents
  if (!input_filter_get_filters(&g_input, region_start(6, address)) ||
      !input_filter_get_filter_routes(&g_input, region_start(7, address)) ||
      !input_filter_get_filters(&g_input_inhibitory, region_start(8, address)) ||
      !input_filter_get_filter_routes(&g_input_inhibitory, region_start(9, address)) ||
      !input_filter_get_filters(&g_input_modulatory, region_start(11, address)) ||
      !input_filter_get_filter_routes(&g_input_modulatory, region_start(12, address))) 
  {
    io_printf(IO_BUF, "[Ensemble] Failed to start.\n");
    return;
  }
  
  if(!get_pes(region_start(13, address)))
  {
    io_printf(IO_BUF, "[Ensemble] Failed to start.\n");
    return;
  }

  // Set up recording
  if (!record_buffer_initialise_spikes(
        &g_ensemble.record_spikes, region_start(15, address),
        simulation_ticks, g_ensemble.n_neurons
       )
     )
  {
    io_printf(IO_BUF, "[Ensemble] Failed to start.\n");
    return;
  }

  if (!record_buffer_initialise_voltages(
        &g_ensemble.record_voltages, region_start(16, address),
        simulation_ticks, g_ensemble.n_neurons
       )
     )
  {
    io_printf(IO_BUF, "[Ensemble] Failed to start.\n");
    return;
  }

  // Setup timer tick, start
  io_printf(IO_BUF, "[Ensemble] C_MAIN Set timer and spin1_start.\n");
  spin1_set_timer_tick(g_ensemble.machine_timestep);

  while (true)
  {
    // Clear timer and packet events
    spin1_flush_rx_packet_queue();
    deschedule(TIMER_TICK);  // This shouldn't be necessary!

    // Wait for data retrieval, etc.
    event_wait();

    // Determine how long to simulate for
    config_get_n_ticks();

    // Reset the recording regions
    record_buffer_reset(&g_ensemble.record_spikes);
    record_buffer_reset(&g_ensemble.record_voltages);

    // Perform the simulation
    io_printf(IO_BUF, ">>>>> Running for %d steps\n", simulation_ticks);
    spin1_start(SYNC_WAIT);
  }
}
