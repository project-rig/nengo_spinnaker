#include "spin1_api.h"
#include "common-typedefs.h"
#include "common-impl.h"
#include "input_filtering.h"

typedef struct _region_system_t
{
  uint32_t timestep;
  uint32_t input_size;
  uint32_t input_offset;
} region_system_t;
region_system_t params;

address_t rec_start, rec_curr;

if_collection_t filters;

void sink_update(uint ticks, uint arg1)
{
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks > simulation_ticks)
  {
    spin1_exit(0);
    return;
  }

  // Filter inputs, write the latest value to SRAM
  input_filtering_step(&filters);
  spin1_memcpy(rec_curr, filters.output, params.input_size * sizeof(value_t));
  rec_curr = &rec_curr[params.input_size];
}

void mcpl_callback(uint key, uint payload)
{
  input_filtering_input_with_dimension_offset(
    &filters, key, payload,
    params.input_offset,   // Offset for all packets
    params.input_size - 1  // Max expected dimension
  );
}

void c_main(void)
{
  address_t address = system_load_sram();

  // Load parameters
  spin1_memcpy(&params, region_start(1, address), sizeof(region_system_t));

  // Prepare filtering
  input_filtering_initialise_output(&filters, params.input_size);
  input_filtering_get_filters(&filters, region_start(2, address), NULL);
  input_filtering_get_routes(&filters, region_start(3, address));

  // Retrieve the recording region
  rec_start = region_start(15, address);

  // Set up callbacks, start
  spin1_set_timer_tick(params.timestep);
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
