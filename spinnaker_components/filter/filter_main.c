#include <stdint.h>

#include "spin1_api.h"
#include "input_filtering.h"
#include "nengo-common.h"
#include "fixed_point.h"

#include "common-impl.h"

/*****************************************************************************/
// Filter parameters (and system region)
typedef struct filter_parameters
{
  uint32_t size_in;             //!< Number of columns in transform
  uint32_t size_out;            //!< Number of rows in transform
  uint32_t machine_timestep;    //!< Machine time step / useconds
  uint32_t transmission_delay;  //!< Ticks between output transmissions
  uint32_t interpacket_pause;   //!< usecs between transmitting packets
} filter_parameters_t;
filter_parameters_t g_filter;

uint delay_remaining;  // Number of ticks until packets are next transmitted
if_collection_t g_filters;  // Collection of input filters
value_t *transform;   // Transform matrix to apply
uint32_t *keys;       // Multicast keys
/*****************************************************************************/

// Get a pointer to the ith row of the transform matrix
inline value_t *get_transform_row(uint i)
{
  return &transform[i * g_filter.size_in];
}

/*****************************************************************************/
// Timer tick
void filter_update(uint ticks, uint arg1)
{
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks)
  {
    spin1_exit(0);
  }

  // Update the filters
  input_filtering_step(&g_filters);

  // Increment the counter and apply the transform and transmit if necessary
  if (--delay_remaining == 0)
  {
    // Reset the remaining delay
    delay_remaining = g_filter.transmission_delay;

    // For every output value first compute the value and then transmit
    for (uint d = 0; d < g_filter.size_out; d++)
    {
      // Get the row of the transform matrix
      value_t *row = get_transform_row(d);

      // Calculate the output value
      value_t output = dot_product(g_filter.size_in, row, g_filters.output);

      // Transmit this value
      spin1_send_mc_packet(keys[d], bitsk(output), WITH_PAYLOAD);
      spin1_delay_us(g_filter.interpacket_pause);
    }
  }
}
/*****************************************************************************/

/*****************************************************************************/
// Retrieve the multicast packet keys from SDRAM
bool data_get_output_keys(address_t addr)
{
  // Allocate space for the keys and then copy in from SDRAM
  MALLOC_FAIL_FALSE(keys, g_filter.size_out * sizeof(uint));
  spin1_memcpy(keys, addr, g_filter.size_out * sizeof(uint));
  return true;
}
/*****************************************************************************/

/*****************************************************************************/
// Retrieve the transform from SDRAM
// TODO Page in the filter as required rather than copying it directly.
bool data_get_transform(address_t addr)
{
  // Allocate space for the transform and then copy it across from SDRAM
  MALLOC_FAIL_FALSE(transform,
                    sizeof(value_t) * g_filter.size_in * g_filter.size_out);
  spin1_memcpy(transform, addr,
               sizeof(value_t) * g_filter.size_in * g_filter.size_out);

  return true;
}
/*****************************************************************************/

/*****************************************************************************/
// Multicast Packet with Payload callback
void mcpl_callback(uint key, uint payload)
{
  // Include the packet in the filter inputs
  input_filtering_input(&g_filters, key, payload);
}
/*****************************************************************************/

/*****************************************************************************/
// Initialisation code
void c_main(void)
{
  // Start loading data from SDRAM
  address_t address = system_load_sram();

  // Copy in the parameters
  spin1_memcpy(&g_filter, region_start(1, address),
               sizeof(filter_parameters_t));

  if (!data_get_output_keys(region_start(2, address)) ||
      !data_get_transform(region_start(5, address)))
  {
    io_printf(IO_BUF, "[Filter] Failed to initialise.\n");
    return;
  }

  // Prepare the filters for receiving packets
  input_filtering_get_filters(&g_filters, region_start(3, address));
  input_filtering_get_routes(&g_filters, region_start(4, address));
  input_filtering_initialise_output(&g_filters, g_filter.size_in);

  // Set the initial delay
  delay_remaining = g_filter.transmission_delay;

  // Setup timer tick, start
  spin1_set_timer_tick(g_filter.machine_timestep);
  spin1_callback_on(MCPL_PACKET_RECEIVED, mcpl_callback, -1);
  spin1_callback_on(TIMER_TICK, filter_update, 2);

  while (true)
  {
    // Wait for data loading, etc.
    event_wait();

    // Determine how long to simulate for
    config_get_n_ticks();

    // Perform the simulation
    spin1_start(SYNC_WAIT);
  }
}
/*****************************************************************************/
