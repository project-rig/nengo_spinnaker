/* Parallel implementation of the filter operator
 * ----------------------------------------------
 *
 * Each core in the parallel filter is responsible for receiving packets for a
 * subspace of the value represented by the filter and for transmitting
 * transformed packets for a different subspace.  The general mode of operation
 * on the timer update is:
 *
 *  1. Filter received values
 *  2. Apply local portion of the transform and transmit transformed packets
 *
 * ---
 *
 * The following SDRAM regions are expected:
 *
 *  1. System region (see `filter_parameters_t`)
 *  2. Output keys
 *  3. Filter parameters
 *  4. Filter routes
 *  5. Transform
 *
 * ---
 */

/*****************************************************************************/
#include <stdint.h>
#include "spin1_api.h"
#include "nengo_typedefs.h"
#include "fixed_point.h"
#include "input_filtering.h"
#include "common-impl.h"
#include "packet_queue.h"

/*****************************************************************************/
// Global variables
// General parameters (system region)
typedef struct filter_parameters
{
  uint32_t machine_timestep;  // Machine time step / useconds
  uint32_t input_size;        // Number of columns
  uint32_t input_offset;      // Offset input subspace
  uint32_t output_size;       // Number of rows
} filter_parameters_t;

static filter_parameters_t params;
static if_collection_t filters;  // Locally applied filters
static if_routing_table_t filter_routing;
static value_t *transform;       // Transform matrix
uint32_t *keys;                  // Multicast keys

static packet_queue_t packets;  // Queued multicast packets
static bool queue_processing;   // Indicate if the queue is being handled
static unsigned int queue_overflows;
/*****************************************************************************/

/*****************************************************************************/
// Multicast packet handling
void multicast_packet_payload(uint key, uint payload)
{
  // Queue the packet for later processing, if no processing is scheduled then
  // trigger the queue processor.
  if (packet_queue_push(&packets, key, payload))
  {
    if (!queue_processing)
    {
      spin1_trigger_user_event(0, 0);
      queue_processing = true;
    }
  }
  else
  {
    // The packet couldn't be included in the queue, thus it was essentially
    // dropped.
    queue_overflows++;
  }
}

void process_queue()
{
  // Continuously remove packets from the queue and include them in filters
  while (packet_queue_not_empty(&packets))
  {
    // Pop a packet from the queue (critical section)
    packet_t packet;
    uint cpsr = spin1_fiq_disable();
    bool packet_is_valid = packet_queue_pop(&packets, &packet);
    spin1_mode_restore(cpsr);

    // Process the received packet
    if (packet_is_valid)
    {
      uint32_t key = packet.key;
      uint32_t payload = packet.payload;

      input_filtering_input_with_dimension_offset(
        &filter_routing, key, payload,
        params.input_offset,   // Offset for all packets
        params.input_size - 1  // Max expected dimension
      );
    }
    else
    {
      io_printf(IO_BUF, "Popped packet from empty queue.\n");
      rt_error(RTE_ABORT);
    }
  }
  queue_processing = false;
}

void user_event(uint arg0, uint arg1)
{
  use(arg0);
  use(arg1);

  process_queue();
}
/*****************************************************************************/

/*****************************************************************************/
// Timer tick
void timer_tick(uint ticks, uint arg1)
{
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks)
  {
    spin1_exit(0);
    return;
  }

  // Process any remaining unprocessed packets
  process_queue();

  // Update the filters
  input_filtering_step(&filters);

  // Perform the matrix multiply, transmitting each output value as it is
  // computed.
  for (unsigned int i = 0; i < params.output_size; i++)
  {
    // Get the desired row of the matrix
    value_t *row = transform + i*params.input_size;

    // Get the output key
    uint32_t key = keys[i];

    // Perform the dot-product and transmit the packet
    value_t output = dot_product(params.input_size, row, filters.output);

    while (!spin1_send_mc_packet(key, bitsk(output), WITH_PAYLOAD))
    {
    }
  }
}
/*****************************************************************************/

/*****************************************************************************/
void c_main(void)
{
  // Start loading data from SDRAM
  address_t address = system_load_sram();

  // Copy in the parameters
  spin1_memcpy(&params, region_start(1, address),
               sizeof(filter_parameters_t));

  // Copy in the keys
  uint key_size = params.output_size * sizeof(uint32_t);
  MALLOC_OR_DIE(keys, key_size);
  spin1_memcpy(keys, region_start(2, address), key_size);

  // Copy in the transform
  uint matrix_size = params.input_size * params.output_size * sizeof(value_t);
  MALLOC_OR_DIE(transform, matrix_size);
  spin1_memcpy(transform, region_start(5, address), matrix_size);

  // Prepare the filters for receiving packets
  input_filtering_get_filters(&filters, region_start(3, address), NULL);
  input_filtering_get_routes(&filters, &filter_routing,
                             (filter_routes_t *) region_start(4, address));
  input_filtering_initialise_output(&filters, params.input_size);

  // Multicast packet queue
  queue_processing = false;
  packet_queue_init(&packets, 1024);
  queue_overflows = 0;

  // Register callbacks
  spin1_set_timer_tick(params.machine_timestep);
  spin1_callback_on(MCPL_PACKET_RECEIVED, multicast_packet_payload, -1);
  spin1_callback_on(TIMER_TICK, timer_tick, 1);
  spin1_callback_on(USER_EVENT, user_event, 1);

  while (true)
  {
    // Wait for data loading, etc.
    event_wait();

    // Determine how long to simulate for
    config_get_n_ticks();

    // Check on the status of the packet queue
    if (queue_overflows)
    {
      io_printf(IO_BUF, "Queue overflows = %u\n", queue_overflows);
      rt_error(RTE_ABORT);
    }

    // Perform the simulation
    spin1_start(SYNC_WAIT);
  }
}
/*****************************************************************************/
