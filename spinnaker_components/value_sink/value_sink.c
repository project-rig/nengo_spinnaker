#include "spin1_api.h"
#include "common-typedefs.h"
#include "common-impl.h"
#include "input_filtering.h"
#include "packet_queue.h"

typedef struct _region_system_t
{
  uint32_t timestep;
  uint32_t input_size;
  uint32_t input_offset;
} region_system_t;
region_system_t params;

address_t rec_start, rec_curr;

if_collection_t filters;
if_routing_table_t filter_routing;

static packet_queue_t packets;  // Queued multicast packets
static bool queue_processing;   // Indicate if the queue is being handled
static unsigned int queue_overflows;

void mcpl_callback(uint key, uint payload)
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

void sink_update(uint ticks, uint arg1)
{
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks > simulation_ticks)
  {
    spin1_exit(0);
    return;
  }

  // Process any remaining unprocessed packets
  process_queue();

  // Filter inputs, write the latest value to SRAM
  input_filtering_step(&filters);
  spin1_memcpy(rec_curr, filters.output, params.input_size * sizeof(value_t));
  rec_curr = &rec_curr[params.input_size];
}

void c_main(void)
{
  address_t address = system_load_sram();

  // Load parameters
  spin1_memcpy(&params, region_start(1, address), sizeof(region_system_t));

  // Prepare filtering
  input_filtering_initialise_output(&filters, params.input_size);
  input_filtering_get_filters(&filters, region_start(2, address), NULL);
  input_filtering_get_routes(&filters, &filter_routing,
                             (filter_routes_t *) region_start(3, address));

  // Retrieve the recording region
  rec_start = region_start(15, address);

  // Multicast packet queue
  queue_processing = false;
  packet_queue_init(&packets, 1024);
  queue_overflows = 0;

  // Set up callbacks, start
  spin1_set_timer_tick(params.timestep);
  spin1_callback_on(MCPL_PACKET_RECEIVED,  mcpl_callback, -1);
  spin1_callback_on(TIMER_TICK, sink_update, 2);
  spin1_callback_on(USER_EVENT, user_event, 2);

  while(true)
  {
    // Wait for data loading, etc.
    event_wait();

    // Determine how long to simulate for
    config_get_n_ticks();

    // Reset the recording region location
    rec_curr = rec_start;

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
