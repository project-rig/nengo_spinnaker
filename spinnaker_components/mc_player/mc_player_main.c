#include <stdbool.h>
#include "spin1_api.h"
#include "nengo-common.h"
#include "common-impl.h"

typedef struct _mc_packet_t
{
  uint32_t timestamp;
  uint32_t key;
  uint32_t payload;
  uint32_t with_payload;
} mc_packet_t;

static uint num_start_packets = 0;
static mc_packet_t *start_packets = NULL;

static uint num_end_packets = 0;
static mc_packet_t *end_packets = NULL;

void transmit_packet_region(uint num_packets, mc_packet_t *packets)
{
  // Transmit each packet in turn
  for (uint i = 0; i < num_packets; i++)
  {
    spin1_send_mc_packet(packets[i].key, packets[i].payload,
                         packets[i].with_payload);
    io_printf(IO_BUF, "\tTime %d, Key 0x%08x, Payload 0x%08x\n",
              packets[i].timestamp, packets[i].key, packets[i].payload);
    spin1_delay_us(1);
  }
}

void tick(uint ticks, uint arg1)
{
  use(arg1);

  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks)
  {
    // Transmit all packets assigned to be sent after the end of the simulation
    transmit_packet_region(num_end_packets, end_packets);
    spin1_exit(0);
  }
}

bool get_packets(address_t source, uint *dest_num_packets, mc_packet_t **dest_packets)
{
  // Read number of packets from first word
  uint num_packets = (uint)source[0];

  // Allocate memory for packets
  mc_packet_t *packets;
  MALLOC_FAIL_FALSE(packets, num_packets * sizeof(mc_packet_t));

  // Copy those packets across
  spin1_memcpy(packets, &source[1], num_packets * sizeof(mc_packet_t));

  // Print all packets
  io_printf(IO_BUF, "%d packets:\n", num_packets);
  for (uint i = 0; i < num_packets; i++)
  {
    io_printf(IO_BUF, "\tTime %d, Key 0x%08x, Payload 0x%08x\n",
              packets[i].timestamp, packets[i].key, packets[i].payload);
  }

  uint8_t *source_bytes = (uint8_t*)source;
  for(uint b = 0; b < (sizeof(uint32_t) + (num_packets * sizeof(mc_packet_t))); b++)
  {
    io_printf(IO_BUF, "%x,", source_bytes[b]);
  }
  io_printf(IO_BUF,"\n");
  
  *dest_num_packets = num_packets;
  *dest_packets = packets;

  return true;
}

void c_main()
{
  // Load in all data
  address_t address = system_load_sram();
  if (!get_packets(region_start(2, address), &num_start_packets, &start_packets) ||
      !get_packets(region_start(4, address), &num_end_packets, &end_packets))
  {
    return;
  }

  spin1_set_timer_tick(1000);
  spin1_callback_on(TIMER_TICK, tick, 2);

  while(true)
  {
    // Wait for data loading, etc.
    event_wait();

    // Determine how long to simulate for
    config_get_n_ticks();

    // Transmit all packets assigned to be sent prior to the start of the
    // simulation
    transmit_packet_region(num_start_packets, start_packets);

    // Synchronise with the simulation
    spin1_start(SYNC_WAIT);
  }
}
