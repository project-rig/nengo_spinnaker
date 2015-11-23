#include <stdbool.h>
#include "spin1_api.h"
#include "nengo-common.h"
#include "common-impl.h"

typedef struct _mc_packet_t {
  uint timestamp;
  uint key;
  uint payload;
  uint with_payload;
} mc_packet_t;

uint *start_packets, *end_packets;

void transmit_packet_region(uint* packets_region) {
  // Transmit each packet in turn
  mc_packet_t *packets = (mc_packet_t *) (&packets_region[1]);
  for (uint i = 0; i < packets_region[0]; i++) {
    spin1_send_mc_packet(packets[i].key, packets[i].payload,
                         packets[i].with_payload);
    io_printf(IO_BUF, "\tTime %d, Key 0x%08x, Payload 0x%08x\n",
              packets[i].timestamp, packets[i].key, packets[i].payload);
    spin1_delay_us(1);
  }
}

void tick(uint ticks, uint arg1) {
  use(arg1);

  if (simulation_ticks != UINT32_MAX && ticks > simulation_ticks) {
    // Transmit all packets assigned to be sent after the end of the simulation
    transmit_packet_region(end_packets);
    spin1_exit(0);
  }
}

bool get_packets(address_t source, uint** dest) {
  // Allocate some space for the packets list
  uint* dest_;

  MALLOC_FAIL_FALSE(dest_, source[0] * sizeof(mc_packet_t) + 1);
  dest[0] = dest_;

  // Copy those packets across
  spin1_memcpy(dest_, source, sizeof(uint) + sizeof(mc_packet_t) * source[0]);

  // Print all packets
  io_printf(IO_BUF, "%d packets:\n", source[0]);
  mc_packet_t *packets = (mc_packet_t *) (&source[1]);
  for (uint i = 0; i < source[0]; i++) {
    io_printf(IO_BUF, "\tTime %d, Key 0x%08x, Payload 0x%08x\n",
              packets[i].timestamp, packets[i].key, packets[i].payload);
  }

  return true;
}

void c_main(void) {
  // Load in all data
  address_t address = system_load_sram();
  if (!get_packets(region_start(2, address), &start_packets) ||
      !get_packets(region_start(4, address), &end_packets)
  ) {
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
    transmit_packet_region(start_packets);

    // Synchronise with the simulation
    spin1_start(SYNC_WAIT);
  }
}
