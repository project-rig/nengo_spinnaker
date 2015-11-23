#include "sdp-rx.h"

sdp_rx_parameters_t g_sdp_rx;

/** \brief Timer tick
 */
void sdp_rx_tick(uint ticks, uint arg1) {
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks > simulation_ticks) {
    spin1_exit(0);
    return;
  }

  for (uint d = 0; d < g_sdp_rx.n_dimensions; d++) {
    if (g_sdp_rx.fresh[d]) {
      spin1_send_mc_packet(g_sdp_rx.keys[d],
                           bitsk(g_sdp_rx.output[d]),
                           WITH_PAYLOAD);
      g_sdp_rx.fresh[d] = false;

      spin1_delay_us(1);
    }
  }
}

/** \brief Receive packed data packed in SDP message
 */
void sdp_received(uint mailbox, uint port) {
  use(port);
  sdp_msg_t *message = (sdp_msg_t*) mailbox;

  // Copy the data into the output buffer
  // Mark values as being fresh
  value_t * data = (value_t*) message->data;
  for (uint d = 0; d < g_sdp_rx.n_dimensions; d++) {
    g_sdp_rx.output[d] = data[d];
    g_sdp_rx.fresh[d] = true;
  }
  spin1_msg_free(message);
}

/** \brief Load in system parameters
 */
bool data_system(address_t addr) {
  g_sdp_rx.transmission_period = addr[0];
  g_sdp_rx.n_dimensions = addr[1];

  io_printf(IO_BUF, "[SDP Rx] Transmission period: %d\n",
            g_sdp_rx.transmission_period);
  io_printf(IO_BUF, "[SDP Rx] %d dimensions.\n", g_sdp_rx.n_dimensions);

  MALLOC_FAIL_FALSE(g_sdp_rx.output, g_sdp_rx.n_dimensions * sizeof(value_t));
  MALLOC_FAIL_FALSE(g_sdp_rx.fresh, g_sdp_rx.n_dimensions * sizeof(bool));
  MALLOC_FAIL_FALSE(g_sdp_rx.keys, g_sdp_rx.n_dimensions * sizeof(uint));

  return true;
}

/** \brief Load output keys
 */
void data_get_keys(address_t addr) {
  spin1_memcpy(g_sdp_rx.keys, addr, g_sdp_rx.n_dimensions * sizeof(uint));

  for (uint d = 0; d < g_sdp_rx.n_dimensions; d++) {
    io_printf(IO_BUF, "[SDP Rx] Key[%2d] = 0x%08x\n", d, g_sdp_rx.keys[d]);
  }
}

/** \brief Main function
 */
void c_main(void) {
  address_t address = system_load_sram();
  if (!data_system(region_start(1, address))) {
    io_printf(IO_BUF, "[Rx] Failed to initialise.\n");
    return;
  }

  data_get_keys(region_start(2, address));

  g_sdp_rx.current_dimension = 0;

  for (uint d = 0; d < g_sdp_rx.n_dimensions; d++) {
    g_sdp_rx.output[d] = 0x00000000;
    g_sdp_rx.fresh[d] = false;
  }

  // Setup timer tick, start
  spin1_set_timer_tick(g_sdp_rx.transmission_period);
  spin1_callback_on(SDP_PACKET_RX, sdp_received, -1);
  spin1_callback_on(TIMER_TICK, sdp_rx_tick, 0);

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
