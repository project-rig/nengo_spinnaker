#include "sdp_tx.h"

sdp_tx_parameters_t g_sdp_tx;
uint delay_remaining;
if_collection_t g_input;

void sdp_tx_update(uint ticks, uint arg1) {
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks) {
    spin1_exit(0);
    return;
  }

  // Update the filters
  input_filtering_step(&g_input);

  // Increment the counter and transmit if necessary
  delay_remaining--;
  if(delay_remaining == 0) {
    delay_remaining = g_sdp_tx.transmission_delay;

    // Construct and transmit the SDP Message
    sdp_msg_t message;
    message.dest_addr = 0x0000;        // (0, 0)
    message.dest_port = 0xff;
    message.srce_addr = sv->p2p_addr;  // Sender P2P address
    message.srce_port = spin1_get_id();
    message.flags = 0x07;              // No reply expected
    message.tag = 1;                   // Send to IPtag 1

    message.cmd_rc = 1;
    spin1_memcpy(
      message.data, g_sdp_tx.input, g_sdp_tx.n_dimensions * sizeof(value_t));

    message.length = sizeof(sdp_hdr_t) + sizeof(cmd_hdr_t) + 
                     g_sdp_tx.n_dimensions * sizeof(value_t);

    spin1_send_sdp_msg(&message, 100);
  }
}

bool data_system(address_t addr) {
  g_sdp_tx.n_dimensions = addr[0];
  g_sdp_tx.machine_timestep = addr[1];
  g_sdp_tx.transmission_delay = addr[2];

  delay_remaining = g_sdp_tx.transmission_delay;
  io_printf(IO_BUF, "[SDP Tx] Tick period = %d microseconds\n",
            g_sdp_tx.machine_timestep);
  io_printf(IO_BUF, "[SDP Tx] transmission delay = %d\n", delay_remaining);

  input_filtering_initialise_output(&g_input, g_sdp_tx.n_dimensions);
  g_sdp_tx.input = g_input.output;

  if (g_sdp_tx.input == NULL)
    return false;
  return true;
}

void mcpl_callback(uint key, uint payload) {
  input_filtering_input(&g_input, key, payload);
}

void c_main(void) {
  address_t address = system_load_sram();
  input_filtering_get_filters(&g_input, region_start(2, address));
  input_filtering_get_routes(&g_input, region_start(3, address));
  if (!data_system(region_start(1, address)))
  {
    io_printf(IO_BUF, "[Tx] Failed to initialise.\n");
    return;
  }

  // Setup timer tick, start
  spin1_set_timer_tick(g_sdp_tx.machine_timestep);
  spin1_callback_on(MCPL_PACKET_RECEIVED, mcpl_callback, -1);
  spin1_callback_on(TIMER_TICK, sdp_tx_update, 2);

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
