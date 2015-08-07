/*
 * Authors:
 *   - Andrew Mundy <mundya@cs.man.ac.uk>
 *   - Terry Stewart
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 */

#include "ensemble.h"
#include "ensemble_output.h"
#include "ensemble_pes.h"

void ensemble_update(uint ticks, uint arg1) {
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks) {
    spin1_exit(0);
  }

  // Values used below
  current_t i_membrane;
  voltage_t v_delta, v_voltage;
  value_t inhibitory_input = 0;
  value_t encoder_d;

  // Filter inputs, updating accumulator for excitatory and inhibitory inputs
  input_filter_step(&g_input, true);
  input_filter_step(&g_input_inhibitory, true);
  input_filter_step(&g_input_modulatory, false);

  // Compute the inhibition
  for (uint d = 0; d < g_ensemble.n_inhib_dims; d++) {
    inhibitory_input += g_input_inhibitory.input[d];
  }

  // Perform neuron updates
  for( uint n = 0; n < g_ensemble.n_neurons; n++ ) {
    // If this neuron is refractory then skip any further processing
    if( neuron_refractory( n ) != 0 ) {
      decrement_neuron_refractory( n );
      continue;
    }

    // Include neuron bias
    i_membrane = (g_ensemble.i_bias[n] +
                  inhibitory_input * g_ensemble.inhib_gain[n]);

    // Encode the input and add to the membrane current
    for( uchar d = 0; d < g_input.n_dimensions; d++ ) {
      encoder_d = neuron_encoder(n, d);
      i_membrane += encoder_d * g_ensemble.input[d];
    }

    v_voltage = neuron_voltage(n);
    v_delta = (i_membrane - v_voltage) * g_ensemble.exp_dt_over_t_rc;

    // Voltages can't go below 0.0
    v_voltage += v_delta;
    if(v_voltage < 0.0k)
    {
      v_voltage = 0.0k;
    }

    // Save state
    set_neuron_voltage(n, v_voltage);

    // If this neuron has fired then process
    if( v_voltage > 1.0k ) {
      // Set the voltage to be the overshoot, set the refractory time
      set_neuron_refractory(n);
      set_neuron_voltage(n, v_voltage - 1.0k);

      // Decrement the refractory time in the case that the overshoot was
      // sufficiently significant.
      if(v_voltage > 2.0k)
      {
        decrement_neuron_refractory(n);
        set_neuron_voltage(n, v_voltage - 1.0k - v_delta);
      }

      // Update the output values
      for( uint d = 0; d < g_n_output_dimensions; d++ ) {
        g_ensemble.output[d] += neuron_decoder( n, d );
      }

      // Record that the spike occurred
      record_spike(&g_ensemble.record_spikes, n);

      // Notify PES that neuron has spiked
      pes_neuron_spiked(n);

      // Ensure the voltage is zeroed before we record it
      v_voltage = 0.0k;
    }

    // Record the neuron voltage
    record_voltage(&g_ensemble.record_voltages, n, v_voltage);
  }

  // Transmit decoded Ensemble representation
  for (uint output_index = 0; output_index < g_n_output_dimensions;
       output_index++) {
    while(!spin1_send_mc_packet(gp_output_keys[output_index],
                                bitsk(gp_output_values[output_index]),
                                WITH_PAYLOAD))
    {
      spin1_delay_us(1);
    }
    gp_output_values[output_index] = 0;
  }

  // Flush the recording buffers
  record_buffer_flush(&g_ensemble.record_spikes);
  record_buffer_flush(&g_ensemble.record_voltages);
}
