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
#include "ensemble_profiler.h"
#include "fixed_point.h"


void ensemble_update(uint ticks, uint arg1) {
  use(arg1);

  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks) {
    profiler_finalise();
    spin1_exit(0);
    return;
  }

  profiler_write_entry(PROFILER_ENTER | PROFILER_TIMER);
  
  // Values used below
  current_t i_membrane;
  voltage_t v_delta, v_voltage;
  value_t inhibitory_input = 0;

  // Filter inputs, updating accumulator for excitatory and inhibitory inputs
  profiler_write_entry(PROFILER_ENTER | PROFILER_TIMER_INPUT_FILTER);
  input_filtering_step(&g_input);
  input_filtering_step(&g_input_inhibitory);
  input_filtering_step_no_accumulate(&g_input_modulatory);
  profiler_write_entry(PROFILER_EXIT | PROFILER_TIMER_INPUT_FILTER);

  // Compute the inhibition
  for (uint d = 0; d < g_ensemble.n_inhib_dims; d++)
  {
    inhibitory_input += g_input_inhibitory.output[d];
  }

  profiler_write_entry(PROFILER_ENTER | PROFILER_TIMER_NEURON);

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
    value_t encoded_input = dot_product(g_input.output_size,
                                        neuron_encoder(n),
                                        g_ensemble.input);
    i_membrane += encoded_input;

    v_voltage = neuron_voltage(n);
    v_delta = (i_membrane - v_voltage) * g_ensemble.exp_dt_over_t_rc;

    // Voltages can't go below 0.0
    v_voltage += v_delta;
    if (bitsk(v_voltage) < bitsk(0.0k))
    {
      v_voltage = 0.0k;
    }

    // Save state
    set_neuron_voltage(n, v_voltage);

    // NOTE: All `value_t` comparisons should be wrapped in `bitsk` otherwise
    // GCC inserts a function call rather than just using a CMP.
    if (bitsk(v_voltage) <= bitsk(1.0k))
    {
      // If this neuron hasn't fired then just store record voltage.
      record_voltage(&g_ensemble.record_voltages, n, v_voltage);
    }
    else
    {
      // If this neuron has fired then process:
      // We don't need to explicitly record the neuron voltage because the
      // buffer is zeroed every timestep (i.e., for the same reason that we
      // don't need to record that a spike didn't occur, or the same reason
      // that we don't store the voltages of neurons in their refractory
      // period).

      // Store the voltage for after the next refractory period and set the
      // refractory time.  The voltage after the refractory period assumes that
      // the neuron cannot experience input transients much faster than the
      // refractory period.
      register uint t_ref = g_ensemble.t_ref;
      v_voltage -= 1.0k;

      // Decrement the refractory time in the case that the overshoot was
      // sufficiently great. Also reduce the voltage that will be present on
      // exiting the refractory period.
      if (bitsk(v_voltage) > bitsk(2.0k))
      {
        t_ref--;
        v_voltage -= v_delta;
      }

      // Finally store the refractory time and voltage for later use.
      set_neuron_refractory(n, t_ref);
      set_neuron_voltage(n, v_voltage);

      // Update the output values
      value_t *decoder = neuron_decoder_vector(n);
      for (uint d = 0; d < g_n_output_dimensions; d++)
      {
        g_ensemble.output[d] += decoder[d];
      }

      // Record that the spike occurred
      record_spike(&g_ensemble.record_spikes, n);

      // Notify PES that neuron has spiked
      pes_neuron_spiked(n);
    }
  }

  profiler_write_entry(PROFILER_EXIT | PROFILER_TIMER_NEURON);

  profiler_write_entry(PROFILER_ENTER | PROFILER_TIMER_OUTPUT);

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

  profiler_write_entry(PROFILER_EXIT | PROFILER_TIMER_OUTPUT);

  // Flush the recording buffers
  record_buffer_flush(&g_ensemble.record_spikes);
  record_buffer_flush(&g_ensemble.record_voltages);

  profiler_write_entry(PROFILER_EXIT | PROFILER_TIMER);
}
