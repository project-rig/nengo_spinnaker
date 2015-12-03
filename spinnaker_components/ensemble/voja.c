/*
 * Ensemble - Voja
 *
 * Authors:
 *   - James Knight <knightj@cs.man.ac.uk>
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 * 
 */

#include "ensemble_voja.h"
#include "ensemble_filtered_activity.h"

#include <string.h>

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
uint32_t g_num_voja_learning_rules = 0;
voja_parameters_t *g_voja_learning_rules = NULL;
value_t g_voja_one_over_radius = 1.0k;

//-----------------------------------------------------------------------------
// Global functions
//-----------------------------------------------------------------------------
bool get_voja(address_t address)
{
  // Read number of Voja learning rules that are configured and the scaling factor
  g_num_voja_learning_rules = address[0];
  g_voja_one_over_radius = kbits(address[1]);

  io_printf(IO_BUF, "Voja learning: Num rules:%u, One over radius:%k\n",
            g_num_voja_learning_rules, g_voja_one_over_radius);
  
  if(g_num_voja_learning_rules > 0)
  {
    // Allocate memory
    MALLOC_FAIL_FALSE(g_voja_learning_rules,
                      g_num_voja_learning_rules * sizeof(voja_parameters_t));
    
    // Copy learning rules from region into new array
    memcpy(g_voja_learning_rules, &address[2], g_num_voja_learning_rules * sizeof(voja_parameters_t));
    
    // Display debug
    for(uint32_t l = 0; l < g_num_voja_learning_rules; l++)
    {
      const voja_parameters_t *parameters = &g_voja_learning_rules[l];
      io_printf(IO_BUF, "\tRule %u, Learning rate:%k, Learning signal filter index:%d, Encoder output offset:%u, Decoded input filter index:%u, Activity filter index:%d\n",
                l, parameters->learning_rate, parameters->learning_signal_filter_index, parameters->encoder_offset, parameters->decoded_input_filter_index, parameters->activity_filter_index);
    }
  }
  return true;
}
//-----------------------------------------------------------------------------
void voja_step()
{
  // Loop through all the learning rules
  for(uint32_t l = 0; l < g_num_voja_learning_rules; l++)
  {
    // If this learning rule operates on filtered activity and should, therefore be updated here
    const voja_parameters_t *parameters = &g_voja_learning_rules[l];
    if(parameters->activity_filter_index != -1)
    {
      // Get learning rate
      const value_t learning_rate = voja_get_learning_rate(parameters);

      // Extract decoded input signal from filter
      const filtered_input_buffer_t *decoded_input = g_input_learnt_encoder.filters[parameters->decoded_input_filter_index];
      const value_t *decoded_input_signal = decoded_input->filtered;

      // Extract filtered activity vector indexed by learning rule
      const value_t *filtered_activity = g_filtered_activities[parameters->activity_filter_index];

      // Loop through neurons
      for(uint n = 0; n < g_ensemble.n_neurons; n++)
      {
        // Get this neuron's encoder vector, offset by the encoder offset
        value_t *encoder_vector = neuron_encoder_vector(n) + parameters->encoder_offset;

        // Calculate scaling factors for the two terms
        const value_t encoder_scale = learning_rate * filtered_activity[n];
        const value_t input_scale = encoder_scale * g_ensemble.gain[n] * g_voja_one_over_radius;

        // Loop through input dimensions
        for(uint d = 0; d < decoded_input->d_in; d++)
        {
          encoder_vector[d] += (input_scale * decoded_input_signal[d]) - (encoder_scale * encoder_vector[d]);
        }
      }
    }
  }
}