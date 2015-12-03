/*
 * Ensemble - Data
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

#include "ensemble_pes.h"
#include "ensemble_filtered_activity.h"

#include <string.h>

//----------------------------------
// Global variables
//----------------------------------
uint32_t g_num_pes_learning_rules = 0;
pes_parameters_t *g_pes_learning_rules = NULL;

//----------------------------------
// Global functions
//----------------------------------
bool get_pes(address_t address)
{
  // Read number of PES learning rules that are configured
  g_num_pes_learning_rules = address[0];
  
  io_printf(IO_BUF, "PES learning: Num rules:%u\n", g_num_pes_learning_rules);
  
  if(g_num_pes_learning_rules > 0)
  {
    // Allocate memory
    MALLOC_FAIL_FALSE(g_pes_learning_rules,
                      g_num_pes_learning_rules * sizeof(pes_parameters_t));
    
    // Copy learning rules from region into new array
    memcpy(g_pes_learning_rules, &address[1], g_num_pes_learning_rules * sizeof(pes_parameters_t));
    
    // Display debug
    for(uint32_t l = 0; l < g_num_pes_learning_rules; l++)
    {
      const pes_parameters_t *parameters = &g_pes_learning_rules[l];
      io_printf(IO_BUF, "\tRule %u, Learning rate:%k, Error signal filter index:%u, Decoder output offset:%u, Activity filter index:%d\n",
               l, parameters->learning_rate, parameters->error_signal_filter_index, parameters->decoder_output_offset, parameters->activity_filter_index);
    }
  }
  return true;
}
//----------------------------------
void pes_step()
{
  // Loop through all the learning rules
  for(uint32_t l = 0; l < g_num_pes_learning_rules; l++)
  {
    // Extract filtered error signal vector indexed by learning rule
    const pes_parameters_t *parameters = &g_pes_learning_rules[l];

    // If this learning rule operates on filtered activity and should, therefore be updated here
    if(parameters->activity_filter_index != -1)
    {
      // Extract input signal from filter
      const filtered_input_buffer_t *filtered_input = g_input_modulatory.filters[parameters->error_signal_filter_index];
      const value_t *filtered_error_signal = filtered_input->filtered;

      // Extract filtered activity vector indexed by learning rule
      const value_t *filtered_activity = g_filtered_activities[parameters->activity_filter_index];

      // Loop through neurons
      for(uint n = 0; n < g_ensemble.n_neurons; n++)
      {
        // Get this neuron's decoder vector
        value_t *decoder_vector = neuron_decoder_vector(n);

        // Loop through output dimensions and apply PES to decoder values offset by output offset
        for(uint d = 0; d < filtered_input->d_in; d++)
        {
          decoder_vector[d + parameters->decoder_output_offset] -= (parameters->learning_rate * filtered_activity[n] * filtered_error_signal[d]);
        }
      }
    }
  }
}