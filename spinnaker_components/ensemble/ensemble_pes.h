/**
 * Ensemble - PES
 * -----------------
 * Functions to perform PES decoder learning
 * 
 * Authors:
 *   - James Knight <knightj@cs.man.ac.uk>
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 * 
 * \addtogroup ensemble
 * @{
 */


#ifndef __ENSEMBLE_PES_H__
#define __ENSEMBLE_PES_H__

#include "ensemble.h"

//----------------------------------
// Structs
//----------------------------------
// Structure containing parameters and state required for PES learning
typedef struct pes_parameters_t
{
  // Scalar learning rate used in PES decoder delta calculation
  value_t learning_rate;
  
  // Index of the input signal filter that contains error signal
  uint32_t error_signal_filter_index;
  
  // Offset into decoder to apply PES
  uint32_t decoder_output_offset;
} pes_parameters_t;

//----------------------------------
// External variables
//----------------------------------
extern uint32_t g_num_pes_learning_rules;
extern pes_parameters_t *g_pes_learning_rules;

//----------------------------------
// Inline functions
//----------------------------------
/**
* \brief When using non-filtered activity, applies PES when neuron spikes
*/
static inline void pes_neuron_spiked(uint n)
{
  // Loop through all the learning rules
  for(uint32_t l = 0; l < g_num_pes_learning_rules; l++)
  {
    const pes_parameters_t *parameters = &g_pes_learning_rules[l];
    if(parameters->learning_rate > 0.0k)
    {
      // Extract error signal vector from 
      const filtered_input_buffer_t *filtered_input = g_input_modulatory.filters[parameters->error_signal_filter_index];
      const value_t *filtered_error_signal = filtered_input->filtered;
      
      // Get filtered activity of this neuron and it's decoder vector
      value_t *decoder_vector = neuron_decoder_vector(n);
      
      // Loop through output dimensions and apply PES to decoder values offset by output offset
      for(uint d = 0; d < filtered_input->d_in; d++) 
      {
        decoder_vector[d + parameters->decoder_output_offset] += (parameters->learning_rate * filtered_error_signal[d]);
      }
    }
  }
}

//----------------------------------
// Functions
//----------------------------------
/**
* \brief Copy in data controlling the PES learning 
* rule to the PES region of the Ensemble.
*/
bool get_pes(address_t address);

/** @} */

#endif  // __ENSEMBLE_PES_H__