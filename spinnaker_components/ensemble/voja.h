/**
 * Ensemble - Voja
 * -----------------
 * Functions to perform Voja encoder learning
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


#ifndef __ENSEMBLE_VOJA_H__
#define __ENSEMBLE_VOJA_H__

#include "ensemble.h"

//----------------------------------
// Structs
//----------------------------------
// Structure containing parameters and state required for Voja learning
typedef struct voja_parameters_t
{
  // Scalar learning rate used in Voja encoder delta calculation
  value_t learning_rate;

  // Index of the input signal filter that contains
  // learning signal. -1 if there is no learning signal
  int32_t learning_signal_filter_index;

  // Offset into encoder to apply Voja
  uint32_t encoder_offset;

  // Index of the input signal filter than contains
  // the decoded input from the pre-synaptic ensemble
  uint32_t decoded_input_filter_index;

  // Index of the activity filter to extract input from
  // -1 if this learning rule should use unfiltered activity
  int32_t activity_filter_index;
} voja_parameters_t;

//----------------------------------
// External variables
//----------------------------------
extern uint32_t g_num_voja_learning_rules;
extern voja_parameters_t *g_voja_learning_rules;
extern value_t g_voja_one_over_radius;

//----------------------------------
// Inline functions
//----------------------------------
/**
* \brief Helper to get the Voja learning rate - can be modified at runtime with a signal
*/
static inline value_t voja_get_learning_rate(const voja_parameters_t *parameters)
{
    // If a learning signal filter index is specified, read the value
    // from it's first dimension and multiply by the constant error rate
    if(parameters->learning_signal_filter_index != -1)
    {
      value_t positive_learning_rate = 1.0k + g_input_modulatory.filters[parameters->learning_signal_filter_index]->filtered[0];
      return parameters->learning_rate * positive_learning_rate;
    }
    // Otherwise, just return the constant learning rate
    else
    {
      return parameters->learning_rate;
    }
}

/**
* \brief When using non-filtered activity, applies Voja when neuron spikes
*/
static inline void voja_neuron_spiked(uint n)
{
  // Loop through all the learning rules
  for(uint32_t l = 0; l < g_num_voja_learning_rules; l++)
  {
    // If this learning rule operates on un-filtered activity and should, therefore be updated here
    const voja_parameters_t *parameters = &g_voja_learning_rules[l];
    if(parameters->activity_filter_index == -1)
    {
      // Get learning rate
      const value_t learning_rate = voja_get_learning_rate(parameters);

      // Extract decoded input signal from filter
      const filtered_input_buffer_t *decoded_input = g_input_learnt_encoder.filters[parameters->decoded_input_filter_index];
      const value_t *decoded_input_signal = decoded_input->filtered;

      // Get this neuron's encoder vector, offset by the encoder offset
      value_t *encoder_vector = neuron_encoder_vector(n) + parameters->encoder_offset;

      // Calculate scaling factor for input
      const value_t input_scale = learning_rate * g_ensemble.gain[n] * g_voja_one_over_radius;

      // Loop through input dimensions
      for(uint d = 0; d < decoded_input->d_in; d++)
      {
        encoder_vector[d] += (input_scale * decoded_input_signal[d]) - (learning_rate * encoder_vector[d]);
      }
    }
  }
}

//----------------------------------
// Functions
//----------------------------------
/**
* \brief Copy in data controlling the Voja learning
* rule from the Voja region of the Ensemble.
*/
bool get_voja(address_t address);

/**
* \brief Apply voja learning to encoders
*/
void voja_step();

/** @} */

#endif  // __ENSEMBLE_VOJA_H__