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


#ifndef __VOJA_H__
#define __VOJA_H__

// Common includes
#include "common-typedefs.h"
#include "input_filtering.h"

// Ensemble includes
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
static inline value_t voja_get_learning_rate(const voja_parameters_t *parameters,
                                             const if_collection_t *modulatory_filters)
{
    // If a learning signal filter index is specified, read the value
    // from it's first dimension and multiply by the constant error rate
    if(parameters->learning_signal_filter_index != -1)
    {
      const if_filter_t *decoded_learning_input = &modulatory_filters->filters[parameters->learning_signal_filter_index];
      value_t positive_learning_rate = 1.0k + decoded_learning_input->output[0];
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
static inline void voja_neuron_spiked(value_t *encoder_vector, value_t gain, uint32_t n_dims,
                                      const if_collection_t *modulatory_filters,
                                      const value_t **learnt_input)
{
  profiler_write_entry(PROFILER_ENTER | PROFILER_VOJA);

  // Loop through all the learning rules
  for(uint32_t l = 0; l < g_num_voja_learning_rules; l++)
  {
    // If this learning rule operates on un-filtered activity and should, therefore be updated here
    const voja_parameters_t *parameters = &g_voja_learning_rules[l];
    if(parameters->activity_filter_index == -1)
    {
      // Get learning rate
      const value_t learning_rate = voja_get_learning_rate(parameters, modulatory_filters);

      // Get correct signal from learnt input
      const value_t *decoded_input_signal = learnt_input[parameters->decoded_input_filter_index];

      // Get this neuron's encoder vector, offset by the encoder offset
      value_t *learnt_encoder_vector = encoder_vector + parameters->encoder_offset;

      // Calculate scaling factor for input
      const value_t input_scale = learning_rate * gain * g_voja_one_over_radius;

      // Loop through input dimensions
      for(uint d = 0; d < n_dims; d++)
      {
        learnt_encoder_vector[d] += (input_scale * decoded_input_signal[d]) - (learning_rate * learnt_encoder_vector[d]);
      }
    }
  }

  profiler_write_entry(PROFILER_EXIT | PROFILER_VOJA);
}

//----------------------------------
// Functions
//----------------------------------
/**
* \brief Copy in data controlling the Voja learning
* rule from the Voja region of the Ensemble.
*/
bool voja_initialise(address_t address);

/**
* \brief Apply voja learning to encoders
*/
//void voja_step(const if_collection_t *modulatory_filters,
//               const if_collection_t *learnt_encoder_filters,
//               const value_t *gain);

/** @} */

#endif  // __VOJA_H__