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

#include <string.h>

//----------------------------------
// Structs
//----------------------------------
struct pes_learning_rule_t
{
  // Scalar learning rate (scaled by dt) used in PES decoder delta calculation
  value_t learning_rate;
  
  // Index of the input signal filter that contains error signal
  uint32_t error_signal_filter_index;
  
  // Offset into decoder to apply PES
  uint32_t decoder_output_offset;
};

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
      io_printf(IO_BUF, "\tRule %u, Learning rate:%k, Error signal filter index:%u, Decoder output offset:%u\n", 
               l, parameters->learning_rate, parameters->error_signal_filter_index, parameters->decoder_output_offset);
    }
  }
  return true;
}
