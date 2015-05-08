/*
 * Ensemble - Output
 * -----------------
 * Structures and functions to deal with arriving multicast packets (input).
 *
 * Authors:
 *   - Andrew Mundy <mundya@cs.man.ac.uk>
 *   - Terry Stewart
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 * 
 */

#include "ensemble_output.h"

uint g_n_output_dimensions, *gp_output_keys;
value_t * gp_output_values;

// Initialise everything necessary for the output system
value_t* initialise_output( region_system_t *pars ){
  io_printf( IO_BUF, "[Ensemble] INITIALISE_OUTPUT.\n" );
  // Store globals, initialise arrays
  g_n_output_dimensions = pars->n_output_dimensions;

  if (g_n_output_dimensions > 0) {
    MALLOC_FAIL_NULL(gp_output_values,
                     pars->n_output_dimensions * sizeof(value_t));
    MALLOC_FAIL_NULL(gp_output_keys,
                     pars->n_output_dimensions * sizeof(uint));

    for (uint n = 0; n < g_n_output_dimensions; n++) {
      gp_output_values[n] = 0;
    }
  }

  io_printf( IO_BUF, "[Ensemble] n_output_dimensions = %d\n",
    g_n_output_dimensions
  );

  // Return the output buffer
  return gp_output_values;
}
