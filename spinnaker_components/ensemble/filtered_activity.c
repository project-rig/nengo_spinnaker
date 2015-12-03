/*
 * Ensemble - Filtered activity
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

#include "ensemble_filtered_activity.h"

#include <string.h>


//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
uint32_t g_num_activity_filters = 0;
value_t **g_filtered_activities = NULL;
activity_filter_parameters_t *g_activity_filter_params = NULL;

//-----------------------------------------------------------------------------
// Global functions
//-----------------------------------------------------------------------------
bool get_filtered_activity(address_t address)
{
  // Read number of PES learning rules that are configured
  g_num_activity_filters = address[0];
  
  io_printf(IO_BUF, "Filtered activity: Num filters:%u\n", g_num_activity_filters);
  
  if(g_num_activity_filters > 0)
  {
    // Allocate memory
    MALLOC_FAIL_FALSE(g_activity_filter_params,
                      g_num_activity_filters * sizeof(activity_filter_parameters_t));

    MALLOC_FAIL_FALSE(g_filtered_activities,
                      g_num_activity_filters * sizeof(value_t*));

    // Copy propogators from region into new array
    memcpy(g_activity_filter_params, &address[1], g_num_activity_filters * sizeof(activity_filter_parameters_t));
    
    // Loop through filters
    for(uint f = 0; f < g_num_activity_filters; f++)
    {
      io_printf(IO_BUF, "\tFilter %u, Filter:%k, 1.0 - Filter:%f\n",
                f, g_activity_filter_params[f].filter, g_activity_filter_params[f].n_filter);

      // Allocate per-neuron filtered g_filtered_activities
      MALLOC_FAIL_FALSE(g_filtered_activities[f],
                        g_ensemble.n_neurons * sizeof(value_t));

      // Initially zero all filters
      memset(g_filtered_activities[f], 0, g_ensemble.n_neurons * sizeof(value_t));
    }
  }

  return true;
}
//-----------------------------------------------------------------------------
void filtered_activity_step()
{
  // Loop through filters
  for(uint f = 0; f < g_num_activity_filters; f++)
  {
    // Loop through neurons and apply propogators
    for(uint n = 0; n < g_ensemble.n_neurons; n++)
    {
      g_filtered_activities[f][n] *= g_activity_filter_params[f].filter;
    }
  }
}
