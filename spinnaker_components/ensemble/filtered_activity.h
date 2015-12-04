/**
 * Ensemble - Filtered activity
 * -----------------------------
 * Functions to perform filtering of neuron activity for use in learning rules
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


#ifndef __FILTERED_ACTIVITY_H__
#define __FILTERED_ACTIVITY_H__

// Common includes
#include "common-typedefs.h"

// Ensemble includes
#include "ensemble.h"

//-----------------------------------------------------------------------------
// Structs
//-----------------------------------------------------------------------------
typedef struct activity_filter_parameters_t
{
  // Filter value, e.g., \f$\exp(-\frac{dt}{\tau})\f$
  value_t filter;

  // 1 - filter value
  value_t n_filter;
} activity_filter_parameters_t;

//-----------------------------------------------------------------------------
// External variables
//-----------------------------------------------------------------------------
extern uint32_t g_num_activity_filters;
extern value_t **g_filtered_activities;
activity_filter_parameters_t *g_activity_filter_params;

//----------------------------------
// Inline functions
//----------------------------------
/**
* \brief apply effect of neuron spiking to all filtered activities
*/
static inline void filtered_activity_neuron_spiked(uint32_t n)
{
  // Loop through filters and add n_filter to activites
  for(uint32_t f = 0; f < g_num_activity_filters; f++)
  {
    g_filtered_activities[f][n] += g_activity_filter_params[f].n_filter;
  }
}

//-----------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------
/**
* \brief Copy in data controlling filtered activities
* from the filtered activity region of the Ensemble.
*/
bool filtered_activity_initialise(address_t address, uint32_t n_neurons);

/**
* \brief Apply decay to all filtered activities
*/
void filtered_activity_step();

/** @} */

#endif  // __FILTERED_ACTIVITY_H__