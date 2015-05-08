/**
 * Ensemble - Data
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
 * \addtogroup ensemble
 * @{
 */

#ifndef __ENSEMBLE_DATA_H__
#define __ENSEMBLE_DATA_H__

#include "ensemble.h"

/**
* \brief Copy in data pertaining to the system region of the Ensemble.
*/
bool data_system( address_t addr );

bool data_get_bias(
  address_t addr,
  uint n_neurons
);

bool data_get_encoders(
  address_t addr,
  uint n_neurons,
  uint n_input_dimensions
);

bool data_get_decoders(
  address_t addr,
  uint n_neurons,
  uint n_output_dimensions
);

bool data_get_keys(
  address_t addr,
  uint n_output_dimensions
);

#endif

/** @} */
