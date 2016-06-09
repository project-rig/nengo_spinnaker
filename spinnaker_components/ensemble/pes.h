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
#ifndef __PES_H_
#define __PES_H_

// Common includes
#include "common-typedefs.h"
#include "input_filtering.h"

// Ensemble includes
#include "ensemble.h"

//----------------------------------
// Inline functions
//----------------------------------
/**
* \brief When using non-filtered activity, applies PES to a spike vector
*/
void pes_apply(const ensemble_state_t *ensemble, const if_collection_t *modulatory_filters);

//----------------------------------
// Functions
//----------------------------------
/**
* \brief Copy in data controlling the PES learning 
* rule from the PES region of the Ensemble.
*/
bool pes_initialise(address_t address);


//void pes_step(const if_collection_t *modulatory_filters);

/** @} */

#endif  // __PES_H__