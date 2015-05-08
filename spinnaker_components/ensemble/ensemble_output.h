/**
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
 * \addtogroup ensemble
 * @{
 */

#ifndef __ENSEMBLE_OUTPUT_H__
#define __ENSEMBLE_OUTPUT_H__

#include "ensemble.h"

/* Buffers and parameters ****************************************************/
extern uint g_n_output_dimensions; //!< Number of output dimensions \f$D_{out}\f$
extern uint * gp_output_keys;      //!< Output dimension keys \f$1 \times D_{out}\f$
extern value_t * gp_output_values; //!< Output buffers \f$1 \times D_{out}\f$

/* Functions *****************************************************************/
/**
 * \brief Initialise the output system.
 * \param pars System region
 */
value_t* initialise_output( region_system_t *pars );

/**
 * \brief Transmit the value associated with one dimension.
 * \param index Dimension to transmit a value for.
 * \param arg1 Unused
 *
 * Timer2 is used to ensure that transmission of outgoing MC packets is
 * regular over time to avoid overloading the network.
 *
 * At each interval the value for one dimension is transmitted, and
 * transmission of the next is scheduled.
 */
void outgoing_dimension_callback( uint index, uint arg1 );

#endif

/** @} */
