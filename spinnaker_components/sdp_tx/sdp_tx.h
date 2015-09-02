/**
 * \addtogroup SDP_TX
 * \brief A component which filters its input and provides output of that
 *        filtered value at regular intervals over SDP.
 *
 * \copyright Advanced Processor Technologies, School of Computer Science,
 *   University of Manchester
 * \copyright Computational Neuroscience Research Group, Centre for
 *   Theoretical Neuroscience, University of Waterloo
 * @{
 */

#ifndef __SDP_TX_H__
#define __SDP_TX_H__

#include "spin1_api.h"
#include "input_filtering.h"

#include "common-impl.h"

/** \brief Shared Tx parameters.
  */
typedef struct sdp_tx_parameters {
  uint machine_timestep;   //!< Machine time step / useconds
  uint transmission_delay; //!< Number of ticks between output transmissions

  uint n_dimensions;       //!< Number of dimensions to represent

  value_t *input;          //!< Input buffer
  uint *keys;              //!< Output keys
} sdp_tx_parameters_t;
extern sdp_tx_parameters_t g_sdp_tx; //!< Global parameters

bool data_system(address_t addr);

#endif

/** @} */
