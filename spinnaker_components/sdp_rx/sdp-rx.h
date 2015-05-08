/**
 * \addtogroup SDP_RX
 * \brief A component which receives appropriately formatted SDP packets,
 *        caches the attached data and transmits the values with preloaded
 *        keys at a defined interval.
 *
 * \copyright Advanced Processor Technologies, School of Computer Science,
 *   University of Manchester
 * \copyright Computational Neuroscience Research Group, Centre for
 *   Theoretical Neuroscience, University of Waterloo
 * @{
 */

#ifndef __SDP_RX_H__
#define __SDP_RX_H__

#include "spin1_api.h"
#include "common-impl.h"
#include "nengo-common.h"
#include "nengo_typedefs.h"

/** \brief Shared Rx parameters.
 */
typedef struct sdp_rx_parameters {
  uint transmission_period; //!< Microsecond period between output values

  uint n_dimensions;        //!< Number of dimensions represented
  uint current_dimension;   //!< Index of the currently selected dimension

  value_t *output;          //!< Currently cached output value
  bool *fresh;              //!< Freshness of output
  uint *keys;               //!< Output keys
} sdp_rx_parameters_t;
extern sdp_rx_parameters_t g_sdp_rx; //!< Global parameters

bool data_system(address_t addr);
void data_get_keys(address_t addr);

#endif

/** @} */
