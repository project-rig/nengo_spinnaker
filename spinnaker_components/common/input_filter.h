/*!\addtogroup INPUT_FILTER
 * \brief An input filter receives tuples (key, values) and applies appropriate
 *        filtering (based upon the key), before combining the results in a
 *        unified input vector.
 *
 * \author Andrew Mundy <mundya@cs.man.ac.uk>
 * \copyright Advanced Processor Technologies, School of Computer Science, 
 *            University of Manchester
 * @{
 */

#include "spin1_api.h"
#include "common-typedefs.h"
#include "nengo-common.h"

#include "dimensional-io.h"

#ifndef __INPUT_FILTER_H__
#define __INPUT_FILTER_H__

/*! \brief Keys, masks and filter number.
 */
typedef struct input_filter_key {
  uint key;     //!< MC packet key
  uint mask;    //!< MC packet mask
  uint filter;  //!< ID of filter to use for packets matching this key, mask
  uint dimension_mask;  //!< Mask to retrieve dimension from key
} input_filter_key_t;

typedef struct input_filter_data {
  value_t filter;       //!< Filter value
  value_t filter_;      //!< 1.0 - filter value
  uint32_t mask;        //!< Filter accumulator mask
  uint32_t dimensions;  //!< Dimensions of filter
} input_filter_data_t;

/*! \brief Input filter collection.
 */
typedef struct _input_filter_t {
  uint n_filters;     //!< Number of filters
  uint n_dimensions;  //!< Number of input dimensions
  uint n_routes;      //!< Number of input routing entries

  input_filter_key_t *routes;        //!< List of keys, masks, filter IDs
  filtered_input_buffer_t **filters; //!< Filters to apply to the inputs

  value_t *input;     //!< Resultant input value
} input_filter_t;

/*! \brief Initialise a input filter collection with the given dimensionality.
 *  \returns Pointer to the filtered input vector
 */
value_t* input_filter_initialise(input_filter_t *input,
                                 uint n_input_dimensions);

/*! \brief Initialise a input filter collection without allocating accumulator
 *  \returns Pointer to the filtered input vector
 */
void input_filter_initialise_no_accumulator(input_filter_t *input);

/*! \brief Malloc sufficient space for filters and copy in filter parameters.
 *  Expects the first word pointed to be the number of filters.
 *  \returns Success of the function.
 */
bool input_filter_get_filters(input_filter_t *input, address_t filter_data);

/*! \brief Malloc sufficient space for the filter routes and copy in.
 *  Expects the first word pointed to be the number of filter routes.
 *  \returns Success of the function.
 */
bool input_filter_get_filter_routes(input_filter_t *input,
                                    address_t routing_data);

/*! \brief Perform a step of filtering.
 *  Accumulated filtered values will be placed in the input field of the
 *  `input_filter_t` struct.
 */
void input_filter_step(input_filter_t *input, bool allocate_accumulator);

/*! \brief Callback handler for a incoming dimensional MC packet.
 *  \returns True if the key routed to a filter within this input_filter
 */
bool input_filter_mcpl_rx(input_filter_t *input, uint key, uint payload);

#endif

/*! @} */
