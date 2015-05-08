/*!\addtogroup INPUT_FILTER_BANK
 * \brief An input filter bank contains multiple input filters of differing
 *        dimensionality.
 *
 * \author Andrew Mundy <mundya@cs.man.ac.uk>
 * \copyright Advanced Processor Technologies, School of Computer Science, 
 *            University of Manchester
 * @{
 */

#include "input_filter.h"

#ifndef __INPUT_FILTER_BANK_H__
#define __INPUT_FILTER_BANK_H__

/*! \brief Input filter collections.
 */
typedef struct _input_filter_bank_t {
  uint n_inputs;            //!< Number of inputs
  input_filter_t* inputs;   //!< Pointer to array of inputs
} input_filter_bank_t;

/*! \brief Initialise an input filter bank.
 *  \param widths pointer to a region of memory containing (1) the number of
 *         inputs to allocate, (2) the dimensionality of each input in turn.
 */
bool input_filter_bank_initialise(input_filter_bank_t *bank, address_t widths, bool allocate_accumulator);

/*! \brief Initialise filter parameters for a input bank.
 *  \param filter_data pointer to a region of memory containing filter data
 *         split into data for each input with the number of parameters as the
 *         first word in each split.
 */
bool input_filter_bank_get_filters(input_filter_bank_t *bank,
                                   address_t filter_data);

/*! \brief Initialise filter routes for an input bank.
 *  \param routing_data pointer to a region of memory containing routing data
 *         split into data for each input with the number of parameters as the
 *         first word in each split.
 */
bool input_filter_bank_get_routes(input_filter_bank_t *bank,
                                  address_t routing_data);

/*! \brief Perform a step of filtering.
 */
void input_filter_bank_step(input_filter_bank_t *bank);

/*! \brief Callback handler for an incoming dimensional MC packet.
 */
void input_filter_bank_mcpl_rx(input_filter_bank_t *bank, uint key,
                               uint payload);

#endif

/*! @} */
