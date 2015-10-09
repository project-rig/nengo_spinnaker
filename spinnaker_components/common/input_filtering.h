/* Input Filtering
 * ---------------
 *
 * nengo_spinnaker executables receive multicast packets which represent
 * components of vectors.  These components must be combined to reform the
 * original vectors and then filtered to simulate the effect of synaptic
 * filtering.  As most filters in Nengo are LTI we can perform this filtering
 * in vector space.
 *
 * This module provides methods for:
 *  (1) accumulating input vectors
 *  (2) applying a variety of filters
 *  (3) combining the output of these filters to form the input for an
 *      operation.
 *
 * Consequently the methods are divided into code to deal with these specific
 * tasks.
 *
 * (1) Receiving vectors
 * ---------------------
 *
 * We assume that each filter has ONE input vector and ONE output vector.  When
 * packets are received their keys are used to determine to which input buffer
 * they should be added and to which component of these buffers.  This routing
 * is performed by accessing a list of `if_route_t`s and from this extracting
 * the index of the input vector and component.
 *
 * (2) Applying filters
 * --------------------
 *
 * Each filter has an input vector, an output vector and some filter-specific
 * state.  This is encapsulated in `if_filter_t`.  A filter can be simulated by
 * calling `_if_filter_step`, this will update the output vector and (if
 * necessary) reset the input vector so that it can accumulate new values.  The
 * specific function called to apply the filter is stored internally in the
 * filter.
 *
 * (3) Combining the output of multiple filters
 * --------------------------------------------
 *
 * Multiple filters can be specified and operated on together by creating a
 * `input_filtering_collection` instance.
 *
 *  - `input_filtering_input` can be used to include the value of a packet in
 *     the inputs of appropriate filters.
 *  - `input_filtering_step` can be used to apply all input filters.
 *  - `input_filtering_step_no_accumulate` can be used to apply all input
 *     filters but to not combine their outputs into a single vector.
 *
 *  - `input_filtering_get_routes` will instantiate a filter routing table
 *  - `input_filtering_get_filters` will instantiate the filters
 */

#include "common-typedefs.h"
#include "nengo-common.h"
#include "nengo_typedefs.h"

#ifndef __INPUT_FILTERING_H__
#define __INPUT_FILTERING_H__

typedef void (*FilterStep)(uint32_t, value_t*, value_t*, void*);

/* An input accumulator.
 */
typedef struct _if_accumulator_t
{
  value_t *value;  // Value of the accumulator
  uint32_t mask;   // Mask used to make the accumulator latching or otherwise
} if_accumulator_t;

/* A pair of input and output which are are joined by a filter function. */
typedef struct _if_filter_t
{
  if_accumulator_t *input;  // Input accumulator
  value_t *output;          // Output value
  uint32_t size;            // Size of input and output vectors
  void *state;              // State maintained by the filter

  FilterStep step;  // Filter evaluation function
} if_filter_t;

typedef void (*FilterInit)(void *, if_filter_t*, uint32_t size);

/* Apply new or additional input to a filter. */
static inline void _if_filter_input(if_filter_t *filter,
                                    uint32_t dimension,
                                    value_t value)
{
  // The new accumulator value for this filter is either the current value plus
  // the new value or just the new value depending on the value of the mask.
  filter->input->value[dimension] = \
    kbits(bitsk(filter->input->value[dimension]) & filter->input->mask) + \
    value;
}

/* Simulate one step of a filter and reset its accumulator if necessary */
static inline void _if_filter_step(if_filter_t* filter)
{
  // Disable interrupts to avoid a race condition
  uint32_t cpsr = spin1_fiq_disable();

  // Apply the simulation step
  filter->step(filter->size, filter->input->value,
               filter->output, filter->state);

  // Apply the input accumulator step.  The mask will either set the
  // accumulator to zero or will leave it at its current value.
  for (uint32_t n = 0; n < filter->size; n++)
  {
    filter->input->value[n] = kbits(bitsk(filter->input->value[n]) &
                                    ~filter->input->mask);
  }

  // Re-enable interrupts
  spin1_mode_restore(cpsr);
}

/* A pseudo routing table entry which can be used to determine which input a
 * packet should be included in.
 */
typedef struct _if_route_t
{
  uint32_t key;   // Key against which to compare the received packet
  uint32_t mask;  // Mask against which to compare the received packet

  uint32_t dimension_mask;  // Mask to extract the index of the component

  uint32_t input_index; // Index of the input add the packet to
} if_route_t;

/* A collection of filters which share routing information (and possibly an
 * accumulated output value).
 */
typedef struct _if_collection_t
{
  // Mandatory components
  uint32_t n_filters;  // Number of filters
  uint32_t n_routes;   // Number of routing entries
  if_filter_t *filters;  // Filters
  if_route_t *routes;    // Packet to filter routes

  // Optional components
  uint32_t output_size;  // Size of output vector (may be 0)
  value_t *output;       // Output vector (may be NULL)
} if_collection_t;

/* Include the value of a packet in a filter's input after first subtracting an
 * offset from the packet's index and ensuring that the packet is within a
 * certain range of dimensions.  Returns true if the packet matched any routing
 * entries, otherwise returns false.
 *
 * `dim_offset` is subtracted from the dimension reported by the packet.  If
 * the result is less than or equal to `max_dim_sub_one` then the packet is
 * handled as normal, otherwise it is deemed to have not matched the route.
 */
static inline bool input_filtering_input_with_dimension_offset(
    if_collection_t* filters, uint32_t key, uint32_t payload,
    uint32_t dim_offset, uint32_t max_dim_sub_one
)
{
  bool handled = false;

  // Look at all the routing entries, if we match an entry then include the
  // packet in the indicated input vector.
  for (uint32_t n = 0; n < filters->n_routes; n++)
  {
    // Get the routing entry and the filter referred to by the entry
    if_route_t route = filters->routes[n];
    if_filter_t *filter = &filters->filters[route.input_index];

    if ((key & route.mask) == route.key)
    {
      // Get the dimension of the packet
      // NOTE: if offset is 0 then the subtraction will be optimised out.
      const uint32_t dim = (key & route.dimension_mask) - dim_offset;

      // NOTE: If max_dim_sub_one is UINT32_MAX then the CMP is optimised out
      // as all packets will match.
      if (dim <= max_dim_sub_one)
      {
        // The packet matches this entry and is in the range of dimensions
        // expected; include the contribution from the packet and indicate that
        // we have handled the packet.
        _if_filter_input(filter, dim, kbits(payload));
        handled = true;
      }
    }
  }

  return handled;
}

/* Include the value of a packet in a filter's input.  Returns true if the
 * packet matched any routing entries, otherwise returns false.
 */
static inline bool input_filtering_input(
    if_collection_t* filters, uint32_t key, uint32_t payload
)
{
  // Input with no dimensional offset, the given arguments result in an
  // optimised version of the previous method being inlined.
  return input_filtering_input_with_dimension_offset(
    filters, key, payload, 0, UINT32_MAX
  );
}

/* Apply all filter steps but DO NOT accumulate their outputs. */
static inline void input_filtering_step_no_accumulate(
    if_collection_t *filters)
{
  // Apply the filter step for each filter in the collection.
  for (uint32_t n = filters->n_filters; n > 0; n--)
  {
    // Get the filter and apply the step function
    if_filter_t *filter = &filters->filters[n - 1];
    _if_filter_step(filter);
  }
}

/* Apply all filter steps and accumulate their outputs. */
static inline void input_filtering_step(
    if_collection_t *filters)
{
  // Zero the accumulator, not using memset as this would entail a further
  // function call.
  for (uint32_t d = filters->output_size; d > 0; d--)
  {
    filters->output[d - 1] = 0.0k;
  }

  // Apply all of the filter step functions and accumulate the outputs of the
  // filters.
  for (uint32_t n = filters->n_filters; n > 0; n--)
  {
    // Get the filter and apply the step function
    if_filter_t *filter = &filters->filters[n - 1];
    _if_filter_step(filter);

    // Get the filter output
    value_t *output = filters->filters[n - 1].output;

    // Include each dimension in turn
    for (uint32_t d = filters->output_size; d > 0; d--)
    {
      filters->output[d - 1] += output[d - 1];
    }
  }
}

/* Copy in a set of routing entries.
 *
 * `routes` should be an array of `_if_routes` preceded with a single word
 * indicating the number of entries.
 */
void input_filtering_get_routes(
    if_collection_t *filters,
    uint32_t *routes);

/* Copy in a set of filters.
 *
 * The first word of `data` should indicate how many entries there are.  The
 * first word of each entry should indicate the length of the entry.
 * Each entry should be a `struct _filter_parameters` with some following words
 * which can be interpreted by the appropriate initialisation function.
 */
void input_filtering_get_filters(
    if_collection_t *filters,
    uint32_t *data);

/* Initialise a filter collection with an output accumulator.
 *
 * Use zero to indicate that no output accumulator should be assigned.
 */
void input_filtering_initialise_output(
    if_collection_t *filters,
    uint32_t n_dimensions);

#endif
