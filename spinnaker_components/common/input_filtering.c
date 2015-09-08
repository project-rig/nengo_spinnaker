#include "nengo-common.h"
#include "input_filtering.h"
#include "common-impl.h"
#include "fixed_point.h"
#include "arm_acle_gcc_selected.h"
#include <string.h>

// Commonly used pair of value_t types
typedef struct _value_t_pair_t
{
  value_t a, b;
} value_t_pair_t;

/* FILTER IMPLEMENTATIONS ****************************************************/
/* None filter : output = input **********************************************/
void _none_filter_step(uint32_t n_dims, value_t *input,
                       value_t *output, void *params)
{
  use(params);

  // The None filter just copies its input to the output
  for (uint32_t d = 0; d < n_dims; d++)
  {
    output[d] = input[d];
  }
}

void _none_filter_init(void *params, if_filter_t *filter, uint32_t size)
{
  use(params);
  use(size);

  debug(">> None filter\n");

  // We just need to set the function pointer for the step function.
  filter->step = _none_filter_step;
}

/* 1st Order Low-Pass ********************************************************/
typedef value_t_pair_t lowpass_state_t;

void _lowpass_filter_step(uint32_t n_dims, value_t *input,
                          value_t *output, void *pars)
{
  // Cast the params
  lowpass_state_t *params = (lowpass_state_t *) pars;
  register int32_t a = bitsk(params->a);
  register int32_t b = bitsk(params->b);

  // Apply the filter to every dimension (realised as a Direct Form I digital
  // filter).
  for (uint32_t d = 0; d < n_dims; d++)
  {
    // The following is equivalent to:
    //
    //    output[d] *= params->a;
    //    output[d] += input[d] * params->b;

    // Compute the next value in a register
    register int64_t next_output;

    // Perform the first multiply
    int32_t current_output = bitsk(output[d]);
    next_output = __smull(current_output, a);

    // Perform the multiply accumulate
    int32_t current_input = bitsk(input[d]);
    next_output = __smlal(next_output, current_input, b);

    // Scale the result back down to store it
    output[d] = kbits(convert_s32_30_s16_15(next_output));
  }
}

void _lowpass_filter_init(void *params, if_filter_t *filter,
                          uint32_t size)
{
  use(size);

  // Copy the filter parameters into memory
  MALLOC_OR_DIE(filter->state, sizeof(lowpass_state_t));
  spin1_memcpy(filter->state, params, sizeof(lowpass_state_t));

  debug(">> Lowpass filter (%k, %k)\n",
            ((lowpass_state_t *)filter->state)->a,
            ((lowpass_state_t *)filter->state)->b);

  // Store a reference to the step function
  filter->step = _lowpass_filter_step;
}

/* General purpose LTI implementation ****************************************/
/* A Direct Form I implementation for linear digital filters.  This
 * implementation cause unnecessary numbers of pipeline flushes, so fixed order
 * filters can be implemented to reduce the cost of branch prediction failures.
 */
typedef lowpass_state_t ab_t;
typedef struct _lti_state_t
{
  uint32_t order;  // Order of the filter
  ab_t *abs;       // Filter coefficients

  // Previous values of the input and output. This is a 2D array, to access the
  // kth last value of dimension d of x use:
  //
  //     xyz[d*order + (n-k) % order].b
  //
  // For y use:
  //
  //     xyz[d*order + (n-k) % order].a
  ab_t *xyz;

  // We treat the previous values as two circular buffers. After a simulation
  // step `n` should be incremented (mod order) to rotate the ring.
  uint32_t n;
} lti_state_t;

void _lti_filter_step(uint32_t n_dims, value_t *input,
                      value_t *output, void *s)
{
  // Cast the state
  lti_state_t *state = (lti_state_t *) s;

  // Apply the filter to every dimension (realised as a Direct Form I digital
  // filter).
  for (uint32_t d = n_dims, dd = n_dims - 1; d > 0; d--, dd--)
  {
    // Point to the correct previous x and y values.
    ab_t *xy = &state->xyz[dd * state->order];

    // Create the new output value for this dimension
    register int64_t output_val = 0;

    // Direct Form I filter
    // `m` acts as an index into the ring buffer of historic input and output.
    for (uint32_t k=0, m = state->n; k < state->order; k++)
    {
      // Update the index into the ring buffer, if this would go negative it
      // wraps to the top of the buffer.
      if (m == 0)
      {
        m += state->order;
      }
      m--;

      // Apply this part of the filter
      // Equivalent to:
      //     output[dd] += ab.a * xyz.a;
      //     output[dd] += ab.b * xyz.b;
      ab_t ab = state->abs[k];
      ab_t xyz = xy[m];
      output_val = __smlal(output_val, bitsk(ab.a), bitsk(xyz.a));
      output_val = __smlal(output_val, bitsk(ab.b), bitsk(xyz.b));
    }

    // Include the initial new input
    xy[state->n].b = input[dd];

    // Save the current output for later steps
    output[dd] = kbits(convert_s32_30_s16_15(output_val));
    xy[state->n].a = output[dd];
  }

  // Rotate the ring buffer by moving the starting pointer, if the starting
  // pointer would go beyond the end of the buffer it is returned to the start.
  if (++state->n == state->order)
  {
    state->n = 0;
  }
}

struct _lti_filter_init_params
{
  uint32_t order;
  value_t data;  // Array of parameters 2*order longer (a[...] || b[...])
};

void _lti_filter_init(void *p, if_filter_t *filter, uint32_t size)
{
  // Cast the parameters block
  struct _lti_filter_init_params *params = \
    (struct _lti_filter_init_params *) p;

  // Malloc space for the parameters
  MALLOC_OR_DIE(filter->state, sizeof(lti_state_t));

  lti_state_t *state = (lti_state_t *) filter->state;
  state->order = params->order;

  debug(">> LTI Filter of order %d", state->order);
  MALLOC_OR_DIE(state->abs, sizeof(ab_t) * state->order);

  // Malloc space for the state
  MALLOC_OR_DIE(state->xyz, sizeof(ab_t) * state->order * size);

  // Copy the parameters across
  value_t *data = &params->data;
  spin1_memcpy(state->abs, data, sizeof(ab_t) * state->order);

  // If debugging then print out all filter parameters
#ifdef DEBUG
  for (uint32_t k = 0; k < state->order; k++)
  {
    io_printf(IO_BUF, "a[%d] = %k, b[%d] = %k\n",
              state->abs[k].a, state->abs[k].b);
  }
#endif

  // Zero all the state holding variables
  state->n = 0;
  memset(state->xyz, 0, sizeof(ab_t) * size * state->order);

  // Store a reference to the correct step function for the filter.  Insert any
  // specially optimised filters here.
  filter->step = _lti_filter_step;
}

/*****************************************************************************/
// Map of filter indices to filter initialisation methods
FilterInit filter_types[] = {
  _none_filter_init,
  _lowpass_filter_init,
  _lti_filter_init,
};

/*Initialisation methods *****************************************************/
/* Copy in a set of routing entries.
 *
 * `routes` should be an array of `_if_routes` preceded with a single word
 * indicating the number of entries.
 */
void input_filtering_get_routes(if_collection_t *filters, uint32_t *routes)
{
  // Copy in the number of routing entries
  filters->n_routes = routes[0];
  routes++;  // Advance the pointer to the first entry
  debug("Loading %d filter routes\n", filters->n_routes);

  // Malloc sufficient room for the entries
  MALLOC_OR_DIE(filters->routes, filters->n_routes * sizeof(if_route_t));

  // Copy the entries across
  spin1_memcpy(filters->routes, routes,
               filters->n_routes * sizeof(if_route_t));

  // DEBUG: Print the route entries
#ifdef DEBUG
  for (uint32_t n = 0; n < filters->n_routes; n++)
  {
    io_printf(IO_BUF, "\tRoute[%d] = (0x%08x, 0x%08x) dmask=0x%08x => %d\n",
              n, filters->routes[n].key, filters->routes[n].mask,
              filters->routes[n].dimension_mask,
              filters->routes[n].input_index);
  }
#endif
}

// Filter specification flags
#define LATCHING 0

// Generic filter parameters
typedef struct _filter_parameters_t
{
  uint32_t n_words;      // # words representing this filter (exc this struct)
  uint32_t init_method;  // Index of the initialisation function to use
  uint32_t size;         // "Width" of the filter (number of dimensions)
  uint32_t flags;        // Flags applied to the filter
  uint32_t data;         // First word of the parameter spec
} filter_parameters_t;

/* Copy in a set of filters.
 *
 * The first word of `filters` should indicate how many entries there are.  The
 * first word of each entry should indicate the length of the entry.  Each
 * entry should be a `filter_parameters_t` with some following words which can
 * be interpreted by the appropriate initialisation function.
 */
void input_filtering_get_filters(if_collection_t *filters, uint32_t *data)
{
  // Get the number of filters and malloc sufficient space for the filter
  // parameters.
  filters->n_filters = data[0];
  MALLOC_OR_DIE(filters->filters, filters->n_filters * sizeof(if_filter_t));

  debug("Loading %d filters\n", filters->n_filters);

  // Move the "filters" pointer to the first element
  data++;

  // Allow casting filters pointer to a _filter_parameters pointer
  filter_parameters_t *params;

  // Initialise each filter in turn
  for (uint32_t f = 0; f < filters->n_filters; f++)
  {
    // Get the parameters
    params = (filter_parameters_t *) data;

    // Get the size of the filter, store it
    filters->filters[f].size = params->size;

    debug("> Filter [%d] size = %d\n", f, params->size);

    // Initialise the input accumulator
    MALLOC_OR_DIE(filters->filters[f].input, sizeof(if_accumulator_t));
    MALLOC_OR_DIE(filters->filters[f].input->value,
                  sizeof(value_t)*params->size);
    filters->filters[f].input->mask = (params->flags & (1 << LATCHING)) ?
                                      0x00000000 : 0xffffffff;

    // Initialise the output vector
    MALLOC_OR_DIE(filters->filters[f].output,
                  sizeof(value_t)*params->size);

    // Zero the input and the output
    memset(filters->filters[f].input->value, 0,
           sizeof(value_t) * params->size);
    memset(filters->filters[f].output, 0, sizeof(value_t) * params->size);

    // Initialise the filter itself
    filter_types[params->init_method]((void *) &params->data,
                                      &filters->filters[f],
                                      params->size);

    // Progress to the next filter, 4 is the number of actual elements in a
    // `filter_parameters_t` (because `->data` is word 0 of the specific filter
    // parameters).
    data += params->n_words + 4;
  }
}

/* Initialise a filter collection with an output accumulator.
 *
 * Use zero to indicate that no output accumulator should be assigned.
 */
void input_filtering_initialise_output(if_collection_t *filters,
                                       uint32_t n_dimensions)
{
  // Store the output size
  filters->output_size = n_dimensions;

  // If the output size is zero then don't allocate an accumulator, otherwise
  // malloc sufficient space.
  if (n_dimensions == 0)
  {
    filters->output = NULL;
  }
  else
  {
    MALLOC_OR_DIE(filters->output, sizeof(value_t) * n_dimensions);
  }
}
