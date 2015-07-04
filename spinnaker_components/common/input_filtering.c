#include "nengo-common.h"
#include "input_filtering.h"
#include "common-impl.h"

/* FILTER IMPLEMENTATIONS ****************************************************/
/* None filter : output = input **********************************************/
void _none_filter_step(uint32_t n_dims, value_t *input,
                       value_t *output, void *params)
{
  use(params);

  // The None filter just copies its input to the output
  for (uint32_t d = 0; d < n_dims; d++)
    output[d] = input[d];
}

void _none_filter_init(void *params, struct _if_filter *filter)
{
  use(params);

  debug(">> None filter\n");

  // We just need to set the function pointer for the step function.
  filter->step = _none_filter_step;
}

/* 1st Order Low-Pass ********************************************************/
struct _lowpass_state
{
  value_t a;  // exp(-dt / tau)
  value_t b;  // 1 - a
};

void _lowpass_filter_step(uint32_t n_dims, value_t *input,
                          value_t *output, void *pars)
{
  // Cast the params
  struct _lowpass_state *params = (struct _lowpass_state *) pars;

  // Apply the filter to every dimension (realised as a Direct Form I digital
  // filter).
  for (uint32_t d = 0; d < n_dims; d++)
  {
    output[d] *= params->a;
    output[d] += input[d] * params->b;
  }
}

void _lowpass_filter_init(void *params, struct _if_filter *filter)
{
  // Copy the filter parameters into memory
  MALLOC_OR_DIE(filter->state, sizeof(struct _lowpass_state));
  spin1_memcpy(filter->state, params, sizeof(struct _lowpass_state));

  debug(">> Lowpass filter (%k, %k)\n",
            ((struct _lowpass_state *)filter->state)->a,
            ((struct _lowpass_state *)filter->state)->b);

  // Store a reference to the step function
  filter->step = _lowpass_filter_step;
}

// Map of filter indices to filter initialisation methods
FilterInit filter_types[] = {
  _none_filter_init,
  _lowpass_filter_init
};

/*Initialisation methods *****************************************************/
/* Copy in a set of routing entries.
 *
 * `routes` should be an array of `_if_routes` preceded with a single word
 * indicating the number of entries.
 */
void input_filtering_get_routes(
    struct input_filtering_collection *filters,
    uint32_t *routes)
{
  // Copy in the number of routing entries
  filters->n_routes = routes[0];
  routes++;  // Advance the pointer to the first entry
  debug("Loading %d filter routes\n", filters->n_routes);

  // Malloc sufficient room for the entries
  MALLOC_OR_DIE(filters->routes, filters->n_routes * sizeof(struct _if_route));

  // Copy the entries across
  spin1_memcpy(filters->routes, routes,
               filters->n_routes * sizeof(struct _if_route));

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
struct _filter_parameters
{
  uint32_t n_words;      // # words representing this filter (exc this struct)
  uint32_t init_method;  // Index of the initialisation function to use
  uint32_t size;         // "Width" of the filter (number of dimensions)
  uint32_t flags;        // Flags applied to the filter
  uint32_t data;         // First word of the parameter spec
};

/* Copy in a set of filters.
 *
 * The first word of `filters` should indicate how many entries there are.  The
 * first word of each entry should indicate the length of the entry.
 * Each entry should be a `struct _filter_parameters` with some following words
 * which can be interpreted by the appropriate initialisation function.
 */
void input_filtering_get_filters(
    struct input_filtering_collection *filters,
    uint32_t *data)
{
  // Get the number of filters and malloc sufficient space for the filter
  // parameters.
  filters->n_filters = data[0];
  MALLOC_OR_DIE(filters->filters, 
                filters->n_filters * sizeof(struct _if_filter));

  debug("Loading %d filters\n", filters->n_filters);

  // Move the "filters" pointer to the first element
  data++;

  // Allow casting filters pointer to a _filter_parameters pointer
  struct _filter_parameters *params;

  // Initialise each filter in turn
  for (uint32_t f = 0; f < filters->n_filters; f++)
  {
    // Get the parameters
    params = (struct _filter_parameters *)data;

    // Get the size of the filter, store it
    filters->filters[f].size = params->size;

    debug("> Filter [%d] size = %d\n", f, params->size);

    // Initialise the input accumulator
    MALLOC_OR_DIE(filters->filters[f].input, sizeof(struct _if_input));
    MALLOC_OR_DIE(filters->filters[f].input->value,
                  sizeof(value_t)*params->size);
    filters->filters[f].input->mask = (params->flags & (1 << LATCHING)) ?
                                      0x00000000 : 0xffffffff;

    // Initialise the output vector
    MALLOC_OR_DIE(filters->filters[f].output,
                  sizeof(value_t)*params->size);

    // Zero the input and the output
    for (uint32_t d = 0; d < params->size; d++)
    {
      filters->filters[f].input->value[d] = 0.0k;
      filters->filters[f].output[d] = 0.0k;
    }

    // Initialise the filter itself
    filter_types[params->init_method]((void *) &params->data,
                                      &filters->filters[f]);

    // Progress to the next filter, 4 is the number of actual elements in a
    // `struct _filter_parameters` (because `->data` is word 0 of the specific
    // filter parameters).
    data += params->n_words + 4;
  }
}

/* Initialise a filter collection with an output accumulator.
 *
 * Use zero to indicate that no output accumulator should be assigned.
 */
void input_filtering_initialise_output(
    struct input_filtering_collection *filters,
    uint32_t n_dimensions)
{
  // Store the output size
  filters->output_size = n_dimensions;

  // If the output size is zero then don't allocate an accumulator, otherwise
  // malloc sufficient space.
  if (n_dimensions == 0)
    filters->output = NULL;
  else
    MALLOC_OR_DIE(filters->output, sizeof(value_t) * n_dimensions);
}
