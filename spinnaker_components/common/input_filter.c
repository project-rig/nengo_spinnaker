#include "input_filter.h"


void input_filter_initialise_no_accumulator(input_filter_t* input) 
{
  // Invalidate accumulator and number of dimensions
  input->n_dimensions = 0;
  input->input = NULL;
}

value_t* input_filter_initialise(input_filter_t* input,
                                 uint n_input_dimensions) 
{
  input->n_dimensions = n_input_dimensions;

  MALLOC_FAIL_NULL(input->input,
                  input->n_dimensions * sizeof(value_t));

  // Return the input (to the encoders) buffer
  return input->input;
}


// Filter initialisation
bool input_filter_get_filters(input_filter_t* input, address_t filter_region) {
  input->n_filters = filter_region[0];

  io_printf(IO_BUF, "[Filters] n_filters = %d, n_input_dimensions = %d\n",
            input->n_filters, input->n_dimensions);

  if (input->n_filters > 0) {
    MALLOC_FAIL_FALSE(input->filters,
                      input->n_filters * sizeof(filtered_input_buffer_t*));

    input_filter_data_t* filters = (input_filter_data_t*) (filter_region + 1);

    for (uint f = 0; f < input->n_filters; f++) 
    {
      input->filters[f] = input_buffer_initialise(filters[f].dimensions);
      input->filters[f]->filter = filters[f].filter;
      input->filters[f]->n_filter = filters[f].filter_;
      input->filters[f]->mask = filters[f].mask;
      input->filters[f]->mask_ = ~filters[f].mask;

      io_printf(IO_BUF, "Filter [%u] = %k/%k Masked: 0x%08x/0x%08x Dimensions:%u\n",
                f, filters[f].filter, filters[f].filter_, filters[f].mask,
                ~filters[f].mask, filters[f].dimensions);
    };
  }

  return true;
}


// Filter routers initialisation
bool input_filter_get_filter_routes(input_filter_t* input,
                                    address_t routing_region) {
  input->n_routes = routing_region[0];

  io_printf(IO_BUF, "[Common/Input] %d filter routes.\n", input->n_routes);

  if (input->n_filters > 0 && input->n_routes > 0) {
    MALLOC_FAIL_FALSE(input->routes,
                      input->n_routes * sizeof(input_filter_key_t));
    spin1_memcpy(input->routes, routing_region + 1, 
                 input->n_routes * sizeof(input_filter_key_t));

    for (uint r = 0; r < input->n_routes; r++) {
      io_printf(IO_BUF,
                "Filter route [%d] 0x%08x && 0x%08x => %d with dmask 0x%08x\n",
                r, input->routes[r].key, input->routes[r].mask,
                input->routes[r].filter, input->routes[r].dimension_mask);
    }
  }

  return true;
}


// Input step
void input_filter_step(input_filter_t* input, bool allocate_accumulator) {
  // Zero the input accumulator
  if(allocate_accumulator)
  {
    for (uint d = 0; d < input->n_dimensions; d++)
    {
      input->input[d] = 0x00000000;
    }
  }

  // For each filter
  for (uint f = 0; f < input->n_filters; f++)
  {
    // Apply filtering
    input_buffer_step(input->filters[f]);

    // If required, accumulate the value in 
    // The global input accumulator.
    if(allocate_accumulator)
    {
      for (uint d = 0; d < input->n_dimensions; d++)
      {
        input->input[d] += input->filters[f]->filtered[d];
      }
    }
  }
}

// Incoming spike callback
bool input_filter_mcpl_rx(input_filter_t* input, uint key, uint payload) {
  /*
   * 1. Look up key in input routing table entry
   * 2. Select appropriate filter
   * 3. Add value (payload) to appropriate dimension of given filter.
   */
  // Compare against each key, value pair held in the input
  for (uint i = 0; i < input->n_routes; i++) {
    if ((key & input->routes[i].mask ) == input->routes[i].key) {
      input_buffer_acc(input->filters[input->routes[i].filter],
                       key & input->routes[i].dimension_mask,
                       kbits(payload));
      return true;
    }
  }
  return false;
}
