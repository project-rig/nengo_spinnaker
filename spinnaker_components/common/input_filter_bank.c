#include "input_filter_bank.h"


bool input_filter_bank_initialise(input_filter_bank_t *bank,
                                  address_t widths, 
                                  bool allocate_accumulator) {
  // Malloc sufficient space for the filters, then initialise each
  bank->n_inputs = widths[0];
  MALLOC_FAIL_FALSE(bank->inputs, bank->n_inputs * sizeof(input_filter_t));

  for (uint d = 0; d < bank->n_inputs; d++) 
  {
    value_t *accumulator = input_filter_initialise(bank->inputs[d], 
        widths[d], allocate_accumulator);

    // If we required one and the accululator
    // Cannot be initialised then return false
    if (allocate_accumulator && accumulator == NULL)
    {
      return false;
    }
  }

  return true;
}


bool input_filter_bank_get_filters(input_filter_bank_t *bank,
                                   address_t filter_data) {
  for (uint i = 0; i < bank->n_inputs; i++) {
    // Initialise each input in turn, then progress the filter
    input_filter_get_filters(bank->inputs[i], filter_data);
    filter_data += 1 + filter_data[0]*sizeof(input_filter_data_t)/4;
  }
}


bool input_filter_bank_get_routes(input_filter_bank_t *bank,
                                  address_t routing_data);
  for (uint i = 0; i < bank->n_inputs; i++) {
    // Initialise each input in turn, then progress the filter
    input_filter_get_routes(bank->inputs[i], routing_data);
    routing_data += 1 + routing_data[0]*sizeof(input_routing_key_t)/4;
  }
}


void input_filter_bank_step(input_filter_bank_t *bank, bool accumulate) {
  for (uint i = 0; i < bank->n_inputs; i++) {
    input_filter_step(bank->inputs + i, accumulate);
  }
}


void input_filter_bank_mcpl_rx(input_filter_bank_t *bank, uint key,
                               uint payload) {
  for (uint i = 0; i < bank->n_inputs; i++) {
    if (input_filter_mcpl_rx(bank->inputs + i, key, bank))
      return;
  }

  io_printf(IO_BUF, __FILE__ " Can't find appropriate input for MCPL: "
            "key=0x%08x, payload=0x%08x\n", key, payload);
}
