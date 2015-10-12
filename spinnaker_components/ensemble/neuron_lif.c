#include <string.h>

#include "neuron_lif.h"
#include "nengo-common.h"

/*****************************************************************************/
// Prepare neuron state
void lif_prepare_state(
    ensemble_state_t *ensemble, // Generic ensemble state
    uint32_t *address           // SDRAM address of neuron parameters
)
{
  // Get the number of neurons
  uint32_t n_neurons = ensemble->parameters.n_neurons;

  // Prepare space for neuron parameters
  MALLOC_OR_DIE(ensemble->state, sizeof(lif_states_t));
  lif_states_t *state = ensemble->state;

  // Allocate space for voltages (and zero)
  MALLOC_OR_DIE(state->voltages, sizeof(value_t) * n_neurons);
  memset(state->voltages, 0, sizeof(value_t) * n_neurons);

  // Allocate space for refractory counters
  MALLOC_OR_DIE(state->refractory, sizeof(uint32_t) * n_neurons);
  memset(state->refractory, 0, sizeof(uint32_t) * n_neurons);

  // Copy in LIF parameters
  spin1_memcpy(&state->parameters, address, sizeof(lif_parameters_t));
}
/*****************************************************************************/
