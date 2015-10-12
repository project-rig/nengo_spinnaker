// Leaky integrate and fire neurons

#include "ensemble.h"
#include "recording.h"

#ifndef __NEURON_LIF_H__
#define __NEURON_LIF_H__

/*****************************************************************************/
// State variables for an ensemble of LIF neurons
typedef struct lif_parameters
{
  value_t exp_dt_over_tau_rc;   // Function of neuron time constant
  uint32_t tau_ref;             // Refractory period
} lif_parameters_t;

typedef struct lif_states
{
  lif_parameters_t parameters;  // Neuron parameters
  value_t *voltages;            // Neuron voltages
  uint8_t *refractory;          // Refractory counters
} lif_states_t;
/*****************************************************************************/

/*****************************************************************************/
// Prepare neuron state
void lif_prepare_state(
    ensemble_state_t *ensemble, // Generic ensemble state
    uint32_t *address           // SDRAM address of neuron parameters
);
/*****************************************************************************/

/*****************************************************************************/
// Get the refractory counter for a given neuron
static inline uint8_t neuron_refractory(
  const uint32_t neuron,  // Index of the neuron to simulate
  const void *state       // Pointer to neuron state(s)
)
{
  // Cast the state to LIF state type
  lif_states_t *lif_state = (lif_states_t *) state;

  // Return the refractory state for the given neuron
  return lif_state->refractory[neuron];
}
/*****************************************************************************/

/*****************************************************************************/
// Decrement the refractory counter for a given neuron
static inline void neuron_refractory_decrement(
  const uint32_t neuron,  // Index of the neuron to simulate
  const void *state       // Pointer to neuron state(s)
)
{
  // Cast the state to LIF state type
  lif_states_t *lif_state = (lif_states_t *) state;

  // Decrement the refractory state for the given neuron
  lif_state->refractory[neuron]--;
}
/*****************************************************************************/

/*****************************************************************************/
// Perform a single neuron step
static inline bool neuron_step(
  const uint32_t neuron,            // Index of the neuron to simulate
  const value_t input,              // Input to the neuron
  const void *state,                // Pointer to neuron state(s)
  recording_buffer_t *rec_voltages  // Pointer to voltage recording
)
{
  // Cast the state to LIF state type
  lif_states_t *lif_state = (lif_states_t *) state;

  // Compute the change in voltage
  value_t voltage = lif_state->voltages[neuron];
  value_t delta_v = (input - voltage) *
                    lif_state->parameters.exp_dt_over_tau_rc;

  // Update the voltage, but clip it to 0.0
  voltage += delta_v;
  if (bitsk(voltage) < bitsk(0.0k))
  {
    voltage = 0.0k;
  }

  // If the neuron hasn't fired then simply store the voltage and return false
  // to indicate that no spike was produced.
  if (bitsk(voltage) <= bitsk(1.0k))
  {
    lif_state->voltages[neuron] = voltage;
    record_voltage(rec_voltages, neuron, voltage);
    return false;
  }

  // The neuron has spiked, so we prepare to set the voltage and refractory
  // period for the next simulation period.
  uint8_t tau_ref = (uint8_t) lif_state->parameters.tau_ref;
  voltage -= 1.0k;

  // If the overshoot was particularly big further decrease the neuron voltage
  // and refractory period.
  if (bitsk(voltage) > bitsk(2.0k))
  {
    tau_ref--;
    voltage -= delta_v;
  }

  // Store the refractory period and voltage, return true to indicate that a
  // spike occurred.
  lif_state->refractory[neuron] = tau_ref;
  lif_state->voltages[neuron] = voltage;
  return true;
}
/*****************************************************************************/

#endif  // __NEURON_LIF_H__
