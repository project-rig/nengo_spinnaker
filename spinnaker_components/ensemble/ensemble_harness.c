/*
 * Ensemble - Harness
 *
 * Authors:
 *   - Andrew Mundy <mundya@cs.man.ac.uk>
 *   - Terry Stewart
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 */

#include "ensemble.h"
#include "ensemble_output.h"
#include "ensemble_pes.h"

/* Parameters and Buffers ***************************************************/
ensemble_parameters_t g_ensemble;
input_filter_t g_input;
input_filter_t g_input_inhibitory;
input_filter_t g_input_modulatory;

/* Multicast Wrapper ********************************************************/
void mcpl_rx(uint key, uint payload) 
{
  bool handled = false;
  handled |= input_filter_mcpl_rx(&g_input, key, payload);
  handled |= input_filter_mcpl_rx(&g_input_inhibitory, key, payload);
  handled |= input_filter_mcpl_rx(&g_input_modulatory, key, payload);

  if(!handled)
  {
      io_printf(IO_BUF, "[MCPL Rx] Unknown key %08x\n", key);
  }
}

/* Initialisation ***********************************************************/
bool initialise_ensemble(region_system_t *pars) {
  // Save constants
  g_ensemble.n_neurons = pars->n_neurons;
  g_ensemble.machine_timestep = pars->machine_timestep;
  g_ensemble.t_ref = pars->t_ref;
  g_ensemble.exp_dt_over_t_rc = pars->exp_dt_over_t_rc;
  g_ensemble.record_spikes.record = pars->flags & RECORD_SPIKES;
  g_ensemble.record_voltages.record = pars->flags & RECORD_VOLTAGES;
  g_ensemble.n_inhib_dims = pars->n_inhibitory_dimensions;

  io_printf(IO_BUF, "[Ensemble] INITIALISE_ENSEMBLE n_neurons = %d," \
            "timestep = %d, t_ref = %d, exp_dt_over_t_rc = 0x%08x\n",
            g_ensemble.n_neurons,
            g_ensemble.machine_timestep,
            g_ensemble.t_ref,
            g_ensemble.exp_dt_over_t_rc
  );

  // Holder for bias currents
  MALLOC_FAIL_FALSE(g_ensemble.i_bias,
                    g_ensemble.n_neurons * sizeof(current_t));

  // Holder for refractory period and voltages
  MALLOC_FAIL_FALSE(g_ensemble.neuron_voltage,
                    g_ensemble.n_neurons * sizeof(value_t));
  MALLOC_FAIL_FALSE(g_ensemble.neuron_refractory,
                    g_ensemble.n_neurons * sizeof(uint8_t));

  for (uint n = 0; n < g_ensemble.n_neurons; n++) {
    g_ensemble.neuron_refractory[n] = 0;
    g_ensemble.neuron_voltage[n] = 0.0k;
  }

  // Initialise some buffers
  MALLOC_FAIL_FALSE(g_ensemble.encoders,
                    g_ensemble.n_neurons * pars->n_input_dimensions *
                      sizeof(value_t));

  MALLOC_FAIL_FALSE(g_ensemble.decoders,
                    g_ensemble.n_neurons * pars->n_output_dimensions *
                      sizeof(value_t));

  // Setup subcomponents
  g_ensemble.input = input_filter_initialise(&g_input, pars->n_input_dimensions);
  if (g_ensemble.input == NULL)
    return false;

  io_printf(IO_BUF, "@\n");
  if (pars->n_inhibitory_dimensions > 0) {
    if (NULL == input_filter_initialise(
          &g_input_inhibitory, pars->n_inhibitory_dimensions))
      return false;
  }
  io_printf(IO_BUF, "@\n");
  input_filter_initialise_no_accumulator(&g_input_modulatory);
  io_printf(IO_BUF, "@\n");

  g_ensemble.output = initialise_output(pars);
  if (g_ensemble.output == NULL && g_n_output_dimensions > 0)
    return false;

  // Register the update function
  spin1_callback_on(TIMER_TICK, ensemble_update, 2);
  spin1_callback_on(MCPL_PACKET_RECEIVED, mcpl_rx, -1);
  return true;
}
