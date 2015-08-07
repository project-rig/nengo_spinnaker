/**
 * \addtogroup Ensemble
 * \brief An implementation of the Nengo LIF neuron with multidimensional
 *        input capabilities.
 *
 * The Ensemble component implements a LIF neuron model which accepts and
 * transmits multidimensional values.  As in the NEF each neuron in the
 * Ensemble has an *Encoder* which is provided by the Nengo framework running
 * on the host. On each time step the encoders are used to convert the real
 * value presented to the ensemble into currents applied to input of each
 * simulated neuron. Spikes are accumulated and converted into real values
 * using *decoders* (again provided by the host). Decoded values are output
 * in an interleaved fashion during the neuron update loop.
 *
 * \author Andrew Mundy <mundya@cs.man.ac.uk>
 * \author Terry Stewart
 * \author James Knight <knightj@cs.man.ac.uk>
 * 
 * \copyright Advanced Processor Technologies, School of Computer Science,
 *   University of Manchester
 * \copyright Computational Neuroscience Research Group, Centre for
 *   Theoretical Neuroscience, University of Waterloo
 * @{
 */

#ifndef __ENSEMBLE_H__
#define __ENSEMBLE_H__

#include "spin1_api.h"
#include "stdfix-full-iso.h"
#include "common-impl.h"

#include "nengo_typedefs.h"
#include "nengo-common.h"

#include "dimensional-io.h"
#include "recording.h"
#include "input_filter.h"

/** \brief Configuration flags for Ensemble applications. */
enum
{
  RECORD_SPIKES   = (1 << 0),
  RECORD_VOLTAGES = (1 << 1),
} EnsembleFlags;

/* Structs ******************************************************************/
/** \brief Representation of system region. See ::data_system. */
typedef struct region_system 
{
  uint n_input_dimensions;
  uint n_output_dimensions;
  uint n_neurons;
  uint machine_timestep;
  uint t_ref;
  value_t exp_dt_over_t_rc;
  uint32_t flags;
  uint n_inhibitory_dimensions;
} region_system_t;

/** \brief Shared ensemble parameters.
  */
typedef struct ensemble_parameters {
  uint n_neurons;          //!< Number of neurons \f$N\f$
  uint machine_timestep;   //!< Machine time step  / useconds

  uint t_ref;              //!< Refractory period \f$\tau_{ref} - 1\f$ / steps
  value_t exp_dt_over_t_rc;    //!< \f$-\exp{\frac{dt}{\tau_{rc}}}\$

  current_t *i_bias;        //!< Population biases \f$1 \times N\f$
  value_t *neuron_voltage;  //!< Neuron voltages
  uint8_t *neuron_refractory;  //!< Refractory states

  uint n_inhib_dims;        //!< Number of dimensions in inhibitory connection
  value_t *inhib_gain;      //!< Gain of inhibitory connection (value of transform)

  value_t *encoders;        //!< Encoder values \f$N \times D_{in}\f$ (including gains)
  value_t *decoders;        //!< Decoder values \f$N \times\sum D_{outs}\f$

  value_t *input;           //!< Input buffer
  value_t *output;          //!< Output buffer

  recording_buffer_t record_spikes;    //!< Spike recording
  recording_buffer_t record_voltages;  //!< Voltage recording
} ensemble_parameters_t;

/* Parameters and Buffers ***************************************************/
extern ensemble_parameters_t g_ensemble;  //!< Global parameters
extern uint g_output_period;       //!< Delay in transmitting decoded output

extern uint g_n_output_dimensions;

extern input_filter_t g_input;     //!< Input filters and buffers
extern input_filter_t g_input_inhibitory;     //!< Input filters and buffers
extern input_filter_t g_input_modulatory;     //!< Input filters and buffers

/* Functions ****************************************************************/
/**
 * \brief Initialise the ensemble.
 */
bool initialise_ensemble(
  region_system_t *pars  //!< Pointer to formatted system region
);

/**
 * \brief Filter input values, perform neuron update and transmit any output
 *        packets.
 * \param arg0 Unused parameter
 * \param arg1 Unused parameter
 *
 * Neurons are then simulated using Euler's Method as in most implementations
 * of the NEF.  When a neuron spikes it is immediately decoded and its
 * contribution to the output of the Ensemble added to ::output_values.
 */
void ensemble_update( uint arg0, uint arg1 );

/* Static inline access functions ********************************************/
// -- Encoder(s) and decoder(s)
//! Get the encoder value for the given neuron and dimension
static inline value_t neuron_encoder( uint n, uint d )
  { return g_ensemble.encoders[ n * g_input.n_dimensions + d ]; };

static inline value_t neuron_decoder( uint n, uint d )
  { return g_ensemble.decoders[ n * g_n_output_dimensions + d ]; };

static inline value_t *neuron_decoder_vector(uint n)
{
  return &g_ensemble.decoders[n * g_n_output_dimensions];
}

// -- Voltages and refractory periods
//! Get the membrane voltage for the given neuron
static inline voltage_t neuron_voltage( uint n )
  {return g_ensemble.neuron_voltage[n];};

//! Set the membrane voltage for the given neuron
static inline void set_neuron_voltage(uint n, voltage_t v)
  {g_ensemble.neuron_voltage[n] = v;}

//! Get the refractory status of a given neuron
static inline uint8_t neuron_refractory(uint n)
  {return g_ensemble.neuron_refractory[n];};

//! Put the given neuron in a refractory state (zero voltage, set timer)
static inline void set_neuron_refractory( uint n )
  {g_ensemble.neuron_refractory[n] = g_ensemble.t_ref;};

//! Decrement the refractory time for the given neuron
static inline void decrement_neuron_refractory( uint n )
  {g_ensemble.neuron_refractory[n]--;};

#endif

/** @} */
