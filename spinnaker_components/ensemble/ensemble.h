#include <stdint.h>
#include <stdbool.h>

#include "profiler.h"
#include "nengo_typedefs.h"

#ifndef __ENSEMBLE_H__
#define __ENSEMBLE_H__

/*****************************************************************************/
// Region indices
#define ENSEMBLE_REGION               1
#define NEURON_REGION                 2
#define ENCODER_REGION                3
#define BIAS_REGION                   4
#define GAIN_REGION                   5
#define DECODER_REGION                6
#define LEARNT_DECODER_REGION         7
#define KEYS_REGION                   8
#define LEARNT_KEYS_REGION            9
#define POPULATION_LENGTH_REGION      10
#define INPUT_FILTERS_REGION          11
#define INPUT_ROUTING_REGION          12
#define INHIB_FILTERS_REGION          13
#define INHIB_ROUTING_REGION          14
#define MODULATORY_FILTERS_REGION     15
#define MODULATORY_ROUTING_REGION     16
#define LEARNT_ENCODER_FILTERS_REGION 17
#define LEARNT_ENCODER_ROUTING_REGION 18
#define PES_REGION                    19
#define VOJA_REGION                   20
#define FILTERED_ACTIVITY_REGION      21
#define PROFILER_REGION               22
#define REC_SPIKES_REGION             23
#define REC_VOLTAGES_REGION           24
#define REC_ENCODERS_REGION           25
/*****************************************************************************/

/*****************************************************************************/
// Profiler tags
#define PROFILER_INPUT_FILTER     0
#define PROFILER_NEURON_UPDATE    1
#define PROFILER_DECODE           2
#define PROFILER_PES              3
#define PROFILER_VOJA             4
/*****************************************************************************/

/*****************************************************************************/
// Flags
enum
{
  RECORD_SPIKES   = (1 << 0),
  RECORD_VOLTAGES = (1 << 1),
  RECORD_ENCODERS = (1 << 2),
} flags;
/*****************************************************************************/

/*****************************************************************************/
// Ensemble parameters and state structs

// Parameters for the locally represented neurons this is all data stored
// within the system region.
typedef struct _ensemble_parameters
{
  uint32_t machine_timestep;              // Length of timestep (in microseconds)
  uint32_t n_neurons;                     // Number of neurons in this portion
  uint32_t n_dims;                        // Number of dimensions represented
  uint32_t encoder_width;                 // Total width of encoder

  uint32_t n_neurons_total;               // Number of neurons overall
  uint32_t n_populations;                 // Number of populations overall
  uint32_t population_id;                 // Index of this population

  struct
  {
    uint32_t offset;                      // Index of first dimension
    uint32_t n_dims;                      // Number of dimensions
  } input_subspace;                       // Parameters for the input subspace

  uint32_t n_decoder_rows;                // Number of output dimensions
  uint32_t n_learnt_decoder_rows;         // Number of learnt output dimensions
  uint32_t n_profiler_samples;            // Number of profiler samples
  uint32_t n_learnt_input_signals;        // Number of learnt input signals
  uint32_t flags;                         // Flags as per `flags` enum

  // Pointers into SDRAM
  value_t *sdram_input_vector;
  uint32_t *sdram_spike_vector;

  volatile uint8_t *sema_input;           // Input vector synchronisation
  volatile uint8_t *sema_spikes;          // Spike vector synchronisation
} ensemble_parameters_t;

typedef struct _ensemble_state
{
  ensemble_parameters_t parameters;   // Generic parameters

  void *state;                        // Neuron state

  value_t *input;                     // Filtered input vector
  value_t *input_local;               // Start of the section of input we update

  value_t **learnt_input;             // Filtered input vector from each signal
  value_t **learnt_input_local;       // Start of section of learnt_input we update

  value_t inhibitory_input;           // Globally inhibitory input

  value_t *encoders;                  // Encoder matrix
  value_t *bias;                      // Neuron biases
  value_t *gain;                      // Neuron gains

  uint32_t *population_lengths;       // Lengths of all populations
  uint32_t sdram_spikes_length;       // Length of padded spike vector (words)
  uint32_t *spikes;                   // Unpadded spike vector

  value_t *decoders;                  // Rows from the decoder matrix
  uint32_t *keys;                     // Output keys
} ensemble_state_t;
/*****************************************************************************/

/*****************************************************************************/
// DMA Events

// Operation codes for DMA tags
enum
{
  WRITE_FILTERED_VECTOR,        // Write subspace of input into SDRAM
  WRITE_FILTERED_LEARNT_VECTOR, // Write subspace of learnt signal into SDRAM
  READ_WHOLE_VECTOR,            // Read input vector into DTCM
  READ_WHOLE_LEARNED_VECTOR,    // Read learned vector into DTCM
  WRITE_SPIKE_VECTOR,           // Write spike vector into SDRAM
  READ_SPIKE_VECTOR,            // Read spike vector into DTCM for decoding

} dma_tag_ops;
/*****************************************************************************/

#endif  // __ENSEMBLE_H__
