#include <stdint.h>
#include <stdbool.h>

#include "profiler.h"
#include "nengo_typedefs.h"

#ifndef __ENSEMBLE_H__
#define __ENSEMBLE_H__

/*****************************************************************************/
// Region indices
#define ENSEMBLE_REGION           1
#define NEURON_REGION             2
#define ENCODER_REGION            3
#define BIAS_REGION               4
#define GAIN_REGION               5
#define DECODER_REGION            6
#define KEYS_REGION               7
#define POPULATION_LENGTH_REGION  8
#define INPUT_FILTERS_REGION      9
#define INPUT_ROUTING_REGION     10
#define INHIB_FILTERS_REGION     11
#define INHIB_ROUTING_REGION     12
#define PROFILER_REGION          13
#define REC_SPIKES_REGION        14
#define REC_VOLTAGES_REGION      15
/*****************************************************************************/

/*****************************************************************************/
// Profiler tags
#define PROFILER_INPUT_FILTER     0
#define PROFILER_NEURON_UPDATE    1
#define PROFILER_DECODE           2
/*****************************************************************************/

/*****************************************************************************/
// Flags
enum
{
  RECORD_SPIKES =   1 << 0,
  RECORD_VOLTAGES = 1 << 1,
} flags;
/*****************************************************************************/

/*****************************************************************************/
// Ensemble parameters and state structs

// Parameters for the locally represented neurons this is all data stored
// within the system region.
typedef struct _ensemble_parameters
{
  uint32_t machine_timestep;  // Length of timestep (in microseconds)
  uint32_t n_neurons;         // Number of neurons in this portion
  uint32_t n_dims;            // Number of dimensions represented

  uint32_t n_neurons_total;   // Number of neurons overall
  uint32_t n_populations;     // Number of populations overall
  uint32_t population_id;     // Index of this population

  struct
  {
    uint32_t offset;          // Index of first dimension
    uint32_t n_dims;          // Number of dimensions
  } input_subspace;           // Parameters for the input subspace

  uint32_t n_decoder_rows;    // Number of output dimensions

  uint32_t n_profiler_samples;    // Number of profiler samples
  uint32_t flags;                 // Flags as per `flags` enum

  // Pointers into SDRAM
  value_t *sdram_input_vector;
  uint32_t *sdram_spike_vector;

  volatile uint8_t *sema_input;   // Input vector synchronisation
  volatile uint8_t *sema_spikes;  // Spike vector synchronisation
} ensemble_parameters_t;

typedef struct _ensemble_state
{
  ensemble_parameters_t parameters;   // Generic parameters

  void *state;                        // Neuron state

  value_t *input;                     // Filtered input vector
  value_t *input_local;               // Start of the section we update
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
  WRITE_FILTERED_VECTOR = 0,  // Write subspace of input into SDRAM
  READ_WHOLE_VECTOR     = 1,  // Read input vector into DTCM
  WRITE_SPIKE_VECTOR    = 2,  // Write spike vector into SDRAM
  READ_SPIKE_VECTOR     = 3,  // Read spike vector into DTCM for decoding
} dma_tag_ops;
/*****************************************************************************/

/*****************************************************************************/
// Decode a spike train to produce a single value
static value_t decode_spike_train(
  const uint32_t n_populations,        // Number of populations
  const uint32_t *population_lengths,  // Length of the populations
  const value_t *decoder,              // Decoder to use
  uint32_t *spikes                     // Spike vector
)
{
  // Resultant decoded value
  value_t output = 0.0k;

  // For each population
  for (uint32_t p = 0; p < n_populations; p++)
  {
    // Get the number of neurons in this population
    uint32_t pop_length = population_lengths[p];

    // While we have neurons left to process
    while (pop_length)
    {
      // Determine how many neurons are in the next word of the spike vector.
      uint32_t n = (pop_length > 32) ? 32 : pop_length;

      // Load the next word of the spike vector
      uint32_t data = *(spikes++);

      // Include the contribution from each neuron
      while (n)  // While there are still neurons left
      {
        // Work out how many neurons we can skip
        // XXX: The GCC documentation claims that `__builtin_clz(0)` is
        // undefined, but the ARM instruction it uses is defined such that:
        // CLZ 0x00000000 is 32
        uint32_t skip = __builtin_clz(data);

        // If `skip` is NOT less than `n` then there are either no firing
        // neurons left in the word (`skip` == 32) or the first `1` in the word
        // is beyond the range of bits we care about anyway.
        if (skip < n)
        {
          // Skip until we reach the next neuron which fired
          decoder += skip;

          // Decode the given neuron
          output += *decoder;

          // Prepare to test the neuron after the one we just processed.
          decoder++;
          skip++;              // Also skip the neuron we just decoded
          pop_length -= skip;  // Reduce the number of neurons left
          n -= skip;           // and the number left in this word.
          data <<= skip;       // Shift out processed neurons
        }
        else
        {
          // There are no neurons left in this word
          decoder += n;     // Point at the decoder for the next neuron
          pop_length -= n;  // Reduce the number left in the population
          n = 0;            // No more neurons left to process
        }
      }
    }
  }

  // Return the decoded value
  return output;
}
/*****************************************************************************/

#endif  // __ENSEMBLE_H__
