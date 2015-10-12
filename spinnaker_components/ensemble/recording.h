/*!
 * \brief Spike recording
 *
 * \author Andrew Mundy <mundya@cs.man.ac.uk>
 *
 * \copyright Advanced Processor Technologies, School of Computer Science,
 *   University of Manchester
 * \copyright Computational Neuroscience Research Group, Centre for
 *   Theoretical Neuroscience, University of Waterloo
 */

#ifndef __RECORDING_H__
#define __RECORDING_H__

#include <stdbool.h>
#include "spin1_api.h"
#include "common-typedefs.h"
#include "nengo_typedefs.h"
#include "nengo-common.h"
#include <string.h>

typedef struct _recording_buffer_t
{
  uint32_t *buffer;             //!< The buffer to write to
  uint32_t block_length_words;  //!< Size of 1 block of the buffer

  bool record;  //!< Whether or not to record the data in the buffer

  uint32_t *_sdram_start;    //!< Start of the buffer in SDRAM
  uint32_t *_sdram_current;  //!< Current location in the SDRAM buffer
} recording_buffer_t;

/*!\brief Reset the recording region for a new period of simulation.
 */
void record_buffer_reset(recording_buffer_t *buffer);

/*!\brief Flush the current buffer.
 *
 * The contents of the buffer will be appended to the recording region in
 * SDRAM, but only if recording is in use.
 */
static inline void record_buffer_flush(recording_buffer_t *buffer)
{
  // Copy the current buffer into SDRAM
  if (buffer->record)
  {
    spin1_memcpy(buffer->_sdram_current, buffer->buffer,
                 buffer->block_length_words * sizeof(uint32_t));
  }

  // Empty the buffer
  memset(buffer->buffer, 0x0, buffer->block_length_words * sizeof(uint32_t));

  // Progress the pointer
  buffer->_sdram_current += buffer->block_length_words;
}

/*****************************************************************************/
/* Spike specific functions.
 */

/*!\brief Initialise a new recording buffer for recording spikes
 */
bool record_buffer_initialise_spikes(
  recording_buffer_t *buffer,
  address_t region,
  uint n_neurons
);

/*!\brief Record a spike for the given neuron.
 */
static inline void record_spike(recording_buffer_t *buffer, uint32_t n_neuron)
{
  // Get the offset within the current buffer, and the specific bit to set
  // We write to the buffer regardless of whether recording is desired or not
  // in order to reduce branching.
  buffer->buffer[n_neuron >> 5] |= 1 << (n_neuron & 0x1f);
}

/*****************************************************************************/
/* Voltage specific functions.
 *
 * As membrane potential in the normalised LIF implementation is clipped to [0,
 * 1] we can discard the most significant 16 bits of the S16.15 representation
 * to use U1.15 to represent the voltage without any loss.
 */

/*!\brief Initialise a new recording buffer for recording voltages
 */
bool record_buffer_initialise_voltages(
  recording_buffer_t *buffer,
  address_t region,
  uint n_neurons
);

/*!\brief Record a voltage for the given neuron.
 */
static inline void record_voltage(
    recording_buffer_t *buffer, uint32_t n_neuron, value_t voltage
)
{
  // Cast the recording buffer to U1.15 and then store the voltage in this
  // form.
  union {struct {uint16_t lo; uint16_t hi;} bits; value_t value;} value;
  value.value = voltage;

  uint16_t *data = (uint16_t *) buffer->buffer;
  data[n_neuron] = (value.bits).lo;
}

#endif
