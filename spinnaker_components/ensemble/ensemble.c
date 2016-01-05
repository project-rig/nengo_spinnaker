#include "ensemble.h"

#include "spin1_api.h"
#include "input_filtering.h"
#include "nengo-common.h"
#include "fixed_point.h"
#include "common-impl.h"

#include "string.h"

#include "neuron_lif.h"

/*****************************************************************************/
// Global variables
ensemble_state_t ensemble;  // Global state
if_collection_t input_filters, inhibition_filters;

value_t *sdram_input_vector_local;   // Our portion of the shared input vector
uint32_t *sdram_spikes_vector_local; // Our portion of the shared spike vector

uint32_t unpadded_spike_vector_size;
uint spikes_write_size;

recording_buffer_t record_spikes, record_voltages;  // Recording buffers
/*****************************************************************************/

/*****************************************************************************/
// Simulate neurons and slowly dribble a spike vector out into a given array.
// This function will also apply any encoder learning rules.
void simulate_neurons(
  ensemble_state_t *ensemble,  // State of the ensemble
  uint32_t *spikes             // Spike vector in which to record spikes
)
{
  profiler_write_entry(PROFILER_ENTER | PROFILER_NEURON_UPDATE);

  // Extract parameters
  value_t *input = ensemble->input;
  value_t inhib_input = ensemble->inhibitory_input;
  uint32_t n_dims = ensemble->parameters.n_dims;

  // Bit to use to indicate that a neuron spiked
  uint32_t bit = (1 << 31);

  // Cache for local spike vector
  uint32_t local_spikes = 0x0;

  // Update each neuron in turn
  for (uint32_t n = 0; n < ensemble->parameters.n_neurons; n++)
  {
    // If the neuron is in its refractory period then decrement the refractory
    // counter and progress to the next neuron.
    if (neuron_refractory(n, ensemble->state))
    {
      neuron_refractory_decrement(n, ensemble->state);
    }
    else
    {
      // Compute the neuron input, this is a combination of (a) the bias, (b)
      // the inhibitory input times the gain and (c) the encoded input.
      value_t neuron_input = ensemble->bias[n];
      neuron_input += inhib_input * ensemble->gain[n];
      neuron_input += dot_product(n_dims,
                                  &ensemble->encoders[n_dims * n],
                                  input);

      // Perform the neuron update
      if (neuron_step(n, neuron_input, ensemble->state, &record_voltages))
      {
        // The neuron fired, record the fact in the spike vector that we're
        // constructing.
        local_spikes |= bit;
        record_spike(&record_spikes, n);
      }
    }

    // Rotate the neuron firing bit and progress the spike vector if necessary.
    bit >>= 1;
    if (bit == 0)
    {
      bit = (1 << 31);         // Bit is out of range so reset it
      *spikes = local_spikes;  // Copy spikes into the spike vector
      spikes++;                // Point at the next word in the spike vector
      local_spikes = 0x0;      // Reset the local spike vector
    }
  }

  // Copy any remaining spikes into the specified spike vector
  if (ensemble->parameters.n_neurons % 32)
  {
    *spikes = local_spikes;  // Copy spikes into the spike vector
  }

  // Finish up the recording
  record_buffer_flush(&record_voltages);
  record_buffer_flush(&record_spikes);

  profiler_write_entry(PROFILER_EXIT | PROFILER_NEURON_UPDATE);
}
/*****************************************************************************/

/*****************************************************************************/
// Apply the decoder to a spike vector and transmit multicast packets
// representing the decoded vector.  This function will also apply any decoder
// learning rules.
static inline void decode_output_and_transmit(const ensemble_state_t *ensemble)
{
  profiler_write_entry(PROFILER_ENTER | PROFILER_DECODE);

  // Extract parameters
  uint32_t n_neurons_total = ensemble->parameters.n_neurons_total;
  uint32_t n_populations = ensemble->parameters.n_populations;
  uint32_t *pop_lengths = ensemble->population_lengths;
  uint32_t n_decoder_rows = ensemble->parameters.n_decoder_rows;
  value_t *decoder = ensemble->decoders;
  uint32_t *keys = ensemble->keys;
  uint32_t *spike_vector = ensemble->spikes;

  // TODO Apply decoder learning rules

  // Apply the decoder and transmit multicast packets.
  // Each decoder row is applied in turn to get the output value, which is then
  // transmitted.
  for (uint32_t n = 0; n < n_decoder_rows; n++)
  {
    // Get the row of the decoder
    value_t *row = &decoder[n * n_neurons_total];

    // Compute the decoded value
    value_t output = decode_spike_train(n_populations, pop_lengths,
                                        row, spike_vector);

    // Transmit this value (keep trying until it sends)
    while(!spin1_send_mc_packet(keys[n], bitsk(output), WITH_PAYLOAD))
    {
    }
  }

  profiler_write_entry(PROFILER_EXIT | PROFILER_DECODE);
}
/*****************************************************************************/

/*****************************************************************************/
// Multicast packet with payload received
void mcpl_received(uint key, uint payload)
{
  // Try to receive the packet in each filter

  // General input
  uint32_t offset = ensemble.parameters.input_subspace.offset;
  uint32_t max_dim_sub_one = ensemble.parameters.input_subspace.n_dims - 1;
  input_filtering_input_with_dimension_offset(&input_filters, key, payload,
                                              offset, max_dim_sub_one);

  // Inhibitory
  input_filtering_input(&inhibition_filters, key, payload);
}
/*****************************************************************************/

/*****************************************************************************/
// DMA callback

// The action to take when receiving a DMA callback differs depending on the
// tag.
//
// WRITE_FILTERED_VECTOR: Schedule reading the whole vector back from SDRAM.
// READ_WHOLE_VECTOR:     Simulate the neurons and schedule writing out the
//                        spike vector.
// WRITE_SPIKE_VECTOR:    Wait for synchronisation then schedule reading back
//                        the whole spike vector.
// READ_SPIKE_VECTOR:     Decode the output using the locally stored rows and
//                        transmit multicast packets.
void dma_complete(uint transfer_id, uint tag)
{
  use(transfer_id);  // Unused

  if (tag == WRITE_FILTERED_VECTOR)
  {
    // Wait for all cores to have written their input vectors into SDRAM
    sark_sema_lower((uchar *) ensemble.parameters.sema_input);
    while (*ensemble.parameters.sema_input) ;

    // Schedule reading in the whole input vector from SDRAM
    value_t *sdram_input_vector = ensemble.parameters.sdram_input_vector;
    spin1_dma_transfer(
      READ_WHOLE_VECTOR,                 // Tag
      sdram_input_vector,                // SDRAM address
      ensemble.input,                    // DTCM address
      DMA_READ,                          // Direction
      sizeof(value_t) * ensemble.parameters.n_dims
    );
  }
  else if (tag == READ_WHOLE_VECTOR)
  {
    // Process the neurons, then copy the spike vector into SDRAM.
    simulate_neurons(&ensemble, ensemble.spikes);

    // Schedule writing out the spike vector
    spin1_dma_transfer(
      WRITE_SPIKE_VECTOR,         // Tag
      sdram_spikes_vector_local,  // SDRAM address
      ensemble.spikes,            // DTCM addess
      DMA_WRITE,                  // Direction
      spikes_write_size
    );
  }
  else if (tag == WRITE_SPIKE_VECTOR)
  {
    // Wait for all cores to have written their spike vectors into SDRAM
    sark_sema_lower((uchar *) ensemble.parameters.sema_spikes);
    while (*ensemble.parameters.sema_spikes) ;

    // Schedule reading back the whole spike vector from SDRAM
    uint32_t *sdram_spike_vector = ensemble.parameters.sdram_spike_vector;
    spin1_dma_transfer(
      READ_SPIKE_VECTOR,                         // Tag
      sdram_spike_vector,                        // SDRAM address
      ensemble.spikes,                           // DTCM addess
      DMA_READ,                                  // Direction
      sizeof(uint32_t) * ensemble.sdram_spikes_length  // Size
    );
  }
  else if (tag == READ_SPIKE_VECTOR)
  {
    // Decode and transmit neuron output
    decode_output_and_transmit(&ensemble);
  }
}
/*****************************************************************************/

/*****************************************************************************/
// Timer tick

// On every timer tick the ensemble executable should apply filtering to the
// portion of the subspace that it is responsible and copy its filtered input
// into the shared input in SDRAM.
void timer_tick(uint ticks, uint arg1)
{
  use(arg1);  // Unused

  // Empty the spike raster
  memset(ensemble.spikes, 0, unpadded_spike_vector_size);

  // Stop if we've completed sufficient simulation steps
  if (simulation_ticks != UINT32_MAX && ticks > simulation_ticks)
  {
    profiler_finalise();
    spin1_exit(0);
    return;
  }

  // If there are multiple populations then raise the synchronisation
  // semaphores
  if (ensemble.parameters.n_populations > 1)
  {
    sark_sema_raise((uchar *) ensemble.parameters.sema_input);
    sark_sema_raise((uchar *) ensemble.parameters.sema_spikes);
  }

  // Apply filtering to the input vector
  profiler_write_entry(PROFILER_ENTER | PROFILER_INPUT_FILTER);

  input_filtering_step(&input_filters);
  input_filtering_step(&inhibition_filters);

  profiler_write_entry(PROFILER_EXIT | PROFILER_INPUT_FILTER);

  // If there are multiple populations then schedule copying the input vector
  // into SDRAM.  Otherwise if this is address is NULL start processing the
  // neurons.
  if (ensemble.parameters.n_populations > 1)
  {
    // Compute the size of the transfer
    uint size = sizeof(value_t) * ensemble.parameters.input_subspace.n_dims;

    // Start the DMA transfer
    spin1_dma_transfer(
      WRITE_FILTERED_VECTOR,
      sdram_input_vector_local,  // Section of the SDRAM vector we manage
      ensemble.input_local,      // Section of the DTCM vector we're updating
      DMA_WRITE,
      size
    );
  }
  else
  {
    // Process the neurons, writing the spikes out into DTCM rather than a
    // shared SDRAM vector.
    simulate_neurons(&ensemble, ensemble.spikes);

    // Decode and transmit output
    decode_output_and_transmit(&ensemble);
  }
}
/*****************************************************************************/

/*****************************************************************************/
// Initialisation and setup
void c_main(void)
{
  // Prepare the system for loading
  address_t address = system_load_sram();

  // --------------------------------------------------------------------------
  // Copy in the ensemble parameters
  spin1_memcpy(&ensemble.parameters, region_start(ENSEMBLE_REGION, address),
               sizeof(ensemble_parameters_t));

  // Prepare the input vector
  MALLOC_OR_DIE(ensemble.input, sizeof(value_t) * ensemble.parameters.n_dims);

  // Store an offset into the input vector
  ensemble.input_local =
    &ensemble.input[ensemble.parameters.input_subspace.offset];
  sdram_input_vector_local =&ensemble.parameters.sdram_input_vector[
    ensemble.parameters.input_subspace.offset
  ];

  // Compute the spike size for writing into SDRAM
  spikes_write_size = ensemble.parameters.n_neurons / 32;
  if (ensemble.parameters.n_neurons % 32)
  {
    spikes_write_size++;
  }
  spikes_write_size *= sizeof(uint32_t);

  // Prepare the filters
  input_filtering_get_filters(&input_filters,
                              region_start(INPUT_FILTERS_REGION, address));
  input_filtering_get_routes(&input_filters,
                             region_start(INPUT_ROUTING_REGION, address));
  input_filters.output_size = ensemble.parameters.input_subspace.n_dims;
  input_filters.output = ensemble.input_local;

  input_filtering_get_filters(&inhibition_filters,
                              region_start(INHIB_FILTERS_REGION, address));
  input_filtering_get_routes(&inhibition_filters,
                             region_start(INHIB_ROUTING_REGION, address));
  inhibition_filters.output_size = 1;
  inhibition_filters.output = &ensemble.inhibitory_input;

  // Copy in encoders
  uint encoder_size = sizeof(value_t) * ensemble.parameters.n_neurons *
                      ensemble.parameters.n_dims;
  MALLOC_OR_DIE(ensemble.encoders, encoder_size);
  spin1_memcpy(ensemble.encoders, region_start(ENCODER_REGION, address),
               encoder_size);

  // Copy in bias
  uint bias_size = sizeof(value_t) * ensemble.parameters.n_neurons;
  MALLOC_OR_DIE(ensemble.bias, bias_size);
  spin1_memcpy(ensemble.bias, region_start(BIAS_REGION, address), bias_size);

  // Copy in gain
  uint gain_size = sizeof(value_t) * ensemble.parameters.n_neurons;
  MALLOC_OR_DIE(ensemble.gain, gain_size);
  spin1_memcpy(ensemble.gain, region_start(GAIN_REGION, address), gain_size);

  // Copy in the population lengths
  uint poplength_size = sizeof(uint32_t) * ensemble.parameters.n_populations;
  MALLOC_OR_DIE(ensemble.population_lengths, poplength_size);
  spin1_memcpy(ensemble.population_lengths,
               region_start(POPULATION_LENGTH_REGION, address),
               poplength_size);

  // Prepare the spike vectors
  uint32_t padded_spike_vector_size = 0;
  for (uint p = 0; p < ensemble.parameters.n_populations; p++)
  {
    // If this is the population we represent then store the offset
    if (p == ensemble.parameters.population_id)
    {
      sdram_spikes_vector_local =
        &ensemble.parameters.sdram_spike_vector[padded_spike_vector_size];
    }

    // Include this population
    padded_spike_vector_size += ensemble.population_lengths[p] / 32;
    if (ensemble.population_lengths[p] % 32)
    {
      padded_spike_vector_size++;
    }
  }
  ensemble.sdram_spikes_length = padded_spike_vector_size;

  padded_spike_vector_size *= sizeof(uint32_t);
  MALLOC_OR_DIE(ensemble.spikes, padded_spike_vector_size);

  // Copy in decoders
  uint32_t decoder_size = sizeof(value_t) * ensemble.parameters.n_neurons_total
                          * ensemble.parameters.n_decoder_rows;
  MALLOC_OR_DIE(ensemble.decoders, decoder_size);
  spin1_memcpy(ensemble.decoders, region_start(DECODER_REGION, address),
               decoder_size);

  // Copy in output keys
  uint32_t keys_size = sizeof(uint32_t) * ensemble.parameters.n_decoder_rows;
  MALLOC_OR_DIE(ensemble.keys, keys_size);
  spin1_memcpy(ensemble.keys, region_start(KEYS_REGION, address), keys_size);

  // Prepare the neuron state
  lif_prepare_state(&ensemble, region_start(NEURON_REGION, address));

  // Prepare the profiler
  profiler_read_region(region_start(PROFILER_REGION, address));
  profiler_init(ensemble.parameters.n_profiler_samples);

  // Prepare recording regions
  record_voltages.record = ensemble.parameters.flags & RECORD_VOLTAGES;
  if (!record_buffer_initialise_voltages(
        &record_voltages, region_start(REC_VOLTAGES_REGION, address),
        ensemble.parameters.n_neurons))
  {
    return;
  }

  record_spikes.record = ensemble.parameters.flags & RECORD_SPIKES;
  if (!record_buffer_initialise_spikes(
        &record_spikes, region_start(REC_SPIKES_REGION, address),
        ensemble.parameters.n_neurons))
  {
    return;
  }
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Prepare callbacks
  spin1_set_timer_tick(ensemble.parameters.machine_timestep);
  spin1_callback_on(TIMER_TICK, timer_tick, 2);
  spin1_callback_on(DMA_TRANSFER_DONE, dma_complete, 0);
  spin1_callback_on(MCPL_PACKET_RECEIVED, mcpl_received, -1);
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Run the simulation
  while (true)
  {
    // Wait for data loading, etc.
    event_wait();

    // Determine how long to simulate for
    config_get_n_ticks();

    // Reset the recording regions
    record_buffer_reset(&record_spikes);
    record_buffer_reset(&record_voltages);

    // Perform the simulation
    spin1_start(SYNC_WAIT);
  }
  // --------------------------------------------------------------------------
}
/*****************************************************************************/
