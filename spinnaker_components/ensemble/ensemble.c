#include "ensemble.h"

// Standard includes
#include "string.h"

// SpiNNaker includes
#include "spin1_api.h"

// Common includes
#include "input_filtering.h"
#include "nengo-common.h"
#include "fixed_point.h"
#include "common-impl.h"

// Ensemble includes
#include "encoder_recording.h"
#include "filtered_activity.h"
#include "neuron_lif.h"
#include "pes.h"
#include "recording.h"
#include "voja.h"
#include "packet_queue.h"

/*****************************************************************************/
// Global variables
ensemble_state_t ensemble;  // Global state
packet_queue_t packets;  // Queue of multicast packets
bool queue_processing;   // Flag indicating that packets are being processed
unsigned int queue_overflows;

// Input filters and buffers for general and inhibitory inputs. Their outputs
// are summed into accumulators which are used to drive the standard neural input
if_collection_t input_filters;
if_collection_t inhibition_filters;

// Input filters and buffers for modulatory signals. Their
// outputs are left seperate for use by learning rules
if_collection_t modulatory_filters;

// Input filters and buffers for signals to be encoded by learnt encoders.
// Each output is encoded by a seperate encoder so these are also left seperate
if_collection_t learnt_encoder_filters;

if_routing_table_t filter_routing_sliced_inputs;
if_routing_table_t filter_routing_unsliced_inputs;

value_t **sdram_learnt_input_vector_local;  // Our porrtion of the shared learnt input vector
value_t *sdram_input_vector_local;          // Our portion of the shared input vector
uint32_t *sdram_spikes_vector_local;        // Our portion of the shared spike vector

// Index of learnt vector currently being DMAd (applies to both
// WRITE_FILTERED_LEARNT_VECTOR and READ_WHOLE_LEARNED_VECTOR tags)
uint32_t dma_learnt_vector = 0;

// Input vectors corresponding to each learnt input signals - these are copied
// From host, but would be a pain to append to the ensemble_parameters_t struct
value_t **sdram_learnt_input_vector;


uint32_t unpadded_spike_vector_size;
uint spikes_write_size;

recording_buffer_t record_spikes, record_voltages;  // Recording buffers

encoder_recording_buffer_t record_encoders;

/*****************************************************************************/


/*****************************************************************************/
// Writes the accumulated local section of the standard filter's output to SDRAM
static inline void write_filtered_vector()
{
  // Compute the size of the transfer
  uint size = sizeof(value_t) * ensemble.parameters.input_subspace.n_dims;

  spin1_dma_transfer(
    WRITE_FILTERED_VECTOR,
    sdram_input_vector_local,  // Section of the SDRAM vector we manage
    ensemble.input_local,      // Section of the DTCM vector we're updating
    DMA_WRITE,
    size
  );
}
/*****************************************************************************/

/*****************************************************************************/
// Writes the local section of the specified learnt input filter's output to SDRAM
static inline void write_filtered_learnt_vector(uint32_t signal)
{
  // Cache index
  dma_learnt_vector = signal;

  // Compute the size of the transfer
  uint size = sizeof(value_t) * ensemble.parameters.input_subspace.n_dims;

  spin1_dma_transfer(
    WRITE_FILTERED_LEARNT_VECTOR,
    sdram_learnt_input_vector_local[signal],   // Section of the SDRAM vector we manage
    ensemble.learnt_input_local[signal],       // Section of the DTCM vector we're updating
    DMA_WRITE,
    size
  );
}
/*****************************************************************************/

/*****************************************************************************/
// Reads whole input vector from SDRAM
static inline void read_whole_vector()
{
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
/*****************************************************************************/

/*****************************************************************************/
// Reads whole input vector from SDRAM
static inline void read_whole_learned_vector(uint32_t signal)
{
  // Cache index
  dma_learnt_vector = signal;

  spin1_dma_transfer(
    READ_WHOLE_LEARNED_VECTOR,          // Tag
    sdram_learnt_input_vector[signal],  // SDRAM address
    ensemble.learnt_input[signal],      // DTCM address
    DMA_READ,                           // Direction
    sizeof(value_t) * ensemble.parameters.n_dims
  );
}
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
  uint32_t encoder_width = ensemble->parameters.encoder_width;

  // Bit to use to indicate that a neuron spiked
  uint32_t bit = (1 << 31);

  // Cache for local spike vector
  uint32_t local_spikes = 0x0;

  // Update each neuron in turn
  for (uint32_t n = 0; n < ensemble->parameters.n_neurons; n++)
  {
    // Get this neuron's encoder vector
    value_t *encoder_vector = &ensemble->encoders[encoder_width * n];

    // Is this neuron in its refractory period
    bool in_refractory_period = neuron_refractory(n, ensemble->state);

    // Loop through learnt input signals and encoder slices
    uint32_t f = 0;
    uint32_t e = n_dims;
    value_t neuron_input = 0.0k;
    for(; f < ensemble->parameters.n_learnt_input_signals; f++, e += n_dims)
    {
      // Get encoder vector for this neuron offset for correct learnt encoder
      const value_t *learnt_encoder_vector = encoder_vector + e;

      // Record learnt encoders
      // **NOTE** idea here is that by interspersing these between encoding
      // operations, write buffer should have time to be written out
      record_learnt_encoders(&record_encoders,
        n_dims, learnt_encoder_vector);

      // If neuron's not in refractory period,
      // apply input encoded by learnt encoders
      if(!in_refractory_period)
      {
        neuron_input += dot_product(n_dims, learnt_encoder_vector,
                                    ensemble->learnt_input[f]);
      }
    }

    // If the neuron's in its refractory period, decrement the refractory counter
    if (in_refractory_period)
    {
      neuron_refractory_decrement(n, ensemble->state);
    }
    else
    {
      // Compute the neuron input, this is a combination of (a) the bias, (b)
      // the inhibitory input times the gain and (c) the non-learnt encoded input.
      neuron_input += ensemble->bias[n];
      neuron_input += inhib_input * ensemble->gain[n];

      // If there are any static input filters
      // **YUCK** this potentially massive optimisation
      // could also extend to memory by not allocating the encoders
      if(input_filters.n_filters > 0)
      {
        neuron_input += dot_product(n_dims, encoder_vector, input);
      }

      // Perform the neuron update
      if (neuron_step(n, neuron_input, ensemble->state, &record_voltages))
      {
        // The neuron fired, record the fact in the spike vector that we're
        // constructing.
        local_spikes |= bit;
        record_spike(&record_spikes, n);

        // Apply effect of neuron spiking to filtered activities
        //filtered_activity_neuron_spiked(n);

        // Update non-filtered Voja learning
        voja_neuron_spiked(encoder_vector, ensemble->gain[n], n_dims,
                           &modulatory_filters, ensemble->learnt_input);
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
// Decode a spike train to produce a single value
static value_t decode_spike_train(
  const uint32_t n_populations,        // Number of populations
  const uint32_t *population_lengths,  // Length of the populations
  const value_t *decoder,              // Decoder to use
  const uint32_t *spikes               // Spike vector
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

/*****************************************************************************/
// Apply the decoder to a spike vector and transmit multicast packets
// representing the decoded vector.  This function will also apply any decoder
// learning rules.
static inline void decode_output_and_transmit(const ensemble_state_t *ensemble)
{
  profiler_write_entry(PROFILER_ENTER | PROFILER_DECODE);

  // Extract parameters
  const ensemble_parameters_t *params = &ensemble->parameters;
  uint32_t n_neurons_total = params->n_neurons_total;
  uint32_t n_populations = params->n_populations;
  uint32_t n_decoder_rows = params->n_decoder_rows + params->n_learnt_decoder_rows;

  uint32_t *pop_lengths = ensemble->population_lengths;
  value_t *decoder = ensemble->decoders;
  uint32_t *keys = ensemble->keys;
  uint32_t *spike_vector = ensemble->spikes;

  // Apply the decoder and transmit multicast packets.
  // Each decoder row is applied in turn to get the output value, which is then
  // transmitted.
  for (uint32_t d = 0; d < n_decoder_rows; d++)
  {
    // Get the row of the decoder
    value_t *row = &decoder[d * n_neurons_total];

    // Compute the decoded value
    value_t output = decode_spike_train(n_populations, pop_lengths,
                                        row, spike_vector);

    // Transmit this value (keep trying until it sends)
    while(!spin1_send_mc_packet(keys[d], bitsk(output), WITH_PAYLOAD))
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
  // Queue the packet for later processing, if no processing is scheduled then
  // trigger the queue processor.
  if (packet_queue_push(&packets, key, payload))
  {
    if (!queue_processing)
    {
      spin1_trigger_user_event(0, 0);
      queue_processing = true;
    }
  }
  else
  {
    // The packet couldn't be included in the queue, thus it was essentially
    // dropped.
    queue_overflows++;
  }
}

void process_queue()
{
  uint32_t offset = ensemble.parameters.input_subspace.offset;
  uint32_t max_dim_sub_one = ensemble.parameters.input_subspace.n_dims - 1;

  // Continuously remove packets from the queue and include them in filters
  while (packet_queue_not_empty(&packets))
  {
    // Pop a packet from the queue (critical section)
    packet_t packet;
    uint cpsr = spin1_fiq_disable();
    bool packet_is_valid = packet_queue_pop(&packets, &packet);
    spin1_mode_restore(cpsr);

    // Process the received packet
    if (packet_is_valid)
    {
      uint32_t key = packet.key;
      uint32_t payload = packet.payload;

      input_filtering_input_with_dimension_offset(
        &filter_routing_sliced_inputs, key, payload,
        offset, max_dim_sub_one
      );
      input_filtering_input(&filter_routing_unsliced_inputs, key, payload);
    }
    else
    {
      io_printf(IO_BUF, "Popped packet from empty queue.\n");
      rt_error(RTE_ABORT);
    }
  }
  queue_processing = false;
}

void user_event(uint arg0, uint arg1)
{
  use(arg0);
  use(arg1);

  process_queue();
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

  if(tag == WRITE_FILTERED_LEARNT_VECTOR)
  {
    // Get index of next learnt vector
    uint32_t next_learnt_vector = dma_learnt_vector + 1;

    // If there are more to go, write next learnt vector to sdram
    if(next_learnt_vector < ensemble.parameters.n_learnt_input_signals)
    {
      write_filtered_learnt_vector(next_learnt_vector);
    }
    // Otherwise, write filtered vector
    else
    {
      write_filtered_vector();
    }
  }
  else if (tag == WRITE_FILTERED_VECTOR)
  {
    // Wait for all cores to have written their input vectors into SDRAM
    sark_sema_lower((uchar *) ensemble.parameters.sema_input);
    while (*ensemble.parameters.sema_input) ;

    // If there are any learnt input signals to transfer, start reading of 1st signal
    if(ensemble.parameters.n_learnt_input_signals > 0)
    {
      read_whole_learned_vector(0);
    }
    else
    {
      read_whole_vector();
    }
  }
  else if(tag == READ_WHOLE_LEARNED_VECTOR)
  {
    // Get index of next learnt vector
    uint32_t next_learnt_vector = dma_learnt_vector + 1;

    // If there are more to go, read next learnt vector from sdram
    if(next_learnt_vector < ensemble.parameters.n_learnt_input_signals)
    {
      read_whole_learned_vector(next_learnt_vector);
    }
    // Otherwise, read whole vector
    else
    {
      read_whole_vector();
    }
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
    // Apply PES learning to spike vector
    pes_apply(&ensemble, &modulatory_filters);

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

  process_queue();

  input_filtering_step(&input_filters);
  input_filtering_step(&inhibition_filters);
  input_filtering_step_no_accumulate(&modulatory_filters);
  input_filtering_step_no_accumulate(&learnt_encoder_filters);

  profiler_write_entry(PROFILER_EXIT | PROFILER_INPUT_FILTER);

  // If there are multiple populations then schedule copying the input vector
  // into SDRAM.  Otherwise if this is address is NULL start processing the
  // neurons.
  if (ensemble.parameters.n_populations > 1)
  {
    // If there are any learnt input signals to transfer, start transfer of 1st signal
    if(ensemble.parameters.n_learnt_input_signals > 0)
    {
      write_filtered_learnt_vector(0);
    }
    // Otherwise, transfer our section of filtered vector
    else
    {
      write_filtered_vector();
    }
  }
  else
  {
    // Process the neurons, writing the spikes out into DTCM rather than a
    // shared SDRAM vector.
    simulate_neurons(&ensemble, ensemble.spikes);

    // Apply PES learning to spike vector
    pes_apply(&ensemble, &modulatory_filters);
    
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
  ensemble_parameters_t *params = &ensemble.parameters;
  spin1_memcpy(params, region_start(ENSEMBLE_REGION, address),
               sizeof(ensemble_parameters_t));


  // Allocate array to hold pointers to SDRAM learnt input vectors and copy in pointers
  MALLOC_OR_DIE(sdram_learnt_input_vector,
                sizeof(value_t*) * params->n_learnt_input_signals);
  spin1_memcpy(sdram_learnt_input_vector,
               ((uint8_t*)region_start(ENSEMBLE_REGION, address)) + sizeof(ensemble_parameters_t),
                sizeof(value_t*) * params->n_learnt_input_signals);

  // Prepare the input vector
  MALLOC_OR_DIE(ensemble.input, sizeof(value_t) * params->n_dims);

  // Store an offset into the input vector
  ensemble.input_local =
    &ensemble.input[params->input_subspace.offset];
  sdram_input_vector_local = &params->sdram_input_vector[
    params->input_subspace.offset
  ];

  // Allocate an array of local, global and shared pointers for each learnt input signals
  MALLOC_OR_DIE(ensemble.learnt_input, sizeof(value_t*) * params->n_learnt_input_signals);
  MALLOC_OR_DIE(ensemble.learnt_input_local, sizeof(value_t*) * params->n_learnt_input_signals);
  MALLOC_OR_DIE(sdram_learnt_input_vector_local, sizeof(value_t*) * params->n_learnt_input_signals);

  // Loop through learnt input signals
  for(uint32_t i = 0; i < params->n_learnt_input_signals; i++)
  {
    // Allocate vector
    MALLOC_OR_DIE(ensemble.learnt_input[i], sizeof(value_t) * params->n_dims);

    // Store local offset
    ensemble.learnt_input_local[i] = &ensemble.learnt_input[i][
      params->input_subspace.offset];

    // Store SDRAM offset
    sdram_learnt_input_vector_local[i] = &sdram_learnt_input_vector[i][
      params->input_subspace.offset];

    io_printf(IO_BUF, "Learnt input signal %u: learnt_input:%08x, learnt_input_local:%08x, sdram_learnt_input:%08x, sdram_learnt_input_local:%08x\n",
      i, ensemble.learnt_input[i], ensemble.learnt_input_local[i], sdram_learnt_input_vector[i], sdram_learnt_input_vector_local[i]);
  }

  // Compute the spike size for writing into SDRAM
  spikes_write_size = params->n_neurons / 32;
  if (params->n_neurons % 32)
  {
    spikes_write_size++;
  }
  spikes_write_size *= sizeof(uint32_t);

  // Prepare the filters
  input_filtering_get_filters(&input_filters,
                              region_start(INPUT_FILTERS_REGION, address),
                              NULL);
  input_filters.output_size = params->input_subspace.n_dims;
  input_filters.output = ensemble.input_local;

  input_filtering_get_filters(&inhibition_filters,
                              region_start(INHIB_FILTERS_REGION, address),
                              NULL);
  inhibition_filters.output_size = 1;
  inhibition_filters.output = &ensemble.inhibitory_input;

  input_filtering_get_filters(&modulatory_filters,
                              region_start(MODULATORY_FILTERS_REGION, address),
                              NULL);

  input_filtering_get_filters(&learnt_encoder_filters,
                              region_start(LEARNT_ENCODER_FILTERS_REGION, address),
                              ensemble.learnt_input_local);

  filter_arg_t sliced_inputs[] = {
    {.filters = &input_filters,
     .routes = (filter_routes_t *) region_start(INPUT_ROUTING_REGION, address)
    },
    {.filters = &learnt_encoder_filters,
     .routes = (filter_routes_t *) region_start(LEARNT_ENCODER_ROUTING_REGION, address)
    },
  };
  input_filter_build_combined_routes(
    &filter_routing_sliced_inputs,
    sizeof(sliced_inputs) / sizeof(filter_arg_t),
    sliced_inputs
  );

  filter_arg_t unsliced_inputs[] = {
    {.filters = &inhibition_filters,
     .routes = (filter_routes_t *) region_start(INHIB_ROUTING_REGION, address)
    },
    {.filters = &modulatory_filters,
     .routes = (filter_routes_t *) region_start(MODULATORY_ROUTING_REGION, address)
    },
  };
  input_filter_build_combined_routes(
    &filter_routing_unsliced_inputs,
    sizeof(unsliced_inputs) / sizeof(filter_arg_t),
    unsliced_inputs
  );
  // Copy in encoders
  uint encoder_size = sizeof(value_t) * params->n_neurons *
                      params->encoder_width;
  MALLOC_OR_DIE(ensemble.encoders, encoder_size);
  spin1_memcpy(ensemble.encoders, region_start(ENCODER_REGION, address),
               encoder_size);

  // Copy in bias
  uint bias_size = sizeof(value_t) * params->n_neurons;
  MALLOC_OR_DIE(ensemble.bias, bias_size);
  spin1_memcpy(ensemble.bias, region_start(BIAS_REGION, address), bias_size);

  // Copy in gain
  uint gain_size = sizeof(value_t) * params->n_neurons;
  MALLOC_OR_DIE(ensemble.gain, gain_size);
  spin1_memcpy(ensemble.gain, region_start(GAIN_REGION, address), gain_size);

  // Copy in the population lengths
  uint poplength_size = sizeof(uint32_t) * params->n_populations;
  MALLOC_OR_DIE(ensemble.population_lengths, poplength_size);
  spin1_memcpy(ensemble.population_lengths,
               region_start(POPULATION_LENGTH_REGION, address),
               poplength_size);

  // Prepare the spike vectors
  uint32_t padded_spike_vector_size = 0;
  for (uint p = 0; p < params->n_populations; p++)
  {
    // If this is the population we represent then store the offset
    if (p == params->population_id)
    {
      sdram_spikes_vector_local =
        &params->sdram_spike_vector[padded_spike_vector_size];
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

  // Allocate array large enough for static and learnt decoders
  const uint32_t decoder_words = params->n_neurons_total * params->n_decoder_rows;
  const uint32_t learnt_decoder_words = params->n_neurons_total * params->n_learnt_decoder_rows;
  MALLOC_OR_DIE(ensemble.decoders,
                (decoder_words + learnt_decoder_words) * sizeof(value_t));

  // Copy static decoders into beginning of this array
  spin1_memcpy(ensemble.decoders, region_start(DECODER_REGION, address),
               decoder_words * sizeof(value_t));

  // Follow this by learnt decoders
  spin1_memcpy(ensemble.decoders + decoder_words,
               region_start(LEARNT_DECODER_REGION, address),
               learnt_decoder_words * sizeof(value_t));

  // Allocate array large enough for static and learnt keys
  MALLOC_OR_DIE(ensemble.keys,
                sizeof(uint32_t) * (params->n_decoder_rows + params->n_learnt_decoder_rows));

  // Copy static keys into beginning of this array
  spin1_memcpy(ensemble.keys,
               region_start(KEYS_REGION, address),
               params->n_decoder_rows * sizeof(uint32_t));

  // Follow this by learnt keys
  spin1_memcpy(ensemble.keys + params->n_decoder_rows,
               region_start(LEARNT_KEYS_REGION, address),
               params->n_learnt_decoder_rows * sizeof(uint32_t));

  // Prepare the neuron state
  lif_prepare_state(&ensemble, region_start(NEURON_REGION, address));

  // Initialise learning rule regions
  if(!pes_initialise(region_start(PES_REGION, address)))
  {
    return;
  }

  if(!voja_initialise(region_start(VOJA_REGION, address)))
  {
    return;
  }

  // Initialise filtered activity region
  /*if(!filtered_activity_initialise(
    region_start(FILTERED_ACTIVITY_REGION, address)))
  {
    return;
  }*/

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

  record_encoders.record = ensemble.parameters.flags & RECORD_ENCODERS;
  if (!record_learnt_encoders_initialise(&record_encoders,
        region_start(REC_ENCODERS_REGION, address)))
  {
    return;
  }

  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Multicast packet queue
  queue_processing = false;
  packet_queue_init(&packets, 1024);
  queue_overflows = 0;
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Prepare callbacks
  spin1_set_timer_tick(ensemble.parameters.machine_timestep);
  spin1_callback_on(TIMER_TICK, timer_tick, 1);
  spin1_callback_on(DMA_TRANSFER_DONE, dma_complete, 0);
  spin1_callback_on(MCPL_PACKET_RECEIVED, mcpl_received, -1);
  spin1_callback_on(USER_EVENT, user_event, 1);
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

    // Check on the status of the packet queue
    if (queue_overflows)
    {
      io_printf(IO_BUF, "Queue overflows = %u\n", queue_overflows);
      rt_error(RTE_ABORT);
    }

    // Perform the simulation
    spin1_start(SYNC_WAIT);
  }
  // --------------------------------------------------------------------------
}
/*****************************************************************************/
