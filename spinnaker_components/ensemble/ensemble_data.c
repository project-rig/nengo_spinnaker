/*
 * Ensemble - Data
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
 * 
 */

#include "ensemble_data.h"
#include "ensemble_output.h"

bool data_system(address_t addr) {
  return initialise_ensemble((region_system_t *) addr);
}

bool data_get_bias(
  address_t addr,
  uint n_neurons
){
  spin1_memcpy( g_ensemble.i_bias, addr,
    n_neurons * sizeof( current_t ) );
  return true;
}

bool data_get_encoders(
  address_t addr,
  uint n_neurons,
  uint n_input_dimensions
){
  spin1_memcpy( g_ensemble.encoders, addr,
    n_neurons * n_input_dimensions * sizeof( value_t ) );
  return true;
}

bool data_get_decoders(
  address_t addr,
  uint n_neurons,
  uint n_output_dimensions
){
  spin1_memcpy( g_ensemble.decoders, addr,
    n_neurons * n_output_dimensions * sizeof( value_t ) );

  for( uint n = 0; n < n_neurons; n++ ){
    io_printf( IO_BUF, "Decoder[%d] = ", n );
    for( uint d = 0; d < n_output_dimensions; d++ ){
      io_printf( IO_BUF, "%k, ", neuron_decoder( n, d ) );
    }
    io_printf( IO_BUF, "\n" );
  }

  return true;
}

bool data_get_keys(
  address_t addr,
  uint n_output_dimensions
){
  spin1_memcpy( gp_output_keys, addr,
    n_output_dimensions * sizeof( uint ) );
  return true;
}
