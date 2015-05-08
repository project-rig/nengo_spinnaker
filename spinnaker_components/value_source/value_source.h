#ifndef __VALUE_SOURCE_H_
#define __VALUE_SOURCE_H_

#include "spin1_api.h"
#include "common-impl.h"
#include "nengo-common.h"
#include "nengo_typedefs.h"

typedef struct _system_parameters_t {
  uint time_step;     //!< Time step of the ValueSource in us
  uint n_dims;        //!< Number of output dimensions (frame length)
  uint flags;         //!< Flags
  uint n_blocks;      //!< Number of FULL blocks
  uint block_length;  //!< Length of a FULL block in frames
  uint partial_block; //!< Length of the last PARTIAL block in frames
} system_parameters_t;

#endif
