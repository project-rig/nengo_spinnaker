/**
 * Ensemble - Spike vector processor
 * ---------------------------------
 * Functions to perform PES decoder learning
 *
 * Authors:
 *   - James Knight <knightj@cs.man.ac.uk>
 *   - Andrew Mundy <mundya@cs.man.ac.uk>
 *
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *
 * \addtogroup ensemble
 * @{
 */


#ifndef __SPIKE_VECTOR_PROCESSOR_H__
#define __SPIKE_VECTOR_PROCESSOR_H__

#define PROCESS_SPIKE_VECTOR(N_POPULATIONS, POPULATION_LENGTHS, SPIKES, ADVANCE, APPLY) \
({\
  /* For each population */ \
  for (uint32_t p = 0; p < N_POPULATIONS; p++)  \
  { \
    /* Get the number of neurons in this population */\
    uint32_t pop_length = POPULATION_LENGTHS[p];  \
\
    /* While we have neurons left to process */ \
    while (pop_length)\
    {\
      /* Determine how many neurons are in the next word of the spike vector. */\
      uint32_t n = (pop_length > 32) ? 32 : pop_length;\
\
      /* Load the next word of the spike vector */\
      uint32_t data = *(SPIKES++);\
\
      /* Include the contribution from each neuron */\
      while (n)  /*While there are still neurons left */\
      {\
        /* Work out how many neurons we can skip\
        XXX: The GCC documentation claims that `__builtin_clz(0)` is\
        undefined, but the ARM instruction it uses is defined such that:\
        CLZ 0x00000000 is 32 */\
        uint32_t skip = __builtin_clz(data);\
\
        /* If `skip` is NOT less than `n` then there are either no firing\
        neurons left in the word (`skip` == 32) or the first `1` in the word\
        is beyond the range of bits we care about anyway. */\
        if (skip < n)\
        {\
          /* Advance to the next neuron which fired */\
          ADVANCE(skip);\
\
          /* Apply spike */\
          APPLY();\
\
          /* Prepare to test the neuron after the one we just processed. */\
          ADVANCE(1);\
          skip++;              /* Also skip the neuron we just decoded */\
          pop_length -= skip;  /* Reduce the number of neurons left */\
          n -= skip;           /* and the number left in this word. */\
          data <<= skip;       /* Shift out processed neurons */\
        }\
        else\
        {\
          /* There are no neurons left in this word */\
          ADVANCE(n);       /* Advance to the next neuron */\
          pop_length -= n;  /* Reduce the number left in the population */\
          n = 0;            /* No more neurons left to process */\
        }\
      }\
    }\
  }\
})

#endif  // __SPIKE_VECTOR_PROCESSOR_H__