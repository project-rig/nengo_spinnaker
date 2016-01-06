/* Selected functions from the ARM C Language Extension
 *
 * NOTE: This header will only work correctly with a GCC compiler.
 */

#include <stdint.h>

#ifndef __ARM_ACLE_GCC_SELECTED_H__
#define __ARM_ACLE_GCC_SELECTED_H__

// This instruction multiplies two signed 32-bit integers
static inline int64_t __smull(int32_t x, int32_t y)
{
  register union {struct {uint32_t lo; uint32_t hi;} words; int64_t val;} result;

  __asm__ __volatile__("smull %[r_lo], %[r_hi], %[x], %[y]"
                       : [r_lo] "=&r" ((result.words).lo),
                         [r_hi] "=&r" ((result.words).hi)
                       : [x] "r" (x),
                         [y] "r" (y)
                       :);

  return result.val;
}

// This instruction multiplies two signed 32-bit integers and accumulates the
// result.
static inline int64_t __smlal(int64_t acc, int32_t x, int32_t y)
{
  register union {struct {uint32_t lo; uint32_t hi;} words; int64_t val;} result;
  result.val = acc;

  __asm__ __volatile__("smlal %[r_lo], %[r_hi], %[x], %[y]"
                       : [r_lo] "+r" ((result.words).lo),
                         [r_hi] "+r" ((result.words).hi)
                       : [x] "r" (x),
                         [y] "r" (y)
                       :);

  return result.val;
}


#endif  // __ARM_ACLE_GCC_SELECTED_H__
