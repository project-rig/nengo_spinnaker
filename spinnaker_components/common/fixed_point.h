/* Optimised fixed point methods.
 */

#include "arm_acle_gcc_selected.h"
#include "nengo_typedefs.h"

#ifndef __FIXED_POINT_H__
#define __FIXED_POINT_H__

/*****************************************************************************/
// Convert from an INT64 to appropriate INT32 fixed point representation

static inline int32_t convert_s32_30_s16_15(int64_t value)
{
  // The lower word is treated as unsigned because we know that the MSB of the
  // low word will be an integral rather than a sign bit and we want to avoid
  // the compiler using ASR rather than LSR when we come to shift this value.
  register union {struct {uint32_t lo; int32_t hi;} words;
                  int64_t value;} result;
  result.value = value;

  // Include the fractional part and 2 bits of the integral part.
  register int32_t result32 = (int32_t) ((result.words).lo >> 15);

  // Include the remainder of the integral part, we will saturate in the next
  // step.  We shift up by 17 to account for the 15 bits of fractional and 2
  // bits of integral that we already have stored.
  result32 |= (result.words).hi << (15 + 2);

  // NOTE: With no saturation the following should compile down to 3
  // instructions:
  //
  //     CMP ((result.words).hi), #0x0
  //     BICGE product, product, #0x8000_0000
  //     ORRLT product, product, #0x8000_0000
  //
  if ((result.words).hi >= 0)
  {
    // If we were to saturate it should happen here for +ve values.
    //   saturated = ((result.words).hi) > (1 << 15);
    //   if (saturated)
    //   {
    //     result32 = INT32_MAX;
    //   }

    // The sign bit should not be set as this value is positive.
    result32 &= ~(1 << 31);
  }
  else
  {
    // If we were to saturate it should happen here for -ve values.
    //   saturated = ~((result.words).hi) <= -(1 << 16);
    //   if (saturated)
    //   {
    //     result32 = -INT32_MAX;
    //   }

    // The sign bit should be set as this value is negative.
    result32 |= (1 << 31);
  }

  // Return the fixed point value
  return result32;
}

/*****************************************************************************/
// Optimised dot product
// Returns the dot product of two vectors of fixed point values.
// NOTE: This dot product is not saturating at all!

static inline value_t dot_product(uint32_t order, value_t *a, value_t *b)
{
  // Initialise the accumulator with the first product
  register int32_t x = bitsk(a[0]);
  register int32_t y = bitsk(b[0]);
  register int64_t acc = __smull(x, y);

  // Include the remaining product terms (looping backward over the vector)
  for (uint32_t i = order - 1; i > 0; i--)
  {
    // Get the individual components to multiply
    x = bitsk(a[i]);
    y = bitsk(b[i]);

    // Perform a signed multiply with accumulate
    //   acc = acc + x * y;
    acc = __smlal(acc, x, y);
  }

  // Convert from the S32.30 value back to S16.15 before returning
  return kbits(convert_s32_30_s16_15(acc));
}

/*****************************************************************************/


#endif  // __FIXED_POINT_H__
