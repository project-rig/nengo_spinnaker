/*
 * stdfix-full-iso.h
 *
 *
 *  SUMMARY
 *    Additions to the stdfix.h file to support full
 *    Draft ISO/IEC standards compliance.
 *
 *  AUTHOR
 *    Dave Lester (david.r.lester@manchester.ac.uk)
 *
 *  COPYRIGHT
 *    Copyright (c) Dave Lester and The University of Manchester, 2013.
 *    All rights reserved.
 *    SpiNNaker Project
 *    Advanced Processor Technologies Group
 *    School of Computer Science
 *    The University of Manchester
 *    Manchester M13 9PL, UK
 *
 *  DESCRIPTION
 *    
 *
 *  CREATION DATE
 *    12 December, 2013
 *
 *  HISTORY
 * *  DETAILS
 *    Created on       : 12 December 2013
 *    Version          : $Revision$
 *    Last modified on : $Date$
 *    Last modified by : $Author$
 *    $Id$
 *
 *    $Log$
 *
 */

#ifndef __STDFIX_FULL_ISO_H__
#define __STDFIX_FULL_ISO_H__

#include <stdint.h>
#include <stdbool.h>

#define __stdfix_min(a,b) (((a)<(b))? (a): (b))
#define __stdfix_max(a,b) (((a)<(b))? (b): (a))
#define __stdfix_use(a)   do {} while ((a)!=(a))
#define __stdfix_abs(a)   (((a)<0)? -(a): (a))

#define __ms_u32(x) ((x) >> 32)
#define __ls_u32(x) ((x) & UINT32_MAX)

#define STATIC_INLINE static __inline__ __attribute__((__always_inline__))

void __64x64_128 (uint64_t *hi, uint64_t *lo, uint64_t x, uint64_t y);

typedef int8_t  int_hr_t;
typedef int16_t int_r_t;
typedef int32_t int_lr_t;
typedef int16_t int_hk_t;
typedef int32_t int_k_t;
typedef int64_t int_lk_t;

typedef uint8_t  uint_uhr_t;
typedef uint16_t uint_ur_t;
typedef uint32_t uint_ulr_t;
typedef uint16_t uint_uhk_t;
typedef uint32_t uint_uk_t;
typedef uint64_t uint_ulk_t;

#ifdef __arm__
#include <stdfix.h>

typedef  short fract s07;
typedef        fract s015;
typedef  long  fract s031;
typedef  short accum s87;
typedef        accum s1615;
typedef  long  accum s3231;

typedef unsigned short fract u08;
typedef unsigned       fract u016;
typedef unsigned long  fract u032;
typedef unsigned short accum u88;
typedef unsigned       accum u1616;
typedef unsigned long  accum u3232;

#else /* ! __arm__ */

typedef  int8_t  s07;
typedef  int16_t s015;
typedef  int32_t s031;
typedef  int16_t s87;
typedef  int32_t s1615;
typedef  int64_t s3231;

typedef uint8_t  u08;
typedef uint16_t u016;
typedef uint32_t u032;
typedef uint16_t u88;
typedef uint32_t u1616;
typedef uint64_t u3232;

#endif /* __arm__ */

// 7.18a.6.5 The bitwise fixed-point to integer conversion functions

static inline int_hr_t   bitshr  (const s07   f)
{ union { int_hr_t   r; s07   fx;} x; x.fx = f; return (x.r); }
static inline int_r_t    bitsr   (const s015  f)
{ union { int_r_t    r; s015  fx;} x; x.fx = f; return (x.r); }
static inline int_lr_t   bitslr  (const s031  f)
{ union { int_lr_t   r; s031  fx;} x; x.fx = f; return (x.r); }
static inline int_hk_t   bitshk  (const s87   f)
{ union { int_hk_t   r; s87   fx;} x; x.fx = f; return (x.r); }
static inline int_k_t    bitsk   (const s1615 f)
{ union { int_k_t    r; s1615 fx;} x; x.fx = f; return (x.r); }
static inline int_lk_t   bitslk  (const s3231 f)
{ union { int_lk_t   r; s3231 fx;} x; x.fx = f; return (x.r); }
static inline uint_uhr_t bitsuhr (const u08   f)
{ union { uint_uhr_t r; u08   fx;} x; x.fx = f; return (x.r); }
static inline uint_ur_t  bitsur  (const u016  f)
{ union { uint_ur_t  r; u016  fx;} x; x.fx = f; return (x.r); }
static inline uint_ulr_t bitsulr (const u032  f)
{ union { uint_ulr_t r; u032  fx;} x; x.fx = f; return (x.r); }
static inline uint_uhk_t bitsuhk (const u88   f)
{ union { uint_uhk_t r; u88   fx;} x; x.fx = f; return (x.r); }
static inline uint_uk_t  bitsuk  (const u1616 f)
{ union { uint_uk_t  r; u1616 fx;} x; x.fx = f; return (x.r); }
static inline uint_ulk_t bitsulk (const u3232 f)
{ union { uint_ulk_t r; u3232 fx;} x; x.fx = f; return (x.r); }

// 7.18a.6.6 The bitwise integer to fixed-point conversion functions

static inline s07   hrbits  (const int_hr_t   n)
{ union { int_hr_t   r; s07   fx;} x; x.r = n; return (x.fx); }
static inline s015  rbits   (const int_r_t    n)
{ union { int_r_t    r; s015  fx;} x; x.r = n; return (x.fx); }
static inline s031  lrbits  (const int_lr_t   n)
{ union { int_lr_t   r; s031  fx;} x; x.r = n; return (x.fx); }
static inline s87   hkbits  (const int_hk_t   n)
{ union { int_hk_t   r; s87   fx;} x; x.r = n; return (x.fx); }
static inline s1615 kbits   (const int_k_t    n)
{ union { int_k_t    r; s1615 fx;} x; x.r = n; return (x.fx); }
static inline s3231 lkbits  (const int_lk_t   n)
{ union { int_lk_t   r; s3231 fx;} x; x.r = n; return (x.fx); }
static inline u08   uhrbits (const uint_uhr_t n)
{ union { uint_uhr_t r; u08   fx;} x; x.r = n; return (x.fx); }
static inline u016  urbits  (const uint_ur_t  n)
{ union { uint_ur_t  r; u016  fx;} x; x.r = n; return (x.fx); }
static inline u032  ulrbits (const uint_ulr_t n)
{ union { uint_ulr_t r; u032  fx;} x; x.r = n; return (x.fx); }
static inline u88   uhkbits (const uint_uhk_t n)
{ union { uint_uhk_t r; u88   fx;} x; x.r = n; return (x.fx); }
static inline u1616 ukbits  (const uint_uk_t  n)
{ union { uint_uk_t  r; u1616 fx;} x; x.r = n; return (x.fx); }
static inline u3232 ulkbits (const uint_ulk_t n)
{ union { uint_ulk_t r; u3232 fx;} x; x.r = n; return (x.fx); }

// Saturation operations

static inline int32_t __stdfix_sat_hr (const int32_t x)
{
  if (x > INT8_MAX) return (INT8_MAX);
  if (x < INT8_MIN) return (INT8_MIN);

  return (x);
}

static inline int32_t __stdfix_sat_r (const int32_t x)
{
  if (x > INT16_MAX) return (INT16_MAX);
  if (x < INT16_MIN) return (INT16_MIN);

  return (x);
}

static inline int32_t __stdfix_sat_lr (const int64_t x)
{
  if (x > INT32_MAX) return (INT32_MAX);
  if (x < INT32_MIN) return (INT32_MIN);

  return ((int32_t)x);
}

static inline int32_t __stdfix_sat_hk   (const int32_t x)
{
  if (x > INT16_MAX) return (INT16_MAX);
  if (x < INT16_MIN) return (INT16_MIN);

  return (x);
}

static inline int32_t __stdfix_sat_k (const int64_t x)
{
  if (x > INT32_MAX) return (INT32_MAX);
  if (x < INT32_MIN) return (INT32_MIN);

  return ((int32_t)x);
}

static inline uint32_t __stdfix_sat_uhr (const uint32_t x)
{
  if (x > UINT8_MAX) return (UINT8_MAX);

  return (x);
}

static inline uint32_t __stdfix_sat_ur (const uint32_t x)
{
  if (x > UINT16_MAX) return (UINT16_MAX);

  return (x);
}

static inline uint32_t __stdfix_sat_ulr (const uint64_t x)
{
  if (x > UINT32_MAX) return (UINT32_MAX);

  return ((uint32_t)x);
}

static inline uint32_t __stdfix_sat_uhk   (const uint32_t x)
{
  if (x > UINT16_MAX) return (UINT16_MAX);

  return (x);
}

static inline uint32_t __stdfix_sat_uk (const uint64_t x)
{
  if (x > UINT32_MAX) return (UINT32_MAX);

  return ((uint32_t)x);
}

// software simulation of basic saturating arithmetic

static inline int32_t   __stdfix_sadd_hr (int_hr_t x, int_hr_t y)
{ return (__stdfix_sat_hr ((int32_t)x + (int32_t)y)); }

static inline int32_t   __stdfix_ssub_hr (int_hr_t x, int_hr_t y)
{ return (__stdfix_sat_hr ((int32_t)x - (int32_t)y)); }

static inline int32_t   __stdfix_sneg_hr (int_hr_t x)
{ return (__stdfix_sat_hr ((int32_t)-x)); }

static inline int32_t  __stdfix_sadd_r (int32_t x, int32_t y)
{ return (__stdfix_sat_r ((int32_t)x + (int32_t)y)); }

static inline int32_t  __stdfix_ssub_r (int32_t x, int32_t y)
{ return (__stdfix_sat_r ((int32_t)x - (int32_t)y)); }

static inline int32_t  __stdfix_sneg_r (int32_t x)
{ return (__stdfix_sat_r ((int32_t)-x)); }

static inline int32_t  __stdfix_sadd_lr (int32_t x, int32_t y)
{ return (__stdfix_sat_lr ((int64_t)x + (int64_t)y)); }

static inline int32_t  __stdfix_ssub_lr (int32_t x, int32_t y)
{ return (__stdfix_sat_lr ((int64_t)x - (int64_t)y)); }

static inline int32_t  __stdfix_sneg_lr (int32_t x)
{ return (__stdfix_sat_lr ((int64_t)-x)); }

static inline int32_t   __stdfix_sadd_hk (int32_t x, int32_t y)
{ return (__stdfix_sat_hk ((int32_t)x + (int32_t)y)); }

static inline int32_t   __stdfix_ssub_hk (int32_t x, int32_t y)
{ return (__stdfix_sat_hk ((int32_t)x - (int32_t)y)); }

static inline int32_t   __stdfix_sneg_hk (int32_t x)
{ return (__stdfix_sat_hk ((int32_t)-x)); }

static inline int32_t __stdfix_sadd_k (int32_t x, int32_t y)
{ return (__stdfix_sat_k ((int64_t)x + (int64_t)y)); }

static inline int32_t __stdfix_ssub_k (int32_t x, int32_t y)
{ return (__stdfix_sat_k ((int64_t)x - (int64_t)y)); }

static inline int32_t __stdfix_sneg_k (int32_t x)
{ return (__stdfix_sat_k ((int64_t)-x)); }

int64_t __stdfix_sadd_lk (int64_t x, int64_t y);
int64_t __stdfix_ssub_lk (int64_t x, int64_t y);
int64_t __stdfix_sneg_lk (int64_t x);

static inline int32_t __stdfix_smul_hr (int32_t x, int32_t y)
{
  if (x == INT8_MIN && y == INT8_MIN) return (INT8_MAX);  // special case for -1.0*-1.0

  return ((int32_t)__stdfix_sat_hr (((int32_t)x * (int32_t)y) >> 7));
}

static inline int32_t __stdfix_smul_r (int32_t x, int32_t y)
{
  if (x == INT16_MIN && y == INT16_MIN) return (INT16_MAX); // special case for -1.0*-1.0

  return ((int32_t)__stdfix_sat_r (((int32_t)x * (int32_t)y) >> 15));
}

static inline int32_t __stdfix_smul_lr (int32_t x, int32_t y)
{
  if (x == INT32_MIN && y == INT32_MIN) return (INT32_MAX);  // special case for -1.0*-1.0

  return (__stdfix_sat_lr (((int64_t)x * (int64_t)y) >> 31));
}

static inline int32_t __stdfix_smul_hk (int32_t x, int32_t y)
{
  if (x == INT16_MIN && y == INT16_MIN) return (INT16_MAX); // special case for -1.0*-1.0

  return (__stdfix_sat_hk (((int32_t)x * (int32_t)y) >> 7));
}

static inline int32_t __stdfix_smul_k (int32_t x, int32_t y)
{
  if (x == INT32_MIN && y == INT32_MIN) return (INT32_MAX); // special case for -1.0*-1.0

  return (__stdfix_sat_k (((int64_t)x * (int64_t)y) >> 15));
}

int64_t __stdfix_smul_lk (int64_t x, int64_t y);

static inline uint32_t __stdfix_sadd_uhr (uint32_t x, uint32_t y)
{ return (__stdfix_sat_uhr ((uint32_t)x + (uint32_t)y)); }

static inline uint32_t __stdfix_ssub_uhr (uint32_t x, uint32_t y)
{ return (__stdfix_sat_uhr ((uint32_t)x - (uint32_t)y)); }

static inline uint32_t __stdfix_sadd_ur (uint32_t x, uint32_t y)
{ return (__stdfix_sat_ur ((uint32_t)x + (uint32_t)y)); }

static inline uint32_t __stdfix_ssub_ur (uint32_t x, uint32_t y)
{ return (__stdfix_sat_ur ((uint32_t)x - (uint32_t)y)); }

static inline uint32_t __stdfix_sadd_ulr (uint32_t x, uint32_t y)
{ return (__stdfix_sat_ulr ((uint64_t)x + (uint64_t)y)); }

static inline uint32_t __stdfix_ssub_ulr (uint32_t x, uint32_t y)
{ return (__stdfix_sat_ulr ((uint64_t)x - (uint64_t)y)); }

static inline uint32_t __stdfix_sadd_uhk (uint32_t x, uint32_t y)
{ return (__stdfix_sat_uhk ((uint32_t)x + (uint32_t)y)); }

static inline uint32_t __stdfix_ssub_uhk (uint32_t x, uint32_t y)
{ return (__stdfix_sat_uhk ((uint32_t)x - (uint32_t)y)); }

static inline uint32_t __stdfix_sadd_uk (uint32_t x, uint32_t y)
{ return (__stdfix_sat_uk ((uint64_t)x + (uint64_t)y)); }

static inline uint32_t __stdfix_ssub_uk (uint32_t x, uint32_t y)
{ return (__stdfix_sat_uk ((uint64_t)x - (uint64_t)y)); }

uint64_t __stdfix_sadd_ulk (uint64_t x, uint64_t y);
uint64_t __stdfix_ssub_ulk (uint64_t x, uint64_t y);

static inline uint32_t  __stdfix_smul_uhr  (uint32_t x,  uint32_t y)
{ return (__stdfix_sat_uhr (((uint32_t)x * (uint32_t)y) >> 8));}

static inline uint32_t __stdfix_smul_ur (uint32_t x, uint32_t y)
{ return (__stdfix_sat_ur (((uint32_t)x * (uint32_t)y) >> 16)); }

static inline uint32_t __stdfix_smul_ulr (uint32_t x, uint32_t y)
{ return (__stdfix_sat_ulr (((uint64_t)x * (uint64_t)y) >> 32)); }

static inline uint32_t __stdfix_smul_uhk (uint32_t x, uint32_t y)
{ return (__stdfix_sat_uhk (((uint32_t)x * (uint32_t)y) >> 8)); }

static inline uint32_t __stdfix_smul_uk (uint32_t x, uint32_t y)
{ return (__stdfix_sat_uk (((uint64_t)x * (uint64_t)y) >> 16)); }

uint64_t __stdfix_smul_ulk (uint64_t x, uint64_t y);

// 7.18a.6.1 The fixed-point arithmetic operation support functions Synopsis

static inline int      mulir  (int n, s015  x)
{ return ((int)(((int64_t)(n) * (int64_t)(bitsr  (x)))  >> 15)); }
static inline long int mulilr (long int n, s031  x)
{ return ((int)(((int64_t)(n) * (int64_t)(bitslr (x)))  >> 31)); }
static inline int      mulik  (int n, s1615 x)
{ return ((int)(((int64_t)(n) * (int64_t)(bitsk  (x)))  >> 15)); }
static inline long int mulilk (long int n, s3231 x)
{
  int64_t r = bitslk (x);
  int64_t k = (int64_t)(n) * (r >> 31);
  int64_t c = ((int64_t)(n) * (r & INT64_C(0x7FFFFFF))) >> 31;

  return ((long int)(k + c));
}

static inline int      divir  (int n, s015 x)
{ return ((int)     ( ((int64_t)(n) << 15) / bitsr  (x))); }
static inline long int divilr (long int n, s031 x)
{ return ((long int)( ((int64_t)(n) << 31) / bitslr (x))); }
static inline int      divik  (int n,      s1615 x)
{ return ((long int)( ((int64_t)(n) << 15) / bitsk  (x))); }
static inline long int divilk (long int n, s3231 x)
{ return ((long int)( ((int64_t)(n) << 31) / bitslk (x))); }

static inline s015  rdivi   (int i,      int j)
{ return (rbits  ((int_r_t) (((int64_t)(i) << 15) / ((int64_t)j)))); }
static inline s031  lrdivi  (long int i, long int j)
{ return (lrbits ((int_lr_t)(((int64_t)(i) << 31) / ((int64_t)j)))); }
static inline s1615 kdivi   (int i,      int j)
{ return (kbits  ((int_k_t) (((int64_t)(i) << 15) / ((int64_t)j)))); }
static inline s3231 lkdivi  (long int i, long int j)
{ return (lkbits ((int_lk_t)(((int64_t)(i) << 31) / ((int64_t)j)))); }

static inline int      idivr   (s015  x, s015  y)
{ return (int)     (rbits  (x) / rbits   (y)); }
static inline long int idivlr  (s031  x, s031  y)
{ return (long int)(lrbits (x) / lrbits  (y)); }
static inline int      idivk   (s1615 x, s1615 y)
{ return (int)     (kbits  (x) / kbits   (y)); }
static inline long int idivlk  (s3231 x, s3231 y)
{ return (long int)(lkbits (x) / lkbits  (y)); }

static inline unsigned int      muliur  (unsigned int n, u016 x)
{ return ((unsigned int)(((uint64_t)(n) * (uint64_t)(bitsur  (x)))  >> 16)); }
static inline unsigned long int muliulr (unsigned long int n, u032 x)
{ return ((unsigned long int)(((uint64_t)(n) *
			       (uint64_t)(bitsulr (x)))  >> 32)); }
static inline unsigned int      muliuk  (unsigned int n, u1616 x)
{ return ((unsigned int)( ((uint64_t)(n) << 16) / bitsuk  (x))); }
static inline unsigned long int muliulk (unsigned long int n, u3232 x)
{
  uint64_t r = bitsulk (x);
  uint64_t k = (uint64_t)(n) * (r >> 32);
  uint64_t c = ((uint64_t)(n) * (r & INT64_C(0xFFFFFFF))) >> 32;

  return ((unsigned long int)(k + c));
}

static inline unsigned int      diviur (unsigned int n, u016 x)
{ return ((unsigned int)     ( ((uint64_t)(n) << 16) / bitsur  (x))); }
static inline unsigned long int diviulr (unsigned long int n, u032 x)
{ return ((unsigned long int)( ((uint64_t)(n) << 32) / bitsulr (x))); }
static inline unsigned int      diviuk  (unsigned int n,      u1616 x)
{ return ((unsigned long int)( ((uint64_t)(n) << 16) / bitsuk  (x))); }
static inline unsigned long int diviulk (unsigned long int n, u3232 x)
{ return ((unsigned long int)( ((uint64_t)(n) << 32) / bitsulk (x))); }

static inline u016  urdivi  (unsigned int i, unsigned int j)
{ return (ulrbits ((uint_ulr_t)(((uint64_t)(i) << 16) / ((uint64_t)j)))); }
static inline u032  ulrdivi (unsigned long int i, unsigned long int j)
{ return (ulrbits ((uint_ulr_t)(((uint64_t)(i) << 32) / ((uint64_t)j)))); }
static inline u1616 ukdivi  (unsigned int i, unsigned int j)
{ return (ukbits  ((uint_uk_t) (((uint64_t)(i) << 16) / ((uint64_t)j)))); }
static inline u3232 ulkdivi (unsigned long int i, unsigned long int j)
{ return (ulkbits ((uint_ulk_t)(((uint64_t)(i) << 32) / ((uint64_t)j)))); }

static inline unsigned int      idivur  (u016 x, u016 y)
{ return (unsigned int)     (urbits   (x) / urbits   (y)); }
static inline unsigned long int idivulr (u032 x, u032 y)
{ return (unsigned long int)(ulrbits  (x) / ulrbits  (y)); }
static inline unsigned int      idivuk  (u1616 x, u1616 y)
{ return (unsigned int)     (ukbits   (x) / ukbits   (y)); }
static inline unsigned long int idivulk (u3232 x, u3232 y)
{ return (unsigned long int)(ulkbits (x)  / ulkbits  (y)); }

// Description
//
// The above functions compute the mathematically exact result of the
// multiplication or division operation on the operands with the indicated
// types, and return a value with the indicated type.
//
// Returns
//
// For functions returning an integer value, the return value is rounded
// towards zero. For functions returning a fixed-point value, the return value
// is saturated on overflow. If the second operand of one of the divide
// functions is zero, the behavior is undefined.

// 7.18a.6.2 The fixed-point absolute value functions

static inline s07   abshr (s07   f)
{
  int_hr_t r = bitshr (f);

  if (r < 0)
    r = (r == INT8_MIN)? INT8_MAX: -r;

  return (hrbits (r));
}


static inline s015  absr  (s015  f)
{
  int_r_t r = bitsr (f);

  if (r < 0)
    r = (r == INT16_MIN)? INT16_MAX: -r;

  return (rbits (r));
}

static inline s031  abslr (s031  f)
{
  int_lr_t r = bitslr (f);

  if (r < 0)
    r = (r == INT32_MIN)? INT32_MAX: -r;

  return (lrbits (r));
}


static inline s87   abshk (s87   f)
{
  int_hk_t r = bitshk (f);

  if (r < 0)
    r = (r == INT16_MIN)? INT16_MAX: -r;

  return (hkbits (r));
}

static inline s1615 absk  (s1615 f)
{
  int_k_t r = bitsk (f);

  if (r < 0)
    r = (r == INT32_MIN)? INT32_MAX: -r;

  return (kbits (r));
}


static inline s3231 abslk (s3231 f)
{
  int_lk_t r = bitslk (f);

  if (r < 0)
    r = (r == INT64_MIN)? INT64_MAX: -r;

  return (lkbits (r));
}


// Description
//
// The above functions compute the absolute value of a fixed-point value f.
//
// Returns
//
// The functions return |f|. If the exact result cannot be represented, the
// saturated result is returned.

// 7.18a.6.3 The fixed-point rounding functions

s07   roundhr  (s07   f, int n);
s015  roundr   (s015  f, int n);
s031  roundlr  (s031  f, int n);
s87   roundhk  (s87   f, int n);
s1615 roundk   (s1615 f, int n);
s3231 roundlk  (s3231 f, int n);
u08   rounduhr (u08   f, int n);
u016  roundur  (u016  f, int n);
u032  roundulr (u032  f, int n);
u88   rounduhk (u88   f, int n);
u1616 rounduk  (u1616 f, int n);
u3232 roundulk (u3232 f, int n);

// Description
//
// The above functions compute the value of f, rounded to the number of
// fractional bits specified in n. The rounding applied is to-nearest, with
// unspecified rounding direction in the halfway case. When saturation has not
// occurred, fractional bits beyond the rounding point are set to zero in the
// result. The value of n must be nonnegative and less than the number of
// fractional bits in the fixed-point type of f.
//
// Returns
//
// The rounding functions return the rounded result, as specified. If the
// value of n is negative or larger than the number of fractional bits in the
// fixed-point type of f, the result is unspecified. If the exact result
// cannot be represented, the saturated result is returned.

// 7.18a.6.4 The fixed-point countls functions

static inline int countlshr  (s07   f)
{
  int32_t n = (int32_t)(bitshr (f));

  if (n == 0) return (7);

  return (__builtin_clrsb (n) - 24);
}

static inline int countlsr   (s015  f)
{
  int32_t n = (int32_t)(bitsr (f));

  if (n == 0) return (15);

  return (__builtin_clrsb (n) - 16);
}

static inline int countlslr  (s031  f)
{
  int_lr_t n = bitslr (f);

  if (n == 0) return (31);

  return (__builtin_clrsb (n));
}

static inline int countlshk  (s87   f)
{
  int32_t n = (int32_t)(bitshk (f));

  if (n == 0) return (15);

  return (__builtin_clrsb (n) - 16);
}

static inline int countlsk   (s1615 f)
{
  int_k_t n = bitsk (f);

  if (n == 0) return (31);

  return (__builtin_clrsb (n));
}

static inline int countlslk  (s3231 f)
{
  int64_t n = (int64_t)(bitslk (f));

  if (n == 0) return (63);

  return (__builtin_clrsbll (n));
}

static inline int countlsuhr (u08   f)
{
  uint32_t n = (uint32_t)(bitsuhr (f));

  if (n == 0) return (8);

  return (__builtin_clz (n) - 24);
}

static inline int countlsur  (u016  f)
{
  uint32_t n = (uint32_t)(bitsur (f));

  if (n == 0) return (16);

  return (__builtin_clz (n) - 16);
}

static inline int countlsulr (u032  f)
{
  uint_ulr_t n = bitsulr (f);

  if (n == 0) return (32);

  return (__builtin_clz (n));
}

static inline int countlsuhk (u88   f)
{
  uint32_t n = (uint32_t)(bitsuhk (f));

  if (n == 0) return (16);

  return (__builtin_clz (n) - 16);
}

static inline int countlsuk  (u1616 f)
{
  uint_uk_t n = (uint_uk_t)(bitsuk (f));

  if (n == 0) return (32);

  return (__builtin_clz (n));
}

static inline int countlsulk (u3232 f)
{
  uint_ulk_t n = (uint_ulk_t)(bitsulk (f));

  if (n == 0) return (64);

  return (__builtin_clzll (n));
}

// Description
//
// The integer return value of the above functions is defined as follows:
// - if the value of the fixed-point argument f is non-zero, the return value
//   is the largest integer k for which the expression f<<k does not overflow;
// - if the value of the fixed-point argument is zero, an integer value is
//   returned that is at least as large as N, where N is the total number of
//   value bits of the fixed-point type of the argument.
//
// Note: if the value of the fixed-point argument is zero, the recommended
// return value is exactly N.
//
// Returns
//
// The countls functions return the integer value as indicated.

// 7.18a.6.7 Type-generic fixed-point functions

// For each of the fixed-point absolute value functions in 7.18a.6.2, the
// fixed-point rounding functions in 7.18a.6.3 and the fixed-point countls
// functions in 7.18a.6.4, a type-generic macro is defined as follows:
//
//                                                   type-generic macro
//      the fixed-point absolute value functions     absfx
//      the fixed-point rounding functions           roundfx
//      the fixed-point countls functions            countlsfx
//
// For each macro, use of the macro invokes the function whose corresponding
// type and type domain is the fixed-point type of the first generic argument.
// If the type of the first generic argument is not a fixed-point type, the
// behavior is undefined

#define absfx(f)						      \
  ({								      \
    __typeof__ (f) tmp = (f);					      \
    if      (__builtin_types_compatible_p (__typeof__(f), s07))	      \
      tmp = abshr (f);						      \
    else if (__builtin_types_compatible_p (__typeof__(f), s015))      \
      tmp = absr (f);						      \
    else if (__builtin_types_compatible_p (__typeof__(f), s031))      \
      tmp = abslr (f);						      \
    else if (__builtin_types_compatible_p (__typeof__(f), s87))	      \
      tmp = abshk (f);						      \
    else if (__builtin_types_compatible_p (__typeof__(f), s1615))     \
      tmp = absk (f);						      \
    else if (__builtin_types_compatible_p (__typeof__(f), s3231))     \
      tmp = abslk (f);						      \
    else							      \
      abort ();							      \
    tmp;							      \
})

#define roundfx(f,n)						      \
  ({								      \
    __typeof__ (f) tmp = (f);					      \
    if      (__builtin_types_compatible_p (__typeof__(f), s07))	      \
      tmp = roundhr (f, n);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), s015))      \
      tmp = roundr (f, n);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), s031))      \
      tmp = roundlr (f, n);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), s87))	      \
      tmp = roundhk (f, n);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), s1615))     \
      tmp = roundk (f, n);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), s3231))     \
      tmp = roundlk (f, n);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u08))	      \
      tmp = rounduhr (f, n);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u016))      \
      tmp = roundur (f, n);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u032))      \
      tmp = roundulr (f, n);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u88))	      \
      tmp = rounduhk (f, n);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u1616))     \
      tmp = rounduk (f, n);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u3232))     \
      tmp = roundulk (f, n);					      \
    else							      \
      abort ();							      \
    tmp;							      \
})

#define countlsfx(f)						      \
  ({								      \
    int tmp = 0;						      \
    if      (__builtin_types_compatible_p (__typeof__(f), s07))	      \
      tmp = countlshr (f);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), s015))      \
      tmp = countlsr (f);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), s031))      \
      tmp = countlslr (f);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), s87))	      \
      tmp = countlshk (f);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), s1615))     \
      tmp = countlsk (f);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), s3231))     \
      tmp = countlslk (f);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u08))	      \
      tmp = countlsuhr (f);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u016))      \
      tmp = countlsur (f);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u032))      \
      tmp = countlsulr (f);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u88))	      \
      tmp = countlsuhk (f);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u1616))     \
      tmp = countlsuk (f);					      \
    else if (__builtin_types_compatible_p (__typeof__(f), u3232))     \
      tmp = countlsulk (f);					      \
    else							      \
      abort ();							      \
    tmp;							      \
})

// 7.18a.6.8 Numeric conversion functions

#ifndef __arm__
s07   strtofxhr  (const char * restrict nptr, char ** restrict endptr);
s015  strtofxr   (const char * restrict nptr, char ** restrict endptr);
s031  strtofxlr  (const char * restrict nptr, char ** restrict endptr);
s87   strtofxhk  (const char * restrict nptr, char ** restrict endptr);
s1615 strtofxk   (const char * restrict nptr, char ** restrict endptr);
s3231 strtofxlk  (const char * restrict nptr, char ** restrict endptr);

u08   strtofxuhr (const char * restrict nptr, char ** restrict endptr);
u016  strtofxur  (const char * restrict nptr, char ** restrict endptr);
u032  strtofxulr (const char * restrict nptr, char ** restrict endptr);
u88   strtofxuhk (const char * restrict nptr, char ** restrict endptr);
u1616 strtofxuk  (const char * restrict nptr, char ** restrict endptr);
u3232 strtofxulk (const char * restrict nptr, char ** restrict endptr);
#endif /*__arm__*/

// Description
//
// The strtofxhr, strtofxr, strtofxlr, strtofxhk, strtofxk, strtofxlk,
// strtofxuhr, strtofxur, strtofxulr, strtofxuhk, strtofxuk and strtofxulk
// functions convert the initial portion of the string pointed to by nptr to
// short fract, fract, long fract, short accum, accum, long accum, unsigned
// short fract, unsigned fract, unsigned long fract, unsigned short accum,
// unsigned accum, and unsigned long accum representation, respectively.
// First,they decompose the input string into three parts: an initial,
// possibly empty, sequence of white-space characters (as specified by the is
// space function), a subject sequence resembling a fixed-point constant; and
// a final string of one or more unrecognized characters, including the
// terminating null character of the input string. Then, they attempt to
// convert the subject sequence to a fixed-point number, and return the
// result.
//
// The expected form of the subject sequence is an optional plus or minus
// sign, then one of the following:
//
// - a nonempty sequence of decimal digits optionally containing a
//   decimal-point character, then an optional exponent part as defined in
//   6.4.4.3;
//
// - a 0x or 0X, then a nonempty sequence of hexadecimal digits optionally
//   containing a decimal- point character, then an optional binary exponent
//   part as defined in 6.4.4.3.
//
// The subject sequence is defined as the longest initial subsequence of the
// input string, starting with the first non-white-space character, that is of
// the expected form. The subject sequence contains no characters if the input
// string is not of the expected form.
//
// If the subject sequence has the expected form for a fixed-point number, the
// sequence of characters starting with the first digit or the decimal-point
// character (whichever occurs first) is interpreted as a fixed-point constant
// according to the rules of 6.4.4.3, except that the decimal-point character
// is used in place of a period, and that if neither an exponent part nor a
// decimal-point character appears in a decimal fixed-point number, or if a
// binary exponent part does not appear in a hexadecimal fixed- point number,
// an exponent part of the appropriate type with value zero is assumed to
// follow the last digit in the string. If the subject sequence begins with a
// minus sign, the sequence is interpreted as negated. A pointer to the final
// string is stored in the object pointed to by endptr, provided that endptr
// is not a null pointer.
//
// The value resulting from the conversion is rounded as necessary in an
// implementation-defined manner.
//
// In other than the "C" locale, additional locale-specific subject sequence
// forms may be accepted.
//
// If the subject sequence is empty or does not have the expected form, no
// conversion is performed; the value of nptr is stored in the object pointed
// to by endptr, provided that endptr is not a null pointer.
//
// Returns
//
// The functions return the converted value, if any. If no conversion could be
// performed, zero is returned. If the correct value is outside the range of
// representable values, a saturated result is returned (according to the
// return type and sign of the value), and the value of the macro ERANGE
// is stored in errno.


#endif /* __STDFIX_FULL_ISO_H__ */
