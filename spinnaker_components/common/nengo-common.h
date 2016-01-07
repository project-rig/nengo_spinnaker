#include "spin1_api.h"

#ifndef __NENGO_COMMON_H__
#define __NENGO_COMMON_H__

/* Debug messages */
#ifdef DEBUG
#define debug(MESSAGE, ...) \
  do { \
    io_printf(IO_BUF, "%s:%d " MESSAGE, __FILE__, __LINE__, ##__VA_ARGS__); \
  } while (0)
#else
#define debug(MESSAGE, ...) \
  do { } while(0)
#endif

/** \def __MALLOC_FAIL
 * Will malloc \a SIZE and save the return pointer in \a VAR.  Should the
 * malloc fail \a DESC and an explanation will be stored in IO_BUF and
 * \a VAL will be returned on behalf of the calling function.
 */
#define __MALLOC_FAIL(VAR, SIZE, VAL) \
do { \
  if ((SIZE) == 0) { \
    VAR = NULL; \
  } else { \
    VAR = spin1_malloc(SIZE); \
    if (VAR == NULL) { \
      io_printf(IO_BUF, "%s:%d Failed to malloc " #VAR " (%d bytes)\n", \
                __FILE__, __LINE__, SIZE); \
      return VAL; \
    } else { \
      io_printf(IO_BUF, "%s:%d Malloc " #VAR " (%d bytes)\n", \
                __FILE__, __LINE__, SIZE); \
    } \
  } \
} while (0)

/*! \def MALLOC_FAIL_NULL
 *  Will malloc \a SIZE and save the returned pointer in \a VAR.  Should the
 *  malloc fail \a DESC will be printed to IO_BUF and NULL will be returned
 *  on behalf of the calling function.
 */
#define MALLOC_FAIL_NULL(VAR, SIZE) \
  __MALLOC_FAIL(VAR, SIZE, NULL)

/*! \def MALLOC_FAIL_FALSE
 *  Will malloc \a SIZE and save the returned pointer in \a VAR.  Should the
 *  malloc fail \a DESC will be printed to IO_BUF and FALSE will be returned
 *  on behalf of the calling function.
 */
#define MALLOC_FAIL_FALSE(VAR, SIZE) \
  __MALLOC_FAIL(VAR, SIZE, false)


#define MALLOC_OR_DIE(VAR, SIZE) \
do { \
  if ((SIZE) == 0) \
    VAR = NULL; \
  else \
  { \
    io_printf(IO_BUF, "%s:%d Malloc " #VAR " (%d bytes)\n", \
              __FILE__, __LINE__, SIZE); \
    VAR = spin1_malloc(SIZE); \
    if (VAR == NULL) \
      rt_error(RTE_MALLOC); \
  } \
} while (0)

#endif
