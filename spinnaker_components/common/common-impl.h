#ifndef __COMMON_IMPLH_H__
#define __COMMON_IMPLH_H__

#include <stdint.h>
#include "common-typedefs.h"

#ifndef use
#define use(x) do {} while ((x)!=(x))
#endif

extern uint32_t simulation_ticks;

address_t system_load_sram();
address_t region_start(uint32_t n, address_t sdram_base);

/** Get the number of ticks for the next simulation.
 */
void config_get_n_ticks();

#endif
