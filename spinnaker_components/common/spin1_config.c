// TODO: Determine where these methods should live
#include "sark.h"
#include "spin1_api.h"
#include "common-impl.h"

uint32_t simulation_ticks = 0;  // TODO: Remove this

// XXX: Will be included as part of next version of SCAMP/SARK
// Get a pointer to a tagged allocation. If the "app_id" parameter is zero
// uses the core's app_id.
void *sark_tag_ptr (uint tag, uint app_id)
{
	if (app_id == 0)
		app_id = sark_vec->app_id;

	return (void *) sv->alloc_tag[(app_id << 8) + tag];
}

address_t system_load_sram()
{
  // Get the block of SDRAM associated with this core and application ID.
  return (address_t) sark_tag_ptr(spin1_get_core_id(), 0);
}


address_t region_start(uint32_t n, address_t address)
{
  return (& address[address[n] >> 2]);
}


void config_get_n_ticks()
{
  // Read the number of ticks from VCPU->User1
  vcpu_t vcpu = ((vcpu_t*)SV_VCPU)[spin1_get_core_id()];
  simulation_ticks = vcpu.user1;
}
