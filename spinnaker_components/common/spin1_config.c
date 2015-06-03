// TODO: Determine where these methods should live
#include "sark.h"
#include "spin1_api.h"
#include "common-impl.h"

uint32_t simulation_ticks = 0;  // TODO: Remove this


address_t system_load_sram()
{
  // We return the address stored in user0, after getting the number of
  // simulation ticks from user1.
  vcpu_t vcpu = ((vcpu_t*)SV_VCPU)[spin1_get_core_id()];
  address_t address = (address_t) vcpu.user0;

  return address;
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
