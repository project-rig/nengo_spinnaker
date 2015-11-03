#include <stdint.h>

#include "spin1_api.h"
#include "input_filtering.h"
#include "nengo-common.h"
#include "fixed_point.h"

#include "common-impl.h"

#define min(X, Y) ((X) < (Y) ? (X) : (Y))

// The size of transform matrix to page
#define MAX_VALS 256*10  // Up to 10KiB of transform matrix

/*****************************************************************************/
// Filter parameters (and system region)
typedef struct filter_parameters
{
  uint32_t size_in;              //!< Number of columns in transform
  uint32_t size_out;             //!< Number of rows in transform
  uint32_t machine_timestep;     //!< Machine time step / useconds
  uint32_t transmission_period;  //!< Ticks between output transmissions
} filter_parameters_t;
filter_parameters_t g_filter;

uint transmission_period;  // Number of ticks until packets are next transmitted
if_collection_t g_filters;  // Collection of input filters
value_t *transform;   // Transform matrix to apply
uint32_t *keys;       // Multicast keys
/*****************************************************************************/

/*****************************************************************************/
// Pages of the transform and method associated with performing the paging.

typedef struct _transform_page
{
  uint32_t first_row;    // Index of the first row
  value_t data[MAX_VALS];  // Up to 10KiB of transform matrix
} transform_page_t;

struct
{
  uint32_t rows_per_page;     // Number of rows in a page
  transform_page_t pages[2];  // 2 pages of transform data
} pages;

// Get the index of the first row in a page
inline uint get_page_first_row(uint page)
{
  return pages.pages[page].first_row;
}

// Get a pointer to the ith row of the transform matrix
inline value_t* get_transform_row(uint page, uint i)
{
  return &pages.pages[page].data[i * g_filter.size_in];
}

// Schedule a DMA page transfer
inline bool dma_page_schedule(uint next_page, uint row)
{
  // Compute the number of rows to read
  uint n_rows = min(g_filter.size_out - row, pages.rows_per_page);

  // Store the index of the next page
  pages.pages[next_page].first_row = row;

  // Schedule the DMA
  return spin1_dma_transfer(
    next_page,                               // Tag
    &transform[row * g_filter.size_in],      // SDRAM address
    pages.pages[next_page].data,             // DTCM address
    DMA_READ,
    n_rows * g_filter.size_in * sizeof(value_t)  // Size
  );
}

// Process a row of the transform
void process_transform_page(uint transfer_id, uint tag)
{
  // Process the arguments
  use(transfer_id);  // Unused
  uint page = tag;   // Tag is the page number we should use

  // Schedule paging in the next block of the transform
  uint next_page = page ^ 1;  // Get the index of the next page to use
  if (get_page_first_row(page) + pages.rows_per_page >= g_filter.size_out)
  {
    // Getting the next page is unnecessary, this is the last page of the
    // matrix to process.
  }
  else 
  {
    // Schedule retrieval of the next page from memory
    uint next_first_row = get_page_first_row(page) + pages.rows_per_page;
    dma_page_schedule(next_page, next_first_row);
  }

  // For every output value first compute the value and then transmit
  // `d` indexes the global row and `i` the row within the page
  for (uint d = get_page_first_row(page), i=0;
      d < g_filter.size_out && i < pages.rows_per_page;
      d++, i++)
  {
    // Get the row of the transform matrix
    value_t *row = get_transform_row(page, i);

    // Calculate the output value
    value_t output = dot_product(g_filter.size_in, row, g_filters.output);

    // Transmit this value
    while (!spin1_send_mc_packet(keys[d], bitsk(output), WITH_PAYLOAD))
    {
      spin1_delay_us(1);
    }
  }
}
/*****************************************************************************/

/*****************************************************************************/
// Timer tick
void filter_update(uint ticks, uint arg1)
{
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks)
  {
    spin1_exit(0);
    return;
  }

  // Update the filters
  input_filtering_step(&g_filters);

  // Increment the counter and apply the transform and transmit if necessary
  if (--transmission_period == 0)
  {
    // Reset the remaining delay
    transmission_period = g_filter.transmission_period;

    // Start the transform processing pipeline, first row
    dma_page_schedule(0, 0);
  }
}
/*****************************************************************************/

/*****************************************************************************/
// Retrieve the multicast packet keys from SDRAM
bool data_get_output_keys(address_t addr)
{
  // Allocate space for the keys and then copy in from SDRAM
  MALLOC_FAIL_FALSE(keys, g_filter.size_out * sizeof(uint));
  spin1_memcpy(keys, addr, g_filter.size_out * sizeof(uint));
  return true;
}
/*****************************************************************************/

/*****************************************************************************/
// Multicast Packet with Payload callback
void mcpl_callback(uint key, uint payload)
{
  // Include the packet in the filter inputs
  input_filtering_input(&g_filters, key, payload);
}
/*****************************************************************************/

/*****************************************************************************/
// Initialisation code
void c_main(void)
{
  // Start loading data from SDRAM
  address_t address = system_load_sram();

  // Copy in the parameters
  spin1_memcpy(&g_filter, region_start(1, address),
               sizeof(filter_parameters_t));

  if (!data_get_output_keys(region_start(2, address)))
  {
    io_printf(IO_BUF, "[Filter] Failed to initialise.\n");
    return;
  }

  // Compute the number of rows per page
  pages.rows_per_page = MAX_VALS / g_filter.size_in;

  // Store the address of the transform
  transform = (value_t *) region_start(5, address);

  // Prepare the filters for receiving packets
  input_filtering_get_filters(&g_filters, region_start(3, address));
  input_filtering_get_routes(&g_filters, region_start(4, address));
  input_filtering_initialise_output(&g_filters, g_filter.size_in);

  // Set the initial period until we next transmit packets
  transmission_period = g_filter.transmission_period;

  // Setup timer tick, start
  spin1_set_timer_tick(g_filter.machine_timestep);
  spin1_callback_on(MCPL_PACKET_RECEIVED, mcpl_callback, -1);
  spin1_callback_on(TIMER_TICK, filter_update, 2);

  // DMA transfers indicate that a new page of the transform is ready to
  // process.
  spin1_callback_on(DMA_TRANSFER_DONE, process_transform_page, 0);

  while (true)
  {
    // Wait for data loading, etc.
    event_wait();

    // Determine how long to simulate for
    config_get_n_ticks();

    // Perform the simulation
    spin1_start(SYNC_WAIT);
  }
}
/*****************************************************************************/
