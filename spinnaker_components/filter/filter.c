/* Parallel implementation of the filter operator
 * ----------------------------------------------
 *
 * Each core in the parallel filter is responsible for receiving packets for a
 * subspace of the value represented by the filter and for transmitting
 * transformed packets for a different subspace.  The general mode of operation
 * on the timer update is:
 *
 *  1. Filter received values
 *  2. Write filtered value into SDRAM
 *  3. Read whole vector from SDRAM
 *  4. Apply local portion of the transform and transmit transformed packets
 *
 * ---
 *
 * The following SDRAM regions are expected:
 *
 *  1. System region (see `filter_parameters_t`)
 *  2. Output keys
 *  3. Filter parameters
 *  4. Filter routes
 *  5. Transform
 *
 * ---
 */

/*****************************************************************************/
#define MAX_VALS 1024  // 4kiB per page (in words)
#define N_PAGES 8      // 8 pages
#define PAGE_MASK 0x7  // 3-bits

#include <stdint.h>
#include "spin1_api.h"

#include "nengo_typedefs.h"
#include "fixed_point.h"

#include "input_filtering.h"
#include "common-impl.h"

#define min(X, Y) ((X) < (Y) ? (X) : (Y))

/*****************************************************************************/
// Pages of the transform and method associated with performing the paging.
typedef struct _transform_page
{
  uint32_t first_row;    // Index of the first row
  value_t data[MAX_VALS];  // Up to 10KiB of transform matrix
} transform_page_t;

typedef struct _pages
{
  uint32_t rows_per_page;           // Number of rows in a page
  transform_page_t pages[N_PAGES];  // Pages of transform data
} pages_t;
/*****************************************************************************/

/*****************************************************************************/
// Global variables
// General parameters (system region)
typedef struct filter_parameters
{
  uint32_t machine_timestep;  // Machine time step / useconds

  uint32_t whole_dimensions;  // Size of the entire vector

  struct
  {
    uint32_t offset;  // Index of first dimension
    uint32_t n_dims;  // Number of dimensions
  } input_subspace;  // Subspace of the input managed by this instance

  uint32_t output_subspace_n_dims;  // Size of output

  value_t *vector;  // Shared input vector (pointer to SDRAM)
} filter_parameters_t;
static filter_parameters_t parallel_filter;

static if_collection_t filters;  // Locally applied filters

static value_t *vector_subspace; // First item of our subspace of the vector

static value_t *input;        // The complete input vector (copy in DTCM)

static pages_t pages;  // Pages of the transform
static value_t *transform;  // Our portion of the transform matrix (in SDRAM)

uint32_t *keys;       // Multicast keys
/*****************************************************************************/

/*****************************************************************************/
// DMA Handling

// The op codes indicate the action performed by the DMA transfer, 2 bits. This
// is always the two most significant bits of any DMA tag.
enum
{
  WRITE_FILTERED_VECTOR = 0,
  READ_WHOLE_VECTOR = 1,
  READ_TRANSFORM_ROWS = 2,
} dma_tag_ops;

// Format of DMA tags
typedef union
{
  struct
  {
    unsigned int op      : 2;  // DMA operation
    unsigned int page_id : 3;  // Page associated with DMA request
  } fields;

  uint32_t as_int;  // Tag as may be used in DMA requests
} dma_tag_t;
/*****************************************************************************/


/*****************************************************************************/
// Get the index of the first row in a page
static inline uint get_page_first_row(pages_t *pages, uint page)
{
  return pages->pages[page].first_row;
}

// Get a pointer to the ith row of the transform matrix
static inline value_t* get_transform_row(
  pages_t *pages,  // Page collection
  uint page,       // Page index
  uint row,        // Index of row to retrieve
  uint n_cols      // Number of cols
)
{
  return &pages->pages[page].data[row * n_cols];
}

// Schedule a DMA page transfer
static inline bool dma_page_schedule(
  pages_t *pages,     // Page collection
  uint next_page,     // Page to transfer into
  value_t *transform, // Matrix to transfer from
  uint row,           // First row to transfer
  uint n_rows,        // Number of rows in the matrix
  uint n_cols         // Number of columns in the matrix
)
{
  // Compute the number of rows to read
  n_rows = min(n_rows - row, pages->rows_per_page);

  // Store the index of the next page
  pages->pages[next_page].first_row = row;

  // Construct the tag for the DMA
  dma_tag_t tag;
  tag.fields.op = READ_TRANSFORM_ROWS;
  tag.fields.page_id = next_page;

  // Schedule the DMA
  return spin1_dma_transfer(
    tag.as_int,                        // Tag
    &transform[row * n_cols],          // SDRAM address
    pages->pages[next_page].data,      // DTCM address
    DMA_READ,
    n_rows * n_cols * sizeof(value_t)  // Size
  );
}
/*****************************************************************************/

/*****************************************************************************/
// Process a page of the transform
static inline void perform_page_multiply(
  pages_t *pages,     // Page collection
  uint page,          // Page to process
  value_t *vector,    // Input vector
  uint n_rows,        // Number of rows in the matrix
  uint n_cols,        // Number of columns in the matrix
  uint32_t *keys      // Multicast keys
)
{
  // For every output value first compute the value and then transmit
  // `d` indexes the global row and `i` the row within the page
  for (uint d = get_page_first_row(pages, page), i=0;
      d < n_rows && i < pages->rows_per_page; d++, i++)
  {
    // Get the row of the transform matrix
    value_t *row = get_transform_row(pages, page, i, n_cols);

    // Calculate the output value
    value_t output = dot_product(n_cols, row, vector);

    // Transmit this value
    while (!spin1_send_mc_packet(keys[d], bitsk(output), WITH_PAYLOAD))
    {
      spin1_delay_us(1);
    }
  }
}
/*****************************************************************************/

/*****************************************************************************/
// Process a page of the transform and schedule a new DMA
static inline void process_transform_page(
  pages_t *pages,     // Page collection
  uint page,          // Page to process
  value_t *vector,    // Input vector
  value_t *transform, // Matrix to transfer from
  uint n_rows,        // Number of rows in the matrix
  uint n_cols,        // Number of columns in the matrix
  uint32_t *keys      // Multicast keys
)
{
  // Schedule paging in the next block of the transform
  uint next_page = (page + 1) & PAGE_MASK;  // Index of the next page to use
  if (get_page_first_row(pages, page) + pages->rows_per_page >= n_rows)
  {
    // Getting the next page is unnecessary, this is the last page of the
    // matrix to process.
  }
  else
  {
    // Schedule retrieval of the next page from memory
    uint next_row = get_page_first_row(pages, page) + pages->rows_per_page;
    dma_page_schedule(pages, next_page, transform, next_row, n_rows, n_cols);
  }

  // Process the given page of the transform
  perform_page_multiply(pages, page, vector, n_rows, n_cols, keys);
}
/*****************************************************************************/

/*****************************************************************************/
// DMA transfer done callback
void dma_transfer_done(uint transfer_id, uint itag)
{
  use(transfer_id); // Unused

  // Cast the tag
  dma_tag_t tag;
  tag.as_int = itag;

  if (tag.fields.op == WRITE_FILTERED_VECTOR)
  {
    // After writing the filtered vector into SDRAM read the whole vector back,
    // this will include the filtered subspaces managed by other cores.
    tag.fields.op = READ_WHOLE_VECTOR;
    spin1_dma_transfer(
      tag.as_int,               // Tag value
      parallel_filter.vector,   // SDRAM address
      input,                    // DTCM address
      DMA_READ,                 // Direction
      parallel_filter.whole_dimensions * sizeof(value_t) // Size
    );
  }
  else if (tag.fields.op == READ_WHOLE_VECTOR)
  {
    // After reading the whole vector from SDRAM schedule transferring the
    // first page of the transform matrix across.
    dma_page_schedule(&pages, 0, transform, 0,
                      parallel_filter.output_subspace_n_dims,
                      parallel_filter.whole_dimensions);
  }
  else if (tag.fields.op == READ_TRANSFORM_ROWS)
  {
    // After reading a page of the transform matrix back process the page (this
    // will schedule the transferral of the next page).
    process_transform_page(&pages, tag.fields.page_id, input, transform,
                           parallel_filter.output_subspace_n_dims,
                           parallel_filter.whole_dimensions, keys);
  }
}
/*****************************************************************************/

/*****************************************************************************/
// Timer tick
void timer_tick(uint ticks, uint arg1)
{
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks)
  {
    spin1_exit(0);
    return;
  }

  // Update the filters
  input_filtering_step(&filters);

  // Start the transform processing pipeline.  This will first copy the locally
  // filtered vector into a larger vector managed in SDRAM; this larger vector
  // will then be copied back and multiplied by a portion of the overall
  // transform matrix to derive the final output packets.

  // Construct the tag
  dma_tag_t tag;
  tag.fields.op = WRITE_FILTERED_VECTOR;
  tag.fields.page_id = 0;

  // Schedule the transfer
  spin1_dma_transfer(
    tag.as_int,                   // Tag
    vector_subspace,              // SDRAM address
    filters.output,               // DTCM address
    DMA_WRITE,                    // Direction
    parallel_filter.input_subspace.n_dims * sizeof(value_t)  // Size
  );
}
/*****************************************************************************/

/*****************************************************************************/
// Multicast packet handling
void multicast_packet_payload(uint key, uint payload)
{
  input_filtering_input_with_dimension_offset(
    &filters, key, payload,
    parallel_filter.input_subspace.offset,     // Offset for all packets
    parallel_filter.input_subspace.n_dims - 1  // Max expected dimension
  );
}
/*****************************************************************************/

/*****************************************************************************/
// Retrieve the multicast packet keys from SDRAM
static bool data_get_output_keys(address_t addr)
{
  // Calculate how big the keys will be
  uint key_size = parallel_filter.output_subspace_n_dims * sizeof(uint);

  // Allocate space for the keys and then copy in from SDRAM
  MALLOC_FAIL_FALSE(keys, key_size);
  spin1_memcpy(keys, addr, key_size);
  return true;
}
/*****************************************************************************/

/*****************************************************************************/
void c_main(void)
{
  // Start loading data from SDRAM
  address_t address = system_load_sram();

  // Copy in the parameters
  spin1_memcpy(&parallel_filter, region_start(1, address),
               sizeof(filter_parameters_t));

  if (!data_get_output_keys(region_start(2, address)))
  {
    io_printf(IO_BUF, "[Filter] Failed to initialise.\n");
    return;
  }

  // Compute the number of rows per page
  pages.rows_per_page = MAX_VALS / parallel_filter.whole_dimensions;

  // Store the address of the transform
  transform = (value_t *) region_start(5, address);

  // Store the address of our subspace of the input vector
  vector_subspace =
    &parallel_filter.vector[parallel_filter.input_subspace.offset];

  // Malloc sufficient space for our copy of the input vector
  MALLOC_OR_DIE(input, sizeof(value_t) * parallel_filter.whole_dimensions);

  // Prepare the filters for receiving packets
  input_filtering_get_filters(&filters, region_start(3, address));
  input_filtering_get_routes(&filters, region_start(4, address));
  input_filtering_initialise_output(
    &filters, parallel_filter.input_subspace.n_dims
  );

  // Register callbacks
  spin1_set_timer_tick(parallel_filter.machine_timestep);
  spin1_callback_on(MCPL_PACKET_RECEIVED, multicast_packet_payload, -1);
  spin1_callback_on(DMA_TRANSFER_DONE, dma_transfer_done, 0);
  spin1_callback_on(TIMER_TICK, timer_tick, 2);

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
