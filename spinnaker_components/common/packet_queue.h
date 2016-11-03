/* Methods and structures required to handle a queue of packets.
 */

#ifndef __PACKET_QUEUE_H__
#define __PACKET_QUEUE_H__

#include <stdbool.h>
#include <stdint.h>
#include "nengo-common.h"

typedef struct
{
  uint32_t key;
  uint32_t payload;
} packet_t;

// Packet queue structure (really a stack)
typedef struct
{
  packet_t *packets;    // The queue
  unsigned int length;  // Length of the queue
  unsigned int current;  // Current position in the queue
} packet_queue_t;


// Create an initialise a packet queue
static inline void packet_queue_init(packet_queue_t *queue,
                                     unsigned int length)
{
  // Allocate space for the queue
  MALLOC_OR_DIE(queue->packets, length * sizeof(packet_t));

  // Store the length and current position
  queue->length = length;
  queue->current = 0;
}


// Add a packet to the queue
static inline bool packet_queue_push(packet_queue_t *queue,
                                     uint32_t key, uint32_t payload)
{
  if (queue->current < queue->length)
  {
    // Add the packet to the queue if it isn't full
    queue->packets[queue->current].key = key;
    queue->packets[queue->current].payload = payload;
    queue->current++;
    return true;
  }
  else
  {
    // Otherwise return false to indicate that the queue was full
    return false;
  }
}


// Pop a packet from the queue, returning true or false to indicate whether
// this succeeded.
static inline bool packet_queue_pop(packet_queue_t *queue,
                                    packet_t *dest)
{
  if (queue->current > 0)
  {
    // Decrement the current index and then copy the key and payload to the
    // destination.
    queue->current--;
    dest->key = queue->packets[queue->current].key;
    dest->payload = queue->packets[queue->current].payload;

    return true;  // Indicate that a packet was popped
  }
  else
  {
    // Otherwise return false to indicate that the queue was empty
    return false;
  }
}


// Query if a queue is empty
static inline bool packet_queue_not_empty(packet_queue_t *queue)
{
  return queue->current > 0;
}

#endif  // __PACKET_QUEUE_H__
