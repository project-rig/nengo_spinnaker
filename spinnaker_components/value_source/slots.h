#ifndef __SLOTS_H__
#define __SLOTS_H__

typedef struct __slot_t {
  uint* data;
  uint current_pos;
  uint length;
} _slot_t;

typedef struct _slots_t {
  _slot_t* current; //!< The current slot
  _slot_t* next;    //!< The next slot

  _slot_t slots[2];
} slots_t;

static inline bool initialise_slots(slots_t* slots, uint size) {
  // Initialise the slots with the given size
  for (uint i = 0; i < 2; i++) {
    MALLOC_FAIL_FALSE(slots->slots[i].data, size);
    slots->slots[i].current_pos = 0;
    slots->slots[i].length = 0;
  }

  slots->current = &slots->slots[0];
  slots->next = &slots->slots[1];
  return true;
}

static inline void slots_progress(slots_t* slots) {
  // Swap the slots pointers
  _slot_t* t = slots->next;
  slots->next = slots->current;
  slots->current = t;

  // Clear the new next slot
  slots->next->length = 0;
  slots->next->current_pos = 0;
}

#endif
