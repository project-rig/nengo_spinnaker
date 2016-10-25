def minimise(on_set, off_set, used_columns=set()):
    """Minimise a set of keys and masks.

    Parameters
    ----------
    on_set : {(key, mask), ...}
        Set of keys and masks to minimise.
    off_set : {(key, mask), ...}
        Set of keys and masks which should *not* be covered by the minimised
        version of the "on-set".

    Returns
    -------
    {(key, mask), ...}
        A set of keys and masks which covers all the terms in the on-set while
        covering none of the terms in the off-set.

    Uses the "Critical Column First" algorithm presented by:

        Yang, Ze, and Kwan L. Yeung. "An efficient flow monitoring algorithm
        using a flexible match structure." High Performance Switching and
        Routing (HPSR), 2016 IEEE 17th International Conference on. IEEE, 2016.
    """
    # Copy the set of columns that have been chosen already
    used_columns = {x for x in used_columns}

    if len(off_set) == 0:
        # If there is no off-set then yield a key and mask combination which
        # will match everything in the on-set.
        any_ones = 0x00000000  # Bits which are 1 in any entry
        all_ones = 0xffffffff  # Bits which are 1 in all entries
        all_selected = 0xffffffff  # Bits which are 1 in all masks

        # Determine which bits to set to 0, 1 and X
        for key, mask in on_set:
            any_ones |= key
            all_ones &= key
            all_selected &= mask

        any_zeros = ~all_ones
        new_xs = any_ones ^ any_zeros

        mask = new_xs & all_selected  # Combine new Xs with existing Xs
        key = all_ones & mask
        yield key, mask
    else:
        # Otherwise determine a column that can be used to break the on- and
        # off-sets apart.
        on_xs, on_zeros, on_ones = _count_bits(on_set)
        off_xs, off_zeros, off_ones = _count_bits(off_set)

        no_xs = tuple(not(a or b) for a, b in zip(on_xs, off_xs))
        zeros = tuple(a - b for a, b in zip(on_zeros, off_zeros))
        ones = tuple(a - b for a, b in zip(on_ones, off_ones))
        scores = tuple(max(p0, p1) for p0, p1 in zip(zeros, ones))

        # Get the best column
        best_column = None
        for i, (score, valid) in enumerate(zip(scores, no_xs)):
            if valid and i not in used_columns:
                if best_column is None or scores[best_column] < score:
                    best_column = i

        # Break the entries apart based on the value of this column
        new_on_set_zeros, new_on_set_ones = _break_set(on_set, best_column)
        new_off_set_zeros, new_off_set_ones = _break_set(off_set, best_column)

        used_columns.add(best_column)  # Mark the column as used

        if len(new_on_set_zeros) > 0:
            for entry in minimise(new_on_set_zeros, new_off_set_zeros,
                                  used_columns):
                yield entry

        if len(new_on_set_ones) > 0:
            for entry in minimise(new_on_set_ones, new_off_set_ones,
                                  used_columns):
                yield entry


def _count_bits(entries):
    xs = [False for _ in range(32)]
    zeros = [0 for _ in range(32)]
    ones = [0 for _ in range(32)]

    for key, mask in entries:
        for i in range(32):
            bit = 1 << i

            if mask & bit:
                if not key & bit:
                    zeros[i] += 1
                else:
                    ones[i] += 1
            else:
                xs[i] = True

    return tuple(xs), tuple(zeros), tuple(ones)


def _break_set(entries, bit):
    zeros = set()
    ones = set()

    bit = 1 << bit  # Select the bit

    for key, mask in entries:
        assert mask & bit

        if not key & bit:
            zeros.add((key, mask))
        else:
            ones.add((key, mask))

    return tuple(zeros), tuple(ones)
