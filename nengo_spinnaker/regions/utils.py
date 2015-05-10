"""Region utilities.
"""
import struct


def create_app_ptr_and_region_files(fp, regions, vertex_slice):
    """Split up a file-like view of memory into smaller views, one per region,
    and write into the first region of memory the offsets to these later
    regions.

    Returns
    -------
    [file-like view, ...]
        A file-like view of memory for each region.
    """
    # First we split off the application pointer region
    ptrs = [0 for n in range(len(regions) + 1)]
    offset = len(ptrs)*4  # 1 word per region

    # Then we go through and assign each region in turn
    region_memory = list()
    for i, r in enumerate(regions, start=1):
        if r is None:
            region_memory.append(None)
        else:
            ptrs[i] = offset
            next_offset = offset + r.sizeof_padded(vertex_slice)
            region_memory.append(fp[offset:next_offset])
            offset = next_offset

    # Write in the pointer table
    fp.seek(0)
    fp.write(struct.pack("<{}I".format(len(ptrs)), *ptrs))

    # Return the file views
    return region_memory


def sizeof_regions(regions, vertex_slice, include_app_ptr=True):
    """Return the total amount of memory required to represent all the regions
    when they are padded to take a whole number of words each.
    """
    size = sum(r.sizeof_padded(vertex_slice) for r in regions if r is not None)
    if include_app_ptr:
        size += len(regions) * 4 + 4
    return size
