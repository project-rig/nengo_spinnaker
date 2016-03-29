"""Region utilities.
"""
import collections
from six import iteritems, iterkeys
import struct


class Args(collections.namedtuple("Args", "args, kwargs")):
    def __new__(cls, *args, **kwargs):
        return super(Args, cls).__new__(cls, args, kwargs)


def create_app_ptr_and_region_files_named(fp, regions, region_args):
    """Split up a file-like view of memory into smaller views, one per region,
    and write into the first region of memory the offsets to these later
    regions.

    Parameters
    ----------
    regions : {name: Region, ...}
        Map from keys to region objects.  The keys MUST support `int`, items
        from :py:class:`enum.IntEnum` are recommended.
    region_args : {name: (*args, **kwargs)}
        Map from keys to the arguments and keyword-arguments that should be
        used when determining the size of a region.

    Returns
    -------
    {name: file-like}
        Map from region name to file-like view of memory.
    """
    # Determine the number of entries needed in the application pointer table
    ptr_len = max(int(k) for k in iterkeys(regions)) + 1

    # Construct an empty pointer table of the correct length
    ptrs = [0] * ptr_len

    # Update the offset and then begin to allocate memory
    region_memory = dict()
    offset = ptr_len * 4  # In bytes
    for k, region in iteritems(regions):
        # Get the size of this region
        args, kwargs = region_args[k]
        region_size = region.sizeof_padded(*args, **kwargs)

        # Store the current offset as the pointer for this region
        ptrs[int(k)] = offset

        # Get the memory region and update the offset
        next_offset = offset + region_size
        region_memory[k] = fp[offset:next_offset]
        offset = next_offset

    # Write the pointer table into memory
    fp.seek(0)
    fp.write(struct.pack("<{}I".format(ptr_len), *ptrs))
    fp.seek(0)

    # Return the region memories
    return region_memory


def create_app_ptr_and_region_files(fp, regions, vertex_slice):
    """Split up a file-like view of memory into smaller views, one per region,
    and write into the first region of memory the offsets to these later
    regions.

    Returns
    -------
    [file-like view, ...]
        A file-like view of memory for each region.
    """
    # Construct an ordered dictionary of regions and additional dictionaries of
    # index and arguments.
    regions, region_args = _name_regions(regions, vertex_slice)

    # Allocate memory as before
    region_mem = create_app_ptr_and_region_files_named(
        fp, regions, region_args
    )

    # Serialise the dictionary in the correct order and return
    filelikes = [None] * max(iterkeys(region_mem))
    for i, mem in iteritems(region_mem):
        filelikes[i - 1] = mem
    return filelikes


def sizeof_regions_named(regions, region_args, include_app_ptr=True):
    """Return the total amount of memory required to represent all regions when
    padded to a whole number of words each.

    Parameters
    ----------
    regions : {name: Region, ...}
        Map from keys to region objects.  The keys MUST support `int`, items
        from :py:class:`enum.IntEnum` are recommended.
    region_args : {name: (*args, **kwargs)}
        Map from keys to the arguments and keyword-arguments that should be
        used when determining the size of a region.
    """
    if include_app_ptr:
        # Get the size of the application pointer
        size = (max(int(k) for k in iterkeys(regions)) + 1) * 4
    else:
        # Don't include the application pointer
        size = 0

    # Get the size of all the regions
    for key, region in iteritems(regions):
        # Get the arguments for the region
        args, kwargs = region_args[key]

        # Add the size of the region
        size += region.sizeof_padded(*args, **kwargs)

    return size


def sizeof_regions(regions, vertex_slice, include_app_ptr=True):
    """Return the total amount of memory required to represent all the regions
    when they are padded to take a whole number of words each.
    """
    # Get the size as before
    return sizeof_regions_named(*_name_regions(regions, vertex_slice),
                                include_app_ptr=include_app_ptr)


def _name_regions(regions, vertex_slice):
    """Convert a list of regions into the correct form for a method expecting
    two dictionaries describing the regions.
    """
    regions = collections.OrderedDict({
        i: r for i, r in enumerate(regions, start=1) if r is not None
    })
    region_args = collections.defaultdict(
        lambda: ((vertex_slice, ), {})
    )

    return regions, region_args
