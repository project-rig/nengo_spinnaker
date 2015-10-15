import struct
from .region import Region


class ListRegion(Region):
    """Region which will write out a list of values."""
    def __init__(self, format_str):
        """Create a new list region.

        Parameters
        ----------
        format_str :
            Format string (see Python `struct` module) to use to format each
            item. Little-endianness is presumed.
        """
        self.format_str = format_str

    def sizeof(self, items):
        """Get the size of the list that will store the given items."""
        return struct.calcsize(self.format_str) * len(items)

    def write_subregion_to_file(self, fp, items):
        """Write the given items to file."""
        fp.write(
            struct.pack("<{}{}".format(len(items), self.format_str), *items)
        )
