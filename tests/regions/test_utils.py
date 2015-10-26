import enum
import mock
import pytest
from six import iteritems, itervalues
import struct
import tempfile

from nengo_spinnaker.regions.region import Region
from nengo_spinnaker.regions import utils


def test_create_app_ptr_and_filelike_named():
    """Test creating an application pointer table and file-like views of memory
    from named regions.
    """
    # Create a file-like object which can be the initial file that we pass in,
    # and wrap it so that creating slices of it creates objects that we can
    # interpret.
    actual_fp = tempfile.TemporaryFile()

    class Subfilelike(object):
        def __init__(self, sl):
            self.slice = sl

        def __repr__(self):
            return "Subfilelike({!r})".format(self.slice)

        def __eq__(self, other):
            return self.slice == other.slice

    def _getitem_(self, sl):
        return Subfilelike(sl)

    fp = mock.Mock(wraps=actual_fp)
    fp.__getitem__ = _getitem_

    # Now create a series of regions with different sizes.
    class MyRegion(Region):
        def __init__(self, size, expected_arg):
            self.size = size
            self.called = False
            self.expected_arg = expected_arg

        def sizeof(self, arg):
            assert arg is self.expected_arg
            self.called = True
            return self.size

    # Create the region names
    class RegionNames(enum.IntEnum):
        a = 1
        b = 2
        c = 3
        d = 4
        e = 6

    # Create all the regions, region indices and region slices
    regions = {
        RegionNames.a: MyRegion(4, mock.Mock()),  # 1 word
        RegionNames.b: MyRegion(3, mock.Mock()),  # < 1 word
        RegionNames.c: MyRegion(5, mock.Mock()),  # < 2 words
        RegionNames.d: MyRegion(100, mock.Mock()),  # 25 words
        RegionNames.e: MyRegion(32, mock.Mock()),  # 8 words
    }
    region_args = {k: ((v.expected_arg, ), {}) for k, v in iteritems(regions)}

    # Now create the application pointer table and all the sub-filelikes
    fps = utils.create_app_ptr_and_region_files_named(
        fp, regions, region_args
    )

    # Check that the size was called in all cases
    assert all(r.called for r in itervalues(regions))

    # Read back the application pointer table
    actual_fp.seek(0)
    pointer_table = struct.unpack("<7I", actual_fp.read())

    # Check that the regions are the correct size and do not overlap
    allocated_bytes = set()
    for k, filelike in iteritems(fps):
        # Check that the size is correct
        exp_size = regions[k].size
        if exp_size % 4:
            exp_size += 4 - (exp_size % 4)

        size = filelike.slice.stop - filelike.slice.start

        assert exp_size == size

        # Check that the offset in the pointer table is correct
        assert pointer_table[int(k)] == filelike.slice.start

        # Finally, ensure that this region of memory wasn't previously assigned
        used = set(range(filelike.slice.start, filelike.slice.stop))
        assert not used & allocated_bytes
        allocated_bytes |= used


@pytest.mark.parametrize("vertex_slice", [slice(0, 1), slice(100, 150)])
def test_create_app_ptr_and_filelikes(vertex_slice):
    """Test creation of an application pointer table and a series of smaller
    file-like objects which can be used with existing region objects.
    """
    # Create a file-like object which can be the initial file that we pass in,
    # and wrap it so that creating slices of it creates objects that we can
    # interpret.
    actual_fp = tempfile.TemporaryFile()

    class Subfilelike(object):
        def __init__(self, sl):
            self.slice = sl

        def __repr__(self):
            return "Subfilelike({!r})".format(self.slice)

        def __eq__(self, other):
            return self.slice == other.slice

    def _getitem_(self, sl):
        return Subfilelike(sl)

    fp = mock.Mock(wraps=actual_fp)
    fp.__getitem__ = _getitem_

    # Now create a series of regions with different sizes.
    class MyRegion(Region):
        def __init__(self, size):
            self.size = size
            self.called = False

        def sizeof(self, sl):
            assert sl == vertex_slice
            self.called = True
            return self.size

    # Create all the regions
    regions = [
        MyRegion(4),  # 1 word
        MyRegion(3),  # < 1 word
        MyRegion(5),  # < 2 words
        MyRegion(100),  # 25 words
        None,  # No region
        None,
        MyRegion(32),  # 8 words
    ]

    # Now create the application pointer table and all the sub-filelikes
    fps = utils.create_app_ptr_and_region_files(fp, regions, vertex_slice)
    assert all(r.called for r in regions if r is not None)

    expected_slices = [
        slice(32, 36),  # 8 words for the pointer table : +1 word
        slice(36, 40),  # : +1 word
        slice(40, 48),  # : +2 words
        slice(48, 148),  # : +25 words
        None,
        None,
        slice(148, 180),  # : +8 words
    ]
    expected_filelikes = [None if sl is None else Subfilelike(sl) for sl in
                          expected_slices]
    assert expected_filelikes == fps

    # Assert that the data written into the application pointer table is
    # correct.
    actual_fp.seek(0)
    assert actual_fp.read() == struct.pack(
        "<8I",
        0, *[0 if s is None else s.slice.start for s in expected_filelikes]
    )


@pytest.mark.parametrize("include_app_ptr", (True, False))
def test_sizeof_regions_named(include_app_ptr):
    """Test getting the total memory usage of some regions."""
    class MyRegion(Region):
        def __init__(self, size, expected_arg):
            self.size = size
            self.called = False
            self.expected_arg = expected_arg

        def sizeof(self, *args, **kwargs):
            assert args == (self.expected_arg, )
            self.called = True
            return self.size

    # Create the region names
    class RegionNames(enum.IntEnum):
        a = 1
        b = 2
        c = 3
        d = 4
        e = 6

    # Create all the regions
    regions = {
        RegionNames.a: MyRegion(4, mock.Mock()),  # 1 word
        RegionNames.b: MyRegion(3, mock.Mock()),  # < 1 word
        RegionNames.c: MyRegion(5, mock.Mock()),  # < 2 words
        RegionNames.d: MyRegion(100, mock.Mock()),  # 25 words
        RegionNames.e: MyRegion(32, mock.Mock()),  # 8 words
    }
    region_args = {k: ((v.expected_arg, ), {}) for k, v in iteritems(regions)}

    # Now query their size
    expected_size = 37*4 + (7*4 if include_app_ptr else 0)
    size = utils.sizeof_regions_named(regions, region_args, include_app_ptr)
    assert all(r.called for r in itervalues(regions))
    assert size == expected_size


@pytest.mark.parametrize(
    "vertex_slice, include_app_ptr",
    [(slice(0, 1), True), (slice(100, 150), False)])
def test_sizeof_regions(vertex_slice, include_app_ptr):
    """Test getting the total memory usage of some regions."""
    class MyRegion(Region):
        def __init__(self, size):
            self.size = size
            self.called = False

        def sizeof(self, sl):
            assert sl == vertex_slice
            self.called = True
            return self.size

    # Create all the regions
    regions = [
        MyRegion(4),  # 1 word
        MyRegion(3),  # < 1 word
        MyRegion(5),  # < 2 words
        MyRegion(100),  # 25 words
        None,  # No region
        MyRegion(32),  # 8 words
    ]

    # Now query their size
    assert (utils.sizeof_regions(regions, vertex_slice, include_app_ptr) ==
            37*4 + (len(regions)*4 + 4 if include_app_ptr else 0))
    assert all(r.called for r in regions if r is not None)
