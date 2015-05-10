import mock
import pytest
import struct
import tempfile

from nengo_spinnaker.regions.region import Region
from nengo_spinnaker.regions import utils


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
        MyRegion(32),  # 8 words
    ]

    # Now create the application pointer table and all the sub-filelikes
    fps = utils.create_app_ptr_and_region_files(fp, regions, vertex_slice)
    assert all(r.called for r in regions if r is not None)

    expected_slices = [
        slice(28, 32),  # 7 words for the pointer table : +1 word
        slice(32, 36),  # : +1 word
        slice(36, 44),  # : +2 words
        slice(44, 144),  # : +25 words
        None,
        slice(144, 176),  # : +8 words
    ]
    expected_filelikes = [None if sl is None else Subfilelike(sl) for sl in
                          expected_slices]
    assert expected_filelikes == fps

    # Assert that the data written into the application pointer table is
    # correct.
    actual_fp.seek(0)
    assert actual_fp.read() == struct.pack(
        "<7I",
        0, *[0 if s is None else s.slice.start for s in expected_filelikes]
    )


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
