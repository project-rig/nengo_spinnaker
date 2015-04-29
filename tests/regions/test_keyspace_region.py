import mock
import pytest
import struct
import tempfile

from rig.bitfield import BitField
from nengo_spinnaker.regions.keyspaces import (
    KeyspacesRegion, KeyField, MaskField)


@pytest.fixture
def ks():
    keyspace = BitField()
    keyspace.add_field("x", length=8, start_at=24, tags="routing")
    keyspace.add_field("y", length=8, start_at=16, tags="routing")
    keyspace.add_field("p", length=5, start_at=11, tags="routing")
    keyspace.add_field("c", length=11, start_at=0)
    keyspace.assign_fields()
    return keyspace


class TestKeyspacesRegion(object):
    @pytest.mark.parametrize(
        "key_bits, n_keys, n_fields, partitioned, vertex_slice",
        [(32, 1, 1, False, slice(0, 1)),
         ])
    def test_sizeof_no_prepends(self, key_bits, n_keys, n_fields, partitioned,
                                vertex_slice):
        # Generate the list of keys, prepends and fields
        keys = [BitField(key_bits) for _ in range(n_keys)]
        fields = [mock.Mock() for _ in range(n_fields)]

        # Create the region
        r = KeyspacesRegion(keys, fields, partitioned)

        # Determine the size
        n_atoms = (n_keys if not partitioned else
                   vertex_slice.stop - vertex_slice.start)
        assert r.sizeof(vertex_slice) == n_atoms * n_fields * 4

    def test_sizeof_with_prepends(self):
        r = KeyspacesRegion([BitField(32)], fields=[],
                            prepend_num_keyspaces=True)
        assert r.sizeof(slice(None)) == 4

    def test_sizeof_partitioned(self):
        r = KeyspacesRegion([BitField(32)]*4, fields=[mock.Mock()],
                            partitioned_by_atom=True,
                            prepend_num_keyspaces=False)
        assert r.sizeof(slice(1, 2)) == 4
        assert r.sizeof(slice(2, 4)) == 8

    def test_write_subregion_calls_fields(self):
        """Check that writing a subregion to file calls the field functions
        with each key and that any extra arguments are passed along.
        """
        # Create some keyspaces
        keys = [BitField(32) for _ in range(10)]

        # Create two fields
        fields = [mock.Mock() for _ in range(2)]
        fields[0].return_value = 0
        fields[1].return_value = 0

        # Create an UNPARTITIONED region and write out a slice, check that
        # field methods were called with EACH key and the kwargs.
        r = KeyspacesRegion(keys, fields)
        fp = tempfile.TemporaryFile()

        kwargs = {"spam": "and eggs", "green_eggs": "and ham"}
        r.write_subregion_to_file(slice(0, 1), fp, **kwargs)

        for f in fields:
            f.assert_has_calls([mock.call(k, **kwargs) for k in keys])
            f.reset_mock()

        # Create a PARTITIONED region and write out a slice, check that
        # field methods were called with EACH key IN THE SLICE and the kwargs.
        r = KeyspacesRegion(keys, fields, partitioned_by_atom=True)

        for sl in (slice(0, 1), slice(2, 5)):
            fp = tempfile.TemporaryFile()

            kwargs = {"spam": "spam spam spam", "in_a_box": "with a fox"}
            r.write_subregion_to_file(sl, fp, **kwargs)

            for f in fields:
                f.assert_has_calls([mock.call(k, **kwargs) for k in keys[sl]])
                f.reset_mock()

    def test_write_subregion_simple(self, ks):
        """A simple test that ensures the appropriate keyspace data is written
        out."""
        # Create a simple keyspace and some instances of it
        keyspaces = [ks(x=1, y=1, p=31), ks(x=3, y=7, p=2), ]

        # Add two field instances, one to get the routing key the other to get
        # the mask.
        kf = KeyField(maps={'c': 'c'})
        mf = MaskField(tag='routing')

        # Create the region
        r = KeyspacesRegion(keyspaces, fields=[kf, mf],
                            prepend_num_keyspaces=True)

        # Write out the region, then check that the data corresponds to what
        # we'd expect.
        fp = tempfile.TemporaryFile()
        r.write_subregion_to_file(slice(0, 10), fp, c=5)

        fp.seek(0)
        assert fp.read(4) == b'\x02\x00\x00\x00'  # Number of keyspaces
        assert fp.read() == struct.pack('4I',
                                        keyspaces[0](c=5).get_value(),
                                        keyspaces[0].get_mask(tag='routing'),
                                        keyspaces[1](c=5).get_value(),
                                        keyspaces[1].get_mask(tag='routing'))


class TestMaskField(object):
    """Test the MaskField object."""
    def test_mask_from_tag(self, ks):
        # Create a MaskField, configured to get the mask for tag "routing".
        mf = MaskField(tag='routing')
        assert (mf(ks, spam="eggs", arthur="King of the Britons") ==
                ks.get_mask('routing'))

    def test_mask_from_field(self, ks):
        # Create a MaskField, configured to get the mask for tag "routing".
        mf = MaskField(field='x')
        assert (mf(ks, spam="eggs", arthur="King of the Britons") ==
                ks.get_mask(field='x'))

    def test_init_fails(self, ks):
        # Create a MaskField, configured to get the mask for tag "routing".
        with pytest.raises(TypeError):
            MaskField(field='x', tag='routing')

        with pytest.raises(TypeError):
            MaskField()


class TestKeyField(object):
    """Test the KeyField object."""
    def test_key_no_fills(self, ks):
        """Check the key field when no key fields require filling in."""
        # Fill in keyfield values
        k = ks(x=1, y=2, p=17, c=33)

        # Create the field, then call the key field
        kf = KeyField()
        assert kf(k, subvertex_index=3, spam=4) == k.get_value()

    def test_key_single_fill(self, ks):
        """Check the key field when no key fields require filling in."""
        # Fill in fields we're not using
        k = ks(x=3, y=6, c=7)

        # Create the field, then call the key field
        kf = KeyField(maps={'subvertex_index': 'p'})
        assert kf(k, subvertex_index=3, spam=4) == k(p=3).get_value()

    def test_key_multiple(self, ks):
        """Check the key field when no key fields require filling in."""
        # Fill in fields we're not using
        k = ks(x=3, c=7)

        # Create the field, then call the key field
        kf = KeyField(maps={'subvertex_index': 'p', 'spam': 'y'})
        assert kf(k, subvertex_index=3, spam=4) == k(y=4, p=3).get_value()

    def test_key_limited_by_field(self, ks):
        # Fill in fields we're not using
        k = ks(x=3, c=7)

        # Create the field, then call the key field
        kf = KeyField(field='x', maps={'subvertex_index': 'p', 'spam': 'y'})
        assert (kf(k, subvertex_index=3, spam=4) ==
                k(y=4, p=3).get_value(field='x'))

    def test_key_limited_by_tag(self, ks):
        # Fill in fields we're not using
        k = ks(x=3, c=7)

        # Create the field, then call the key field
        kf = KeyField(tag='routing',
                      maps={'subvertex_index': 'p', 'spam': 'y'})
        assert (kf(k, subvertex_index=3, spam=4) ==
                k(y=4, p=3).get_value(tag='routing'))
