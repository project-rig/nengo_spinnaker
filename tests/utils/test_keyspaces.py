import pytest
from rig.bitfield import BitField
from nengo_spinnaker.utils import keyspaces


def test_get_derived_keyspaces():
    """Test creation of derived keyspaces."""
    ks = BitField()
    ks.add_field("index")
    ks.add_field("spam")

    # General usage
    kss = keyspaces.get_derived_keyspaces(ks, (slice(5), 5, 6, 7))
    for i, x in enumerate(kss):
        assert x.index == i

    # Specify a field
    kss = keyspaces.get_derived_keyspaces(ks, slice(1, 3),
                                          field_identifier="spam")
    for x, i in zip(kss, (1, 2)):
        assert x.spam == i

    # Fail when no maximum is specified
    with pytest.raises(ValueError):
        list(keyspaces.get_derived_keyspaces(ks, (slice(None))))


def test_Keyspaces_and_is_nengo_keyspace():
    """Test the dictionary-like getter for keyspaces."""
    kss = keyspaces.KeyspaceContainer()

    default_ks = kss["nengo"]
    default_ks(connection_id=0, cluster=0, index=0)

    other_ks = kss["other"]

    assert kss.routing_tag is not None
    assert kss.filter_routing_tag is not None

    # Can easily determine what is and isn't a default keyspace
    assert keyspaces.is_nengo_keyspace(default_ks)
    assert not keyspaces.is_nengo_keyspace(other_ks)

    # Assigning fields fixes sizing and positioning
    with pytest.raises(Exception):
        other_ks.get_mask()

    kss.assign_fields()
    other_ks.get_mask()
