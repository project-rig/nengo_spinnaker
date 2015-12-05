import pytest
from rig.bitfield import UnavailableFieldError

from nengo_spinnaker import netlist as nl
from nengo_spinnaker import partition as pac
from nengo_spinnaker.utils.keyspaces import KeyspaceContainer


def test_constraint():
    """Test creating a constraint."""
    # Constraints consist of: a limit, and an optional target.
    constraint = pac.Constraint(100)
    assert constraint.maximum == 100
    assert constraint.target == 1.0
    assert constraint.max_usage == 100.0

    constraint = pac.Constraint(100, 0.9)
    assert constraint.maximum == 100
    assert constraint.target == 0.9
    assert constraint.max_usage == 90.0


class TestPartition(object):
    """Test partitioning slices."""
    def test_no_partitioning(self):
        # Create the constraint
        constraint = pac.Constraint(100, 0.9)

        # Create the constraint -> usage mapping
        constraints = {constraint: lambda sl: sl.stop - sl.start + 10}

        # Perform the partitioning
        assert list(pac.partition(slice(0, 80), constraints)) == [slice(0, 80)]
        assert list(pac.partition(slice(80), constraints)) == [slice(0, 80)]

    def test_single_partition_step(self):
        # Create the constraint
        constraint_a = pac.Constraint(100, .7)
        constraint_b = pac.Constraint(50)

        # Create the constraint -> usage mapping
        constraints = {constraint_a: lambda sl: sl.stop - sl.start + 10,
                       constraint_b: lambda sl: sl.stop - sl.start}

        # Perform the partitioning
        assert list(pac.partition(slice(100), constraints)) == [
            slice(0, 50), slice(50, 100)
        ]

    def test_just_partitionable(self):
        # Create the constraint
        constraint_a = pac.Constraint(50)

        # Create the constraint -> usage mapping
        constraints = {constraint_a: lambda sl: sl.stop - sl.start + 49}

        # Perform the partitioning
        assert (list(pac.partition(slice(100), constraints)) ==
                [slice(n, n+1) for n in range(100)])  # pragma : no cover

    def test_unpartitionable(self):
        # Create the constraint
        constraint_a = pac.Constraint(50)

        # Create the constraint -> usage mapping
        constraints = {constraint_a: lambda sl: sl.stop - sl.start + 100}

        # Perform the partitioning
        with pytest.raises(pac.UnpartitionableError):
            list(pac.partition(slice(100), constraints))


class TestPartitionMultiple(object):
    """Test partitioning multiple slices simultaneously."""
    def test_no_partitioning(self):
        # Create the constraint
        constraint = pac.Constraint(100, 0.9)

        # Create the constraint -> usage mapping
        def cons(*slices):
            return sum(sl.stop - sl.start for sl in slices) + 10

        constraints = {constraint: cons}

        # Perform the partitioning
        assert (
            list(pac.partition_multiple((slice(0, 40), slice(0, 30)),
                                        constraints)) ==
            list(pac.partition_multiple((slice(40), slice(30)),
                                        constraints)) ==
            [(slice(0, 40), slice(0, 30))]
        )

    def test_single_partition_step(self):
        # Create the constraint
        constraint = pac.Constraint(50)

        # Create the constraint -> usage mapping
        def cons(*slices):
            return sum(sl.stop - sl.start for sl in slices)

        constraints = {constraint: cons}

        # Perform the partitioning
        assert (
            list(pac.partition_multiple((slice(80), slice(20)), constraints))
            == [(slice(0, 40), slice(0, 10)), (slice(40, 80), slice(10, 20))]
        )

    def test_multiple_partition_steps(self):
        # Create the constraint
        constraint = pac.Constraint(50)

        # Create the constraint -> usage mapping
        def cons(*slices):
            return sum(sl.stop - sl.start for sl in slices) + 10

        constraints = {constraint: cons}

        # Perform the partitioning
        assert (
            list(pac.partition_multiple((slice(60), slice(30)), constraints))
            == [(slice(0, 20), slice(0, 10)),
                (slice(20, 40), slice(10, 20)),
                (slice(40, 60), slice(20, 30))]
        )

    def test_unpartitionable(self):
        # Create the constraint
        constraint = pac.Constraint(50)

        # Create the constraint -> usage mapping
        def cons(*slices):
            return sum(sl.stop - sl.start for sl in slices) + 50

        constraints = {constraint: cons}

        # Perform the partitioning
        with pytest.raises(pac.UnpartitionableError):
            list(pac.partition_multiple((slice(10), slice(2)), constraints))


@pytest.mark.parametrize(
    "start, stop, n_items",
    [(0, 10, 5), (0, 10, 4), (0, 10, 6)]
)
def test_divide_slice(start, stop, n_items):
    slices = list(pac.divide_slice(slice(start, stop), n_items))
    assert slices[0].start == start
    assert slices[-1].stop == stop
    assert len(slices) == n_items
