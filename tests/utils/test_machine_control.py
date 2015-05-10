import mock
import pytest
from rig.machine_control import MachineController
from rig.machine_control.machine_controller import CoreInfo
from rig.machine_control.scp_connection import TimeoutError

from nengo_spinnaker.utils import machine_control


class TestTestAndBoot(object):
    def test_board_is_checked(self):
        """Test that we're happy if the board responds to an sver."""
        controller = mock.Mock(spec_set=MachineController)
        controller.get_software_version.return_value = CoreInfo(
            (0, 0), 0, 0, 1.33, 128, 0, "Test")

        machine_control.test_and_boot(controller, "spam", 2, 2)

    @pytest.mark.parametrize("width, height", [(2, 2), (8, 8)])
    def test_board_was_not_booted(self, width, height):
        """Test that the board is booted if it wasn't."""
        # Create a controller which times out on the first get_software_version
        # but which works on the second.
        controller = mock.Mock(spec_set=MachineController)
        controller.get_software_version.side_effect = [
            TimeoutError, CoreInfo((0, 0), 0, 0, 1.33, 128, 0, "Test")
        ]

        machine_control.test_and_boot(controller, "spam", width, height)

        # Check that the boot code was called correctly
        controller.boot.assert_called_once_with(width, height)

    @pytest.mark.parametrize("width, height", [(2, 2), (8, 8)])
    def test_board_not_booted_fails_to_boot(self, width, height):
        """Test that we raise an exception if booting the board is
        unsuccessful.
        """
        # Create a controller which times out on get_software_version.
        controller = mock.Mock(spec_set=MachineController)
        controller.get_software_version.side_effect = TimeoutError

        with pytest.raises(Exception) as excinfo:
            machine_control.test_and_boot(controller, "spam", width, height)
        assert "failed to boot" in str(excinfo.value)

        # Check that the boot code was called correctly
        controller.boot.assert_called_once_with(width, height)
