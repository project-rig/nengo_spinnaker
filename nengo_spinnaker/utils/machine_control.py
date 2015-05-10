import logging
from rig.machine_control.scp_connection import TimeoutError

logger = logging.getLogger(__name__)


def test_and_boot(controller, hostname, machine_width, machine_height):
    """Check if the board the controller controls is booted and boot it if it
    isn't.

    Parameters
    ----------
    hostname : string
        Hostname used to initialise the controller.
    machine_width : int
        Width of the SpiNNaker machine to boot.
    machine_height : int
        Height of the SpiNNaker machine to boot.
    """
    try:
        # Check if the board is booted
        logger.info("Checking that SpiNNaker board '{}' is "
                    "booted".format(hostname))

        sver = controller.get_software_version(0, 0)
    except TimeoutError:
        # The board isn't booted, so we try to boot
        logger.info("Booting board")
        controller.boot(machine_width, machine_height)

        # Check if the board is booted
        try:
            logger.info("Checking that SpiNNaker board '{}' is "
                        "booted".format(hostname))

            sver = controller.get_software_version(0, 0)
        except TimeoutError:
            raise Exception(
                "It appears that the SpiNNaker board \"{}\" is not "
                "connected, or has failed to boot".format(hostname)
            )

    logger.info(
        "Board is booted with {} v{:.2f}".format(sver.version_string,
                                                 sver.version))
