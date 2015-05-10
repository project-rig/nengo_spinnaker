import logging
from nengo.utils.compat import configparser

from .utils import paths

logger = logging.getLogger(__name__)

# Read nengo_spinnaker.conf files
rc_files = [paths.nengo_spinnaker_rc["system"],
            paths.nengo_spinnaker_rc["user"],
            paths.nengo_spinnaker_rc["project"]]

logger.info("Reading configuration files {!s}".format(rc_files))
rc = configparser.ConfigParser()
rc.read(rc_files)
