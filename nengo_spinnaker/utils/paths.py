import os
from nengo.utils.paths import config_dir


install_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir
))


_conf_file = "nengo_spinnaker.conf"
nengo_spinnaker_rc = {
    "system": os.path.join(install_dir, "nengo_spinnaker-data", _conf_file),
    "user": os.path.join(config_dir, _conf_file),
    "project": os.path.abspath(os.path.join(os.curdir, _conf_file)),
}
