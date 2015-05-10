import mock
import pytest

from nengo_spinnaker.utils import application


@pytest.mark.parametrize("app_name", ["Arthur", "Robin"])
def test_get_application(app_name):
    with mock.patch.object(application, "pkg_resources") as pkg_resources:
        pkg_resources.resource_filename.return_value = "Camelot"

        # Get the application filename
        assert application.get_application(app_name) == "Camelot"

    pkg_resources.resource_filename.assert_called_once_with(
        "nengo_spinnaker", "binaries/nengo_{}.aplx".format(app_name)
    )
