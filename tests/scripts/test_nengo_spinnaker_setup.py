import pytest

from mock import Mock

import tempfile

import os

import shutil

from nengo_spinnaker.scripts.nengo_spinnaker_setup \
    import main, generate_config_file

from nengo_spinnaker.utils.paths import nengo_spinnaker_rc

from rig import wizard


def test_bad_args():
    # Should fail if more than one config file is specified
    with pytest.raises(SystemExit):
        main("--project --user".split())


def test_generate_config_file():
    # Config file generation should work straight-forwardly
    fileno, filename = tempfile.mkstemp()
    print(filename)

    generate_config_file(filename, ip_address="127.0.0.1")

    with open(filename, "r") as f:
        config = f.read()
    os.remove(filename)

    assert "[spinnaker_machine]\n" in config
    assert "hostname: 127.0.0.1\n" in config


def test_bad_main(monkeypatch):
    # Check that when questions aren't answered right, the program exits with a
    # failing status.
    mock_cli_wrapper = Mock(return_value=None)
    monkeypatch.setattr(wizard, "cli_wrapper", mock_cli_wrapper)

    assert main("-pf".split()) != 0


def test_main(monkeypatch):
    # Check that questions are asked and a config file generated
    mock_cli_wrapper = Mock(return_value={"ip_address": "127.0.0.1"})
    monkeypatch.setattr(wizard, "cli_wrapper", mock_cli_wrapper)

    # Temporarily any existing project config file out of the way
    config = nengo_spinnaker_rc["project"]
    if os.path.isfile(config):  # pragma: no cover
        _, temp = tempfile.mkstemp()
        print(config, temp)
        shutil.move(config, temp)
    else:
        temp = None

    # Create a project config file in the test runner's directory (which
    # shouldn't exist yet).
    assert main("-p".split()) == 0
    assert mock_cli_wrapper.called
    mock_cli_wrapper.reset_mock()
    assert os.path.isfile(config)

    # Should fail to create a config file when one already exists
    assert main("-p".split()) != 0
    assert not mock_cli_wrapper.called

    # ...unless forced
    assert main("-p --force".split()) == 0
    assert mock_cli_wrapper.called
    mock_cli_wrapper.reset_mock()

    # Restore the old config file
    if temp is not None:  # pragma: no cover
        shutil.move(temp, config)
    else:
        os.remove(config)
