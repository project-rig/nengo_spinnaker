from setuptools import setup, find_packages
import sys

setup(
    name="nengo_spinnaker",
    version="0.0.1-dev",
    packages=find_packages(),

    # Metadata for PyPi
    author="Andrew Mundy",
    description="Tools for simulating neural models generated using Nengo on "
                "the SpiNNaker platform",
    license="GPLv2",

    # Requirements
    install_requires=["nengo", "rig"],
    tests_require=["pytest>=2.6", "pytest-cov", "mock"],
)
