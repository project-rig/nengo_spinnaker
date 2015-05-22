from setuptools import setup, find_packages
import sys

setup(
    name="nengo_spinnaker",
    version="0.1.0",
    packages=find_packages(),

    # Metadata for PyPi
    author="Andrew Mundy",
    description="Tools for simulating neural models generated using Nengo on "
                "the SpiNNaker platform",
    license="GPLv2",

    # Requirements
    install_requires=["nengo>=2.0.0, <3.0.0", "rig>=0.2.2, <1.0.0"],
    zip_safe=False,  # Partly for performance reasons
)
