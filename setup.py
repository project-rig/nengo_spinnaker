import io
import re
from setuptools import setup, find_packages


def read_file(filename, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")

    with io.open(filename, encoding=encoding) as f:
        return f.read()


def replace_local_hyperlinks(
        text,
        base_url="https://github.com/project-rig/nengo_spinnaker/blob/master/"
        ):
    """Replace local hyperlinks in RST with absolute addresses using the given
    base URL.

    This is used to make links in the long description function correctly
    outside of the repository (e.g. when published on PyPi).

    NOTE: This may need adjusting if further syntax is used.
    """
    def get_new_url(url):
        return base_url + url[2:]

    # Deal with anonymous URLS
    for match in re.finditer(r"^__ (?P<url>\./.*)", text, re.MULTILINE):
        orig_url = match.groupdict()["url"]
        url = get_new_url(orig_url)

        text = re.sub("^__ {}".format(orig_url),
                      "__ {}".format(url), text, flags=re.MULTILINE)

    # Deal with named URLS
    for match in re.finditer(r"^\.\. _(?P<identifier>[^:]*): (?P<url>\./.*)",
                             text, re.MULTILINE):
        identifier = match.groupdict()["identifier"]
        orig_url = match.groupdict()["url"]
        url = get_new_url(orig_url)

        text = re.sub(
            "^\.\. _{}: {}".format(identifier, orig_url),
            ".. _{}: {}".format(identifier, url),
            text, flags=re.MULTILINE)

    # Deal with image URLS
    for match in re.finditer(r"^\.\. image:: (?P<url>\./.*)",
                             text, re.MULTILINE):
        orig_url = match.groupdict()["url"]
        url = get_new_url(orig_url)

        text = text.replace(".. image:: {}".format(orig_url),
                            ".. image:: {}".format(url))

    return text

setup(
    name="nengo_spinnaker",
    version="0.2.2",
    packages=find_packages(),
    package_data={'nengo_spinnaker': ['binaries/*.aplx']},

    # Metadata for PyPi
    url="https://github.com/project-rig/nengo_spinnaker",
    author="University of Manchester and University of Waterloo",
    description="Tools for simulating neural models generated using Nengo on "
                "the SpiNNaker platform",
    long_description=replace_local_hyperlinks(read_file("README.rst")),
    license="GPLv2",
    classifiers=[
        "Development Status :: 3 - Alpha",

        "Intended Audience :: Science/Research",

        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",

        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",

        "Topic :: Scientific/Engineering",
    ],
    keywords="spinnaker nengo neural cognitive simulation",

    # Requirements
    install_requires=["nengo>=2.0.0, <3.0.0", "rig>=0.5.3, <1.0.0",
                      "bitarray>=0.8.1, <1.0.0"],
    zip_safe=False,  # Partly for performance reasons

    # Scripts
    entry_points={
        "console_scripts": [
            "nengo_spinnaker_setup = nengo_spinnaker.scripts.nengo_spinnaker_setup:main",
        ],
    }
)
