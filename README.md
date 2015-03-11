nengo_spinnaker
===============

SpiNNaker backend for Nengo


Building SpiNNaker Components
-----------------------------

To build SpiNNaker components you will require a release of the latest
SpiNNaker103 package and an appropriate ARM cross compiler toolchain.
The latest SpiNNaker103 package (when released) will be available from
[the SpiNNaker website](https://spinnaker.cs.man.ac.uk/).

If ```${SPINN103_DIR}``` is the location where you have installed the SpiNNaker
package (e.g., ```~/spinnaker_103```), and ```${NENGO_SPINNAKER}``` is the 
directory where you downloaded/cloned nengo_spinnaker, then building the
components requires that you:

```bash
# 1. Edit and source the SpiNNaker tools
$ cd ${SPINN103_DIR}
# Edit the file `spinnaker_tools/setup` as appropriate
$ source ./setup
```

```bash
# 2. Run make
$ cd ${NENGO_SPINNAKER}/spinnaker_components/
$ make
```
