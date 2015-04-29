#!/usr/bin/env python

"""Print the path of the local Nengo/SpiNNaker installation."""

if __name__=="__main__":  # pragma: no cover
    import nengo_spinnaker
    import os.path
    print(os.path.dirname(nengo_spinnaker.__file__))
